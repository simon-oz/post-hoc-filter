# -*- coding: utf-8 -*-
"""
Ensemble relevance classifier:
- Keyword prefilter
- Embeddings (gemma-300m or any local embedding model path)
- Zero-shot NLI (BART or DeBERTa-v3 zero-shot)
- LLM fallback with strict prompt markers and token-level slicing
- Robust parser & configurable thresholds
- Writes JSON with "classification" inserted immediately after "url" for each result (OrderedDict)

Usage:
    python classify.py --input input.json --output output.json \
        --embed_model_path /path/to/gemma-300m \
        --nli_model_id facebook/bart-large-mnli \
        --llm_model_path /path/to/gpt-oss-model

Flags:
    --include_diagnostics  (optional) also insert a "diagnostics" dict right after "classification"

"""

import json
import re
import time
import string
import sys
import logging

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from collections import OrderedDict, defaultdict

import torch
import os

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    AutoModel, 
    AutoConfig,
    AutoModelForSequenceClassification,
    pipeline,
    PretrainedConfig
)

import torch
from contextlib import contextmanager
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

# Import the Cache class that causes the latest error
try:
    from transformers.cache_utils import DynamicCache
except ImportError:
    DynamicCache = None # Fallback if on very old transformers

# The 'MISTRAL_INPUTS_DOCSTRING' was removed in transformers >= 4.45
# We manually inject an empty string so the legacy model code finds it and doesn't crash.
# PATCH A: Fix 'MISTRAL_INPUTS_DOCSTRING' missing in transformers >= 4.45
try:
    import transformers.models.mistral.modeling_mistral
    if not hasattr(transformers.models.mistral.modeling_mistral, "MISTRAL_INPUTS_DOCSTRING"):
        transformers.models.mistral.modeling_mistral.MISTRAL_INPUTS_DOCSTRING = ""
except ImportError:
    pass

# PATCH B: Fix multiple missing attributes in Custom Configs (transformers >= 4.46)
# The NV-Embed custom config class is missing several keys required by newer transformers.
def _patched_getattr(self, key):
    # 1. Fixes crash during model init
    if key == "dtype":
        return torch.float32
    
    # 2. Fixes crash regarding Flash Attention checks
    if key == "_attn_implementation":
        return "eager"
    
    # 3. Fixes crash during weight tying (NEW FIX)
    if key == "is_encoder_decoder":
        return False
    
    # If the key is something else, raise the standard error
    raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")

# Inject the patch into the base configuration class
if not hasattr(PretrainedConfig, "__getattr__"):
    PretrainedConfig.__getattr__ = _patched_getattr
    
# PATCH C: Fix 'DynamicCache' object has no attribute 'get_usable_length'
# NV-Embed calls .get_usable_length(seq_len), but new Transformers uses .get_seq_length()
if DynamicCache is not None and not hasattr(DynamicCache, "get_usable_length"):
    def _legacy_get_usable_length(self, input_seq_len, layer_idx=None):
        # FIX: Ensure layer_idx is an integer (defaulting to 0) before calling get_seq_length.
        layer_idx = layer_idx if layer_idx is not None else 0
        return self.get_seq_length(layer_idx)
    DynamicCache.get_usable_length = _legacy_get_usable_length   


# ============================================================================
logging.basicConfig(
    level=logging.INFO, # Log messages with severity INFO or higher
    format='%(asctime)s - %(levelname)s - %(message)s' # Define the log message format
)

logger = logging.getLogger(__name__)

lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()

LLM_FALLBACK_CALLS = 0

def _bytes_to_gib(x: int) -> float:
    return x / (1024 ** 3)

@contextmanager

def _cuda_dev(i: int):
    prev = torch.cuda.current_device()
    torch.cuda.set_device(i)
    try:
        yield
    finally:
        torch.cuda.set_device(prev)

# -----------------------------
# Label normalization utilities
# -----------------------------
def normalize_label(s: Optional[str]) -> Optional[str]:
    """Map textual labels to canonical 'R', 'P', 'I'.
    Accepts 'R','P','I' or words like 'Relevant','Partial','Partially relevant','Irrelevant'.
    Returns None if not mappable.
    """
    if s is None:
        return None
    t = s.strip().lower()
    if t in {"r", "rel", "relevant"}:
        return "R"
    if t in {"p", "partially", "partial", "partially relevant"}:
        return "P"
    if t in {"i", "irr", "irrelevant"}:
        return "I"
    # Occasionally people write 'not relevant' -> treat as I
    if "not relevant" in t:
        return "I"
    return None



# -------------------------------------------------
# Inject per-item agreement and normalized reference
# -------------------------------------------------
def compare_and_mark(results: List[Dict], write_agreement: bool = True) -> None:
    """
    For each result item, normalize any reference answer and (optionally) add agreement flag.
    Recognizes keys: 'Reference answer', 'reference_answer', 'reference', case-insensitive.
    """
    for r in results:
        # find a ground-truth key (case-insensitive)
        ref_raw = None
        for k in list(r.keys()):
            if k.lower().strip() in {"reference answer", "reference_answer", "reference", "ground_truth", "label"}:
                ref_raw = r.get(k)
                break
        ref_norm = normalize_label(ref_raw)
        if ref_norm is not None:
            # store normalized reference under a stable key
            r["reference_answer_norm"] = ref_norm
            # if we have a prediction, mark agreement
            pred_norm = normalize_label(r.get("classification"))
            if write_agreement and pred_norm is not None:
                r["agreement"] = (pred_norm == ref_norm)




# ----------------------------------------
# Metrics: confusion matrix & derived stats
# ----------------------------------------
def compute_metrics(all_items: List[Dict]) -> Dict:
    """
    Compute agreement and classification metrics using items that have both
    normalized reference and predicted classification.
    Returns a dict with per-class agreement, confusion matrix, accuracy, precision,
    recall, F1, macro-F1, Cohen's kappa, and support per class.
    """
    labels = ["R", "P", "I"]
    label_to_idx = {l: i for i, l in enumerate(labels)}

    # confusion matrix counts
    cm = [[0]*3 for _ in range(3)]
    total = 0
    correct = 0

    for r in all_items:
        gt = normalize_label(r.get("reference_answer_norm") or r.get("reference answer") or r.get("reference_answer") or r.get("reference"))
        pred = normalize_label(r.get("classification"))
        if gt is None or pred is None:
            continue
        gi, pi = label_to_idx[gt], label_to_idx[pred]
        cm[gi][pi] += 1
        total += 1
        if gt == pred:
            correct += 1

    # Safeguard for division by zero
    if total == 0:
        return {
            "total_compared": 0,
            "message": "No comparable items with both reference and prediction.",
        }

    # per-class agreement (recall on each reference class)
    per_class_agreement = {}
    recalls = {}
    precisions = {}
    f1s = {}
    supports = {}

    # column sums (predicted counts) and row sums (reference counts)
    col_sums = [sum(cm[r][c] for r in range(3)) for c in range(3)]
    row_sums = [sum(cm[r]) for r in range(3)]

    
    for i, lab in enumerate(labels):
        tp = cm[i][i]
        fn = row_sums[i] - tp
        fp = col_sums[i] - tp
        tn = total - tp - fn - fp

        # Agreement per class: proportion of reference lab predicted as lab (i.e., recall)
        agreement_i = tp / row_sums[i] if row_sums[i] > 0 else 0.0
        per_class_agreement[lab] = agreement_i

        # Precision/Recall/F1
        prec = tp / col_sums[i] if col_sums[i] > 0 else 0.0
        rec = tp / row_sums[i] if row_sums[i] > 0 else 0.0
        f1  = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        precisions[lab] = prec
        recalls[lab] = rec
        f1s[lab] = f1
        supports[lab] = row_sums[i]

    # Overall accuracy
    accuracy = correct / total


    # Macro-F1
    macro_f1 = sum(f1s.values()) / 3

    # Cohen's kappa
    # p_o = accuracy
    # p_e = sum(row_sum_i * col_sum_i) / total^2
    p_e = sum(row_sums[i] * col_sums[i] for i in range(3)) / (total * total)
    kappa = (accuracy - p_e) / (1 - p_e) if (1 - p_e) > 1e-12 else 0.0

    metrics = {
        "total_compared": total,
        "overall_accuracy": accuracy,                 # overall agreement
        "per_class_agreement": per_class_agreement,   # agreement on R/P/I respectively (recalls)
        "confusion_matrix": {
            "labels": labels,
            "matrix": cm,  # rows=reference, cols=predicted
            "row_sums": row_sums,
            "col_sums": col_sums,
        },
        "precision": precisions,
        "recall": recalls,
        "f1": f1s,
        "macro_f1": macro_f1,
        "cohen_kappa": kappa,
        "support_per_class": supports,
    }
    return metrics



def pick_best_device_auto(
    llm=None,
    min_free_gb: float = 8.0,
    prefer_free_mem: bool = True
) -> int:
    """
    Returns a CUDA device index (0..N-1) with the most headroom for NLI,
    or -1 to use CPU if no GPU is suitable.

    Heuristics:
    - If an LLM (transformers model) is provided and has `hf_device_map`,
      prefer GPUs with fewer assigned layers from that map.
    - Among candidates, prefer the one with the most free memory.
    - If the best candidate has free_gb < min_free_gb, return -1 (CPU).
    """
    if not torch.cuda.is_available():
        return -1

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        return -1

    # 1) Score devices by "LLM occupancy" from hf_device_map
    assigned_counts = {i: 0 for i in range(num_gpus)}
    if llm is not None and hasattr(llm, "hf_device_map") and isinstance(llm.hf_device_map, dict):
        for module, dev in llm.hf_device_map.items():
            if isinstance(dev, int):
                assigned_counts[dev] = assigned_counts.get(dev, 0) + 1
    # 2) Gather free memory per GPU
    free_gb = {}
    for i in range(num_gpus):
        with _cuda_dev(i):
            free_b, total_b = torch.cuda.mem_get_info()
        free_gb[i] = _bytes_to_gib(free_b)

    # 3) Build candidate list: sort by (fewest assigned layers, most free mem)
    devices = list(range(num_gpus))
    devices.sort(key=lambda d: (assigned_counts[d], -free_gb[d]))

    best = devices[0] if devices else None
    if best is None:
        return -1

    # 4) Threshold check
    if free_gb[best] < min_free_gb:
        return -1  # Too little headroom, be safe and use CPU

    return best

# Optional: if prefer SentenceTransformer pooling for certain embedding models
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except Exception:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


# Set the environment variable for the current process
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"

# ------------ Thresholds & Config ------------

@dataclass
class Thresholds:
    # Embedding cosine similarity thresholds (tune with a small dev set)
    embed_R: float = 0.55
    embed_P: float = 0.35

    # NLI top-label confidence thresholds (tune per model)
    nli_R: float = 0.60     # accept "Relevant" if >= 0.60
    nli_P: float = 0.45     # accept "Partially Relevant" if >= 0.45


@dataclass
class ModelPaths:
    embed_model_path: str                 # Local path to gemma-300m embedding model (or any)
    nli_model_id: str = "facebook/bart-large-mnli"  # Or "MoritzLaurer/deberta-v3-large-zeroshot-v1"
    llm_model_path: str = "./gpt-oss-model"         # local instruction-following LLM


def normalize_text(s: str) -> str:
    s = s or ""
    return " ".join(s.split()).strip()


# ------------ Embedding Wrapper ------------

class EmbeddingModel:
    """
    Wrapper for embedding models (HuggingFace or SentenceTransformers).
    
    ARCHITECTURAL CHANGE: 
    Accepts a specific `device_id` to ensure this model stays on the 
    designated 'Utility GPU' and does not interfere with the LLM on GPU 0.
    """

    def __init__(self, model_path: str, device_id: int = 0):
        self.model_path = model_path
        self.use_st = False
        
        # CRITICAL FIX: Explicitly define the device object (e.g., "cuda:7").
        # If we just say "cuda", PyTorch defaults to index 0, causing collisions.
        self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

        # 1. Try loading as a SentenceTransformer (optimized pooling)
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                # Pass the specific device string so ST doesn't grab GPU 0
                self.st_model = SentenceTransformer(model_path, device=str(self.device))
                self.use_st = True
            except Exception:
                self.st_model = None
                self.use_st = False

        # 2. Fallback to standard HuggingFace Transformers
        if not self.use_st:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
            self.model.eval()
            # Move the model weights to the specific reserved GPU
            self.model.to(self.device)

    def encode(self, texts: List[str]) -> torch.Tensor:
        """Generates normalized embeddings for a list of texts."""
        
        # Path A: SentenceTransformer
        if self.use_st and self.st_model is not None:
            import numpy as np
            vecs = self.st_model.encode(texts, normalize_embeddings=True)
            return torch.from_numpy(np.array(vecs)).float()

        # Path B: Raw HuggingFace
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Move input tensors to self.device (e.g., GPU 7).
        # The original code used `.to("cuda")` which forces data to GPU 0,
        # causing a crash if the model is on GPU 7.
        if torch.cuda.is_available():
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Mean pooling strategy
            last_hidden = outputs.last_hidden_state  # [B, T, H]
            mask = inputs["attention_mask"].unsqueeze(-1)
            summed = (last_hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            pooled = summed / counts
            # Normalize for cosine similarity
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            return pooled.float().cpu()

    @staticmethod
    def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
        """Computes dot product (cosine similarity) between two normalized vectors."""
        return float(torch.dot(a.view(-1), b.view(-1)).item())


def embed_similarity(embed_model: EmbeddingModel, query: str, result_text: str) -> float:
    q_vec = embed_model.encode([normalize_text(query)])
    r_vec = embed_model.encode([normalize_text(result_text)])
    return EmbeddingModel.cosine(q_vec[0], r_vec[0])


def embedding_label(score: float, thr: Thresholds) -> str:
    if score >= thr.embed_R:
        return "R"
    elif score >= thr.embed_P:
        return "P"
    else:
        return "I"


# ------------ NLI Zero-shot Wrapper ------------
class ZeroShotNLI:
    """
    Zero-shot NLI with automatic device selection.
    - device_index: "auto" | -1 (CPU) | int (CUDA id)
    - llm: pass loaded LLM (so we can read its hf_device_map)
    - min_free_gb: minimum free memory required to pick a GPU; else use CPU
    - max_length: tokenizer max length to enforce real truncation
    """
    def __init__(self,
                 model_id: str = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
                 device_index="auto",
                 llm=None,
                 min_free_gb: float = 8.0,
                 max_length: int = 512):

        if device_index == "auto":
            device_index = pick_best_device_auto(llm=llm, min_free_gb=min_free_gb)

        use_gpu = (device_index >= 0 and torch.cuda.is_available())
        dtype = torch.bfloat16 if use_gpu else torch.float32

        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        tok.model_max_length = max_length  # Ensure truncation works

        mdl = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            torch_dtype=dtype
        )

        self.labels = ["Relevant", "Partially Relevant", "Irrelevant"]
        self.device_index = device_index

        self.zsc = pipeline(
            "zero-shot-classification",
            model=mdl,
            tokenizer=tok,
            device=(device_index if use_gpu else -1)
        )

    def classify(self, query: str, text: str):
        hypothesis_template = f"This example is {{}} to the query: {query}."
        out = self.zsc(text, self.labels, multi_label=False,
                       hypothesis_template=hypothesis_template)
        label = out["labels"][0]
        score = float(out["scores"][0])
        return {"Relevant": "R", "Partially Relevant": "P", "Irrelevant": "I"}[label], score


def nli_accept(label: str, score: float, thr: Thresholds) -> Optional[str]:
    if label == "R" and score >= thr.nli_R:
        return "R"
    if label == "P" and score >= thr.nli_P:
        return "P"
    if label == "I":
        # If top label is I, accept I regardless of score (NLI typically confident)
        return "I"
    return None


class LocalLLMClassifier:
    """
    Wrapper for the Large Language Model (e.g., GPT-OSS, Llama-3).
    
    ARCHITECTURAL CHANGE:
    Includes `exclude_gpus` logic. This allows us to tell the LLM:
    "Use all GPUs *except* the one I reserved for Embeddings/NLI."
    """
    def __init__(
        self,
        model_path: str,
        use_4bit: bool = False,
        use_8bit: bool = False,
        flash_attn: bool = True,
        max_mem_gib_per_gpu: int = 130,
        exclude_gpus: List[int] = None # New argument for hardware isolation
    ):
        assert torch.cuda.is_available(), "CUDA not available"
        num_gpus = torch.cuda.device_count()
        
        # If no GPUs are excluded, default to empty list
        exclude_gpus = exclude_gpus or []

        # Reduce memory fragmentation settings for PyTorch
        os.environ.setdefault("PYTORCH_ALLOC_CONF", 
                              "expandable_segments:True,max_split_size_mb:512")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=True, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # --- Hardware Isolation Strategy ---
        # We construct a memory map for the `accelerate` library.
        # If a GPU is in `exclude_gpus`, we set its allowed memory to "0GiB".
        # This forces `device_map="auto"` to skip that GPU entirely.
        max_memory = {}
        for i in range(num_gpus):
            if i in exclude_gpus:
                max_memory[i] = "0GiB" # The "Firewall": Ban LLM from this GPU
            else:
                max_memory[i] = f"{max_mem_gib_per_gpu}GiB"

        print(f"DEBUG: LLM Loading with Memory Map: {max_memory}")

        load_kwargs = dict(
            torch_dtype=torch.bfloat16,
            device_map="auto",        # Automatically shard model across allowed GPUs
            max_memory=max_memory,    # Apply the exclusion rules
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation="eager",  # use "eager" may cause OOM issue, 
            offload_state_dict=True,  # Stream weights to avoid RAM spikes
        )

        # Config adjustments (Disabling Flash Attention 2 based on your original snippet)
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        setattr(cfg, "attn_implementation", "eager") 
        
        # Quantization Setup (Optional)
        if use_4bit or use_8bit:
             bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=use_4bit,
                load_in_8bit=use_8bit,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
             load_kwargs["quantization_config"] = bnb_cfg
             load_kwargs.pop("torch_dtype", None)

        # Load the model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=cfg,
            **load_kwargs
        ).eval()

    def call(self, prompt: str, max_new_tokens: int = 8) -> str:
        """Runs inference. Automatically finds which GPU the model's first layer is on."""
        inputs = self.tokenizer(
            prompt, return_tensors="pt", padding=False, truncation=True
        )
        
        # ROBUST DEVICE PLACEMENT:
        # Since the model is sharded, we need to find where the first layer lives.
        # We cannot assume "cuda:0" because we might have excluded GPU 0.
        if hasattr(self.model, "hf_device_map"):
            # Get list of all devices used by the model
            used_devices = [v for v in self.model.hf_device_map.values() if isinstance(v, int)]
            # Pick the first one found
            target_device = torch.device(f"cuda:{used_devices[0]}") if used_devices else torch.device("cuda:0")
        else:
            target_device = self.model.device

        # Move inputs to the correct entry-point GPU
        inputs = {k: v.to(target_device) for k, v in inputs.items()}

        with torch.inference_mode():
            gen_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=False
            )

        new_tokens = gen_ids[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        

def build_llm_prompt(query: str, results: List[Dict[str, str]]) -> str:
    prompt = f"""
        You are a strict classifier. Classify each search result's relevance to the query.
        Query: "{query}"
        
        Label definitions:
        - R (Relevant): Directly addresses the query or is a core resource for it.
        - P (Partially Relevant): Overlaps with the query but is tangential, broader, or missing core aspects.
        - I (Irrelevant): Not about the query; no meaningful overlap.
        
        Return EXACTLY {len(results)} lines.
        Each line must be ONLY ONE LETTER: R, P, or I.
        Do not include any other text.
        
        Examples:
        Query: "Python machine learning libraries"
        Result: "How to Bake a Cake" -> I
        Result: "scikit-learn: Machine Learning in Python" -> R
        Result: "TensorFlow: An Open Source Machine Learning Framework" -> R
        
        RESULTS:
        """
    for i, r in enumerate(results, 1):
        title = r.get("title", "No Title")
        snippet = r.get("snippet", "No Snippet")
        prompt += f"\n{i}. Title: {title}\n   Snippet: {snippet}\n"
    prompt += "\nBEGIN_OUTPUT\n"
    return prompt


def parse_llm_output(response: str, num_results: int, strict: bool = False) -> List[str]:
    """
    Robustly parse LLM output for relevance labels.
    Accepts lines like:
      R / P / I
      R. / (R) / - R
      Relevant / Partially Relevant / Partial / Irrelevant / Not relevant / Not related

    Args:
        response: full text returned by the LLM
        num_results: how many labels to extract
        strict: if True, raise on invalid lines; if False, default-missing to 'I'

    Returns:
        List[str] of length num_results, elements in {'R','P','I'}
    """
    
    # 1) Block extraction
    start = response.find("BEGIN_OUTPUT")
    end = response.find("END_OUTPUT", start + 1)
    if start != -1:
        body = response[start + len("BEGIN_OUTPUT"): end if end != -1 else None]
    else:
        body = response

    # 2) Normalize lines
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]

    labels: List[str] = []

    # Patterns:
    # - Single letter R/P/I with optional wrappers/punctuations: "R", "R.", "(R)", "[R]", "- R"
    single_letter_pat = re.compile(r'^[\-\(\[\{ ]*\b([RPI])\b[ \)\]\}\.\,]*$', re.IGNORECASE)

    def map_word_line(c: str) -> str:
        c = c.strip().lower()
        # Single letter already handled; below are word mappings
        if c.startswith("rel"):                     # relevant, relevance, etc.
            return "R"
        if c.startswith("par"):                     # partial, partially relevant
            return "P"
        if c.startswith("irr"):                     # irrelevant, irrelevance
            return "I"
        if c in {"partial", "partially"}:
            return "P"
        # Negations → Irrelevant
        if c.startswith("not rel") or "not relevant" in c or "not related" in c:
            return "I"
        # Some multilingual hints (optional, extend as needed)
        if c in {"relevante", "pertinent"}:        # e.g., French/Spanish synonyms
            return "R"
        if c in {"irrelevante", "irrélevant"}:
            return "I"
        return ""

    for ln in lines:
        # 3) Try single-letter with punctuation
        m = single_letter_pat.match(ln)
        if m:
            labels.append(m.group(1).upper())
        else:
            # 4) Try word mapping
            mapped = map_word_line(ln)
            if mapped:
                labels.append(mapped)
            else:
                # If strict, may raise or log here
                if strict:
                    raise ValueError(f"Unrecognized label line: {ln!r}")
                # else ignore and continue; we'll pad later

        if len(labels) == num_results:
            break

    # 5) Pad with 'I' to ensure length
    while len(labels) < num_results:
        labels.append("I")

    return labels
    

# ------------ Keyword Prefilter ------------

def keyword_prefilter(query: str, text: str, min_len: int = 3) -> bool:
    """
    Improved keyword prefilter:
    - Removes all punctuation from both query and text.
    - Tokenizes and stems both.
    - Returns True if any stemmed query token (len >= min_len) appears in the stemmed text.
    """
    ps = PorterStemmer()

    # Remove punctuation and normalize whitespace ---
    def normalize(s: str) -> str:
        s = s.lower()
        s = s.translate(str.maketrans("", "", string.punctuation))  # remove all punctuation
        s = re.sub(r"\s+", " ", s).strip()  # collapse multiple spaces
        return s

    query_clean = normalize(query)
    text_clean = normalize(text)

    # --- Tokenize and stem ---
    query_tokens = [ps.stem(tok) for tok in query_clean.split() if len(tok) >= min_len]
    text_tokens = [ps.stem(tok) for tok in text_clean.split()]

    # --- Check for overlap ---
    return any(qtok in text_tokens for qtok in query_tokens)


# ------------ Ensemble Logic ------------

def ensemble_decision(
    query: str,
    result_text: str,
    embed_model: EmbeddingModel,
    nli_model: ZeroShotNLI,
    llm: Optional[LocalLLMClassifier],
    thr: Thresholds
) -> Tuple[str, Dict[str, float], str]:
    """
    Determine final classification label for a single search result using
    multiple models (keyword filter, embedding similarity, NLI, and optional LLM fallback).
    
    Returns (final_label, diagnostics) for one result.
    explanation (str): Textual justification of why the final label was chosen
    
    Policy:
    1) Keyword prefilter: if false -> I
    2) Embedding similarity -> R/P/I (thresholds)
    3) NLI zero-shot -> label + score; accept if above thresholds
    4) If disagreement/borderline -> LLM fallback over the single item
    5) Conservative combination to avoid false "R"
    
    """
    
    global LLM_FALLBACK_CALLS

    text = normalize_text(result_text)
    diag = {}

    # Keyword prefilter
    if not keyword_prefilter(query, text):
        diag["prefilter"] = 0.0
        explanation = "No query keywords detected in the text; marked as Irrelevant."  # Added explanation
        return "I", diag, explanation
    diag["prefilter"] = 1.0

    s_time = time.time()
    # Embedding similarity
    s_embed = embed_similarity(embed_model, query, text)
    diag["embed_score"] = s_embed
    embed_lbl = embedding_label(s_embed, thr)
    s_embed_time_et = time.time()
    logger.info("s_embed time: %.4f", (s_embed_time_et - s_time))
    
    # NLI zero-shot
    nli_lbl, nli_score = nli_model.classify(query, text)
    diag["nli_score"] = nli_score
    diag["nli_label"] = nli_lbl
    nli_accepted = nli_accept(nli_lbl, nli_score, thr)
    nli_time_e = time.time()
    logger.info("nli time: %.4f", nli_time_e - s_embed_time_et)

    # Agreement logic between embedding and NLI outputs
    if embed_lbl == "R" and nli_accepted == "R":
        explanation = "High embedding similarity and NLI both confirm relevance."
        return "R", diag, explanation
        
    if embed_lbl == "I" and nli_accepted == "I":
        explanation = "Embedding and NLI both indicate irrelevance."
        return "I", diag, explanation

    # Disagreements → Partial
    if (embed_lbl == "R" and nli_accepted == "I") or (embed_lbl == "I" and nli_accepted == "R"):
        explanation = f"Embedding and NLI disagree — one relevant, one irrelevant — marked as partially relevant. embed_lbl: {embed_lbl}; nli_accepted: {nli_accepted}"
        return "P", diag, explanation

    # Both P
    if embed_lbl == "P" and nli_accepted == "P":
        explanation = "Both models give partial signals of relevance."
        return "P", diag, explanation

    # If NLI accepts I, trust I unless embeddings are *much* higher than R threshold
    if nli_accepted == "I":
        if s_embed >= (thr.embed_R + 0.10):
            explanation = f"NLI judged as Irrelevant but embeddings show moderate similarity, downgraded to Partial. s_embed ({s_embed}) >= embed_R ({embed_R})" 
            return "P", diag, explanation
        explanation = f"NLI predicts Irrelevant and embedding score is low. s_embed: {s_embed}" 
        return "I", diag, explanation

    # If NLI accepts R but embeddings are P/I, require stronger NLI confidence for R
    if nli_accepted == "R":
        if s_embed >= thr.embed_P and nli_score >= (thr.nli_R + 0.10):
            explanation = f"Strong NLI confidence and adequate embedding similarity confirm relevance. s_embed: {s_embed}; nli_score: {nli_score} > nli_R"
            return "R", diag, explanation
        explanation = f"NLI predicts Relevant but embedding similarity only partial — downgraded to Partial. s_embed: {s_embed}"
        return "P", diag, explanation

    # Borderline or None → LLM fallback (strict)
    if llm is not None:
        LLM_FALLBACK_CALLS += 1
        prompt = build_llm_prompt(query, [{"title": "", "snippet": text}])
        if len(prompt) > 30000:   # characters ≈ 8k tokens
            prompt = prompt[-30000:]
            
        resp = llm.call(prompt) + "\nEND_OUTPUT"
        lbl = parse_llm_output(resp, 1)[0]

        llm_time_e = time.time()
        logger.info("LLM time: %.4f", llm_time_e - nli_time_e)
        explanation = "LLM fallback used due to uncertain or conflicting model outputs." 
        return lbl, diag, explanation

    # Default conservative outcome
    explanation = "Models uncertain; conservatively assigned as Partial."
    return "P", diag, explanation


# ------------ JSON Writing: insert classification after "url" ------------

def ordered_result_with_classification(
    result: Dict[str, str],
    classification: str,
    include_diagnostics: bool,
    diagnostics: Optional[Dict[str, float]] = None,
    explanation: str = None
) -> OrderedDict:
    """
    Build an OrderedDict with keys ordered so that "classification" appears
    immediately after "url".
    Default order: title, snippet, url, classification, [diagnostics?]
    """
    od = OrderedDict()
    # Preserve common human-friendly order
    if "title" in result:
        od["title"] = result["title"]
    if "snippet" in result:
        od["snippet"] = result["snippet"]
    if "url" in result:
        od["url"] = result["url"]
    # Insert classification right after url
    od["classification"] = classification
    if include_diagnostics and diagnostics is not None:
        od["diagnostics"] = diagnostics
    if explanation is not None:                # Add explanation if provided
        od["explanation"] = explanation    

    # Include any extra keys from original result (but avoid duplicating)
    for k, v in result.items():
        if k not in od:
            od[k] = v
    return od


# ------------ File-level API ------------

def classify_search_results_from_file(
    input_file: str,
    output_file: str,
    paths: ModelPaths,
    thr: Thresholds = Thresholds(),
    use_llm_fallback: bool = True,
    include_diagnostics: bool = False,
    metrics_out_file: Optional[str] = None,
    write_agreement: bool = True
) -> None:
    """
    Main execution loop.
    
    STRATEGY:
    1. Identify available GPUs.
    2. Reserve the LAST GPU for 'Utility Models' (Embedding + NLI).
    3. Reserve the REMAINING GPUs (0 to N-2) for the 'LLM'.
    4. Initialize models with these constraints to prevent OOM.
    """
    
    # --- Step 1: Define Hardware Strategy ---
    num_gpus = torch.cuda.device_count()
    
    # If we have 8 GPUs, indices are 0..7.
    # We grab the last one (Index 7) for small models.
    utility_gpu_id = num_gpus - 1 if num_gpus > 1 else 0
    
    # We tell the LLM to exclude that specific GPU ID.
    llm_exclude_gpus = [utility_gpu_id] if num_gpus > 1 else []

    logger.info(f"Hardware Strategy: Embed/NLI on GPU {utility_gpu_id}. LLM on GPUs 0-{utility_gpu_id-1}.")

    # --- Step 2: Initialize Models ---

    # A. Embeddings: Forced to Utility GPU
    embed_model = EmbeddingModel(paths.embed_model_path, device_id=utility_gpu_id)

    # B. LLM: Forced to everything EXCEPT Utility GPU
    llm = None
    if use_llm_fallback:
        llm = LocalLLMClassifier(
            paths.llm_model_path, 
            exclude_gpus=llm_exclude_gpus
        )

    # C. NLI: Forced to Utility GPU
    # Note: We skip 'auto' placement and force it to sit next to the embeddings.
    nli_model = ZeroShotNLI(
        paths.nli_model_id,
        device_index=utility_gpu_id, 
        max_length=512
    )

    # --- Step 3: Process Data ---
    start_time = time.time()
    
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    out_data = []
    for entry in data:
        query = entry.get("query", "")
        results = entry.get("results", []) or []

        new_results: List[OrderedDict] = []
        for r in results:
            # Construct text for classification
            title = r.get("title", "")
            snippet = r.get("snippet", "")
            text = f"{title}. {snippet}".strip()

            # Get Classification (R/P/I) from Ensemble
            final_lbl, diag, explanation = ensemble_decision(
                query, text, embed_model, nli_model, llm, thr)

            # Format Output
            new_r = ordered_result_with_classification(
                result=r,
                classification=final_lbl,
                include_diagnostics=include_diagnostics,
                diagnostics=diag if include_diagnostics else None,
                explanation=explanation
            )
            new_results.append(new_r)

        # Compare against Ground Truth if available
        compare_and_mark(new_results, write_agreement=write_agreement)
        
        out_entry = OrderedDict()
        out_entry["query"] = query
        out_entry["results"] = new_results
        out_data.append(out_entry)

    # --- Step 4: Save and Wrap up ---
    end_time = time.time()
    print(f"Running time: {end_time - start_time:.4f}")
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)

    # Calculate metrics (Precision/Recall/F1)
    flat_items: List[Dict] = []
    for entry in out_data:
        flat_items.extend(entry["results"])
    summary_metrics = compute_metrics(flat_items)

    if metrics_out_file:
        with open(metrics_out_file, "w", encoding="utf-8") as f:
            json.dump(summary_metrics, f, indent=2, ensure_ascii=False)
    else:
        # If no specific metrics file, append to main output object (optional structure)
        summary_obj = {"summary_metrics": summary_metrics, "data": out_data}
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(summary_obj, f, indent=2, ensure_ascii=False)
            

# ------------ CLI ------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ensemble relevance classifier with ordered JSON output")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file")
    parser.add_argument("--embed_model_path", type=str, required=True,
                        help="Local path to embedding model (e.g., gemma-300m)")
    parser.add_argument("--nli_model_id", type=str, default="facebook/bart-large-mnli",
                        help="NLI model id: 'facebook/bart-large-mnli' or 'MoritzLaurer/deberta-v3-large-zeroshot-v1'")
    parser.add_argument("--llm_model_path", type=str, default="./gpt-oss-model",
                        help="Local LLM path for fallback classification")
    parser.add_argument("--no_llm_fallback", action="store_true",
                        help="Disable LLM fallback")
    parser.add_argument("--include_diagnostics", action="store_true",
                        help="Include diagnostics after 'classification'")

    args = parser.parse_args()

    paths = ModelPaths(
        embed_model_path=args.embed_model_path,
        nli_model_id=args.nli_model_id,
        llm_model_path=args.llm_model_path,
    )

    classify_search_results_from_file(
        input_file=args.input,
        output_file=args.output,
        paths=paths,
        use_llm_fallback=not args.no_llm_fallback,
        include_diagnostics=args.include_diagnostics
    )
    
    print(f"LLM fallback calls: {LLM_FALLBACK_CALLS}")
