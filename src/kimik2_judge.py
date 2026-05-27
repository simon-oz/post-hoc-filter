"""
/home/jovyan/nfs_share/models/huggingface/hub/kimi/kimi-k2-thinking/modeling_deepseek.py
Open the file and search for the text: seen_tokens It will be inside a function called prepare_inputs_for_generation (around line 1654).
# OLD (Bad)
past_length = past_key_values.seen_tokens
# NEW (Fixed)
past_length = past_key_values.get_seq_length()
"""
import json
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Configuration ---
INPUT_FILE = '/home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/ori_input_data.json'  # Replace with your actual file name
OUTPUT_FILE = '/home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/input_data.json'
# Replace with the actual path or HuggingFace ID for Kimi-K2
# Note: Kimi-K2 is huge. Ensure you are using the correct quantized version if needed.
MODEL_PATH = "/home/jovyan/nfs_share/models/huggingface/hub/kimi/kimi-k2-thinking" 

# --- 1. Load Model & Tokenizer ---
def load_local_model(model_path):
    print(f"Loading model from: {model_path}...")
    print("This may take a while depending on model size...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=True
    )
    
    # device_map="auto" will automatically spread the model across your GPUs
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto", 
        torch_dtype=torch.float16, # or torch.bfloat16 if your GPUs support it (Ampere+)
        trust_remote_code=True,
        # attn_implementation="flash_attention_2" # Uncomment if you have flash-attn installed
    )
    
    return model, tokenizer

# --- 2. Inference Function ---
def get_judgment(model, tokenizer, query, content):
    """
    Constructs the prompt and runs inference locally on the GPUs.
    """
    
    # Specialized prompt to force the model to output a single character
    prompt = f"""User: You are a relevance judge.
Task: Determine if the Content is Relevant (R), Partially Relevant (P), or Irrelevant (I) to the Query.
Query: "{query}"
Content: "{content}"

Instructions:
1. Analyze the relationship between the query and the content.
2. Output ONLY one single letter: R, P, or I. Do not output any other text.

Model:"""

    # If the model uses a chat template, use that instead (Uncomment below if needed)
    # messages = [{"role": "user", "content": prompt}]
    # text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    text_input = prompt 

    inputs = tokenizer(text_input, return_tensors="pt").to(model.device)

    # Generate output
    # max_new_tokens is low because we only expect a single letter
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=10, 
            temperature=0.1, # Low temp for deterministic output
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            # use_cache=False
        )
    
    # Decode and clean the output
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    # Cleaning logic: Extract the first occurrence of R, P, or I
    cleaned_response = response.strip().upper()
    
    # Fallback parsing if the model chats (e.g., "The answer is R")
    if len(cleaned_response) > 1:
        for char in cleaned_response:
            if char in ['R', 'P', 'I']:
                return char
        return "I" # Default fallback if completely confused
        
    return cleaned_response if cleaned_response in ['R', 'P', 'I'] else "I"

# --- 3. Main Processing Logic ---
def main():
    # Load Model
    model, tokenizer = load_local_model(MODEL_PATH)
    
    # Load Data
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {INPUT_FILE}")
        return

    total_comparisons = 0
    matches = 0
    
    print("\nStarting Inference...")
    
    for item in data:
        query = item.get("query", "")
        print(f"Processing Query: {query}")
        
        for result in item.get("results", []):
            title = result.get("title", "")
            snippet = result.get("snippet", "")
            reference = result.get("reference answer", "")
            
            # Concatenate Content
            full_content = f"{title} {snippet}"
            
            # Get Prediction
            prediction = get_judgment(model, tokenizer, query, full_content)
            
            # Save Result
            result["kimik2_rel"] = prediction
            
            # Update Stats
            total_comparisons += 1
            if prediction == reference:
                matches += 1
            else:
                print(f"  [Mismatch] Ref: {reference} vs Kimi: {prediction} | Content: {title[:30]}...")

    # Calculate Statistics
    agreement_rate = (matches / total_comparisons * 100) if total_comparisons > 0 else 0
    
    # Save Output
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        
    print("\n" + "="*40)
    print(f"Processing Complete.")
    print(f"Overall Agreement Rate: {agreement_rate:.2f}%")
    print(f"Results saved to: {OUTPUT_FILE}")
    print("="*40)

if __name__ == "__main__":
    main()

