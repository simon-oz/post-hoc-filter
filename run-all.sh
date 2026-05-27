python -m src.rel_classify \
--input /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/input_data.json \
--output /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/output_qwen8b-gptoss120b.json \
--embed_model_path /home/jovyan/nfs_share/models/huggingface/hub/qwen/qwen_embed_8b \
--nli_model_id /home/jovyan/nfs_share/models/huggingface/hub/MoritzLaurer/deberta-v3-large-zeroshot-v2.0 \
--llm_model_path /home/jovyan/nfs_share/models/huggingface/hub/openai/gpt-oss-120b 

python -m src.rel_classify \
--input /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/input_data.json \
--output /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/output_qwen4b-gptoss120b.json \
--embed_model_path /home/jovyan/nfs_share/models/huggingface/hub/qwen/qwen_embed_4b \
--nli_model_id /home/jovyan/nfs_share/models/huggingface/hub/MoritzLaurer/deberta-v3-large-zeroshot-v2.0 \
--llm_model_path /home/jovyan/nfs_share/models/huggingface/hub/openai/gpt-oss-120b 

python -m src.rel_classify \
--input /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/input_data.json \
--output /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/output_qwen06b-gptoss120b.json \
--embed_model_path /home/jovyan/nfs_share/models/huggingface/hub/qwen/qwen_embed_06b \
--nli_model_id /home/jovyan/nfs_share/models/huggingface/hub/MoritzLaurer/deberta-v3-large-zeroshot-v2.0 \
--llm_model_path /home/jovyan/nfs_share/models/huggingface/hub/openai/gpt-oss-120b 

python -m src.rel_classify \
--input /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/input_data.json \
--output /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/output_nv_embed7b-gptoss120b.json \
--embed_model_path /home/jovyan/nfs_share/models/huggingface/hub/NV-Embed \
--nli_model_id /home/jovyan/nfs_share/models/huggingface/hub/MoritzLaurer/deberta-v3-large-zeroshot-v2.0 \
--llm_model_path /home/jovyan/nfs_share/models/huggingface/hub/openai/gpt-oss-120b 

python -m src.rel_classify \
--input /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/input_data.json \
--output /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/output_mpnet-v2-gptoss120b.json \
--embed_model_path /home/jovyan/nfs_share/models/huggingface/hub/all-mpnet-base-v2 \
--nli_model_id /home/jovyan/nfs_share/models/huggingface/hub/MoritzLaurer/deberta-v3-large-zeroshot-v2.0 \
--llm_model_path /home/jovyan/nfs_share/models/huggingface/hub/openai/gpt-oss-120b 

python -m src.rel_classify \
--input /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/input_data.json \
--output /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/output_gemma300m-gptoss120b.json \
--embed_model_path /home/jovyan/nfs_share/models/huggingface/hub/google/embeddinggemma-300m \
--nli_model_id /home/jovyan/nfs_share/models/huggingface/hub/MoritzLaurer/deberta-v3-large-zeroshot-v2.0 \
--llm_model_path /home/jovyan/nfs_share/models/huggingface/hub/openai/gpt-oss-120b 

python -m src.rel_classify \
--input /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/input_data.json \
--output /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/output_nemotron8b-gptoss120b.json \
--embed_model_path /home/jovyan/nfs_share/models/huggingface/hub/nvidia/llama-embed-nemotron-8b \
--nli_model_id /home/jovyan/nfs_share/models/huggingface/hub/MoritzLaurer/deberta-v3-large-zeroshot-v2.0 \
--llm_model_path /home/jovyan/nfs_share/models/huggingface/hub/openai/gpt-oss-120b 

python -m src.rel_classify \
--input /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/input_data.json \
--output /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/output_qwen8b-gptoss20b.json \
--embed_model_path /home/jovyan/nfs_share/models/huggingface/hub/qwen/qwen_embed_8b \
--nli_model_id /home/jovyan/nfs_share/models/huggingface/hub/MoritzLaurer/deberta-v3-large-zeroshot-v2.0 \
--llm_model_path /home/jovyan/nfs_share/models/huggingface/hub/openai/gpt-oss-20b 

python -m src.rel_classify \
--input /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/input_data.json \
--output /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/output_qwen4b-gptoss20b.json \
--embed_model_path /home/jovyan/nfs_share/models/huggingface/hub/qwen/qwen_embed_4b \
--nli_model_id /home/jovyan/nfs_share/models/huggingface/hub/MoritzLaurer/deberta-v3-large-zeroshot-v2.0 \
--llm_model_path /home/jovyan/nfs_share/models/huggingface/hub/openai/gpt-oss-20b 

python -m src.rel_classify \
--input /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/input_data.json \
--output /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/output_qwen06b-gptoss20b.json \
--embed_model_path /home/jovyan/nfs_share/models/huggingface/hub/qwen/qwen_embed_06b \
--nli_model_id /home/jovyan/nfs_share/models/huggingface/hub/MoritzLaurer/deberta-v3-large-zeroshot-v2.0 \
--llm_model_path /home/jovyan/nfs_share/models/huggingface/hub/openai/gpt-oss-20b 


python -m src.rel_classify \
--input /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/input_data.json \
--output /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/output_nv_embed7b-gptoss20b.json \
--embed_model_path /home/jovyan/nfs_share/models/huggingface/hub/NV-Embed \
--nli_model_id /home/jovyan/nfs_share/models/huggingface/hub/MoritzLaurer/deberta-v3-large-zeroshot-v2.0 \
--llm_model_path /home/jovyan/nfs_share/models/huggingface/hub/openai/gpt-oss-20b 

python -m src.rel_classify \
--input /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/input_data.json \
--output /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/output_mpnet-v2-gptoss120b.json \
--embed_model_path /home/jovyan/nfs_share/models/huggingface/hub/all-mpnet-base-v2 \
--nli_model_id /home/jovyan/nfs_share/models/huggingface/hub/MoritzLaurer/deberta-v3-large-zeroshot-v2.0 \
--llm_model_path /home/jovyan/nfs_share/models/huggingface/hub/openai/gpt-oss-20b 

python -m src.rel_classify \
--input /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/input_data.json \
--output /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/output_gemma300m-gptoss20b.json \
--embed_model_path /home/jovyan/nfs_share/models/huggingface/hub/google/embeddinggemma-300m \
--nli_model_id /home/jovyan/nfs_share/models/huggingface/hub/microsoft/deberta-v3-large \
--llm_model_path /home/jovyan/nfs_share/models/huggingface/hub/openai/gpt-oss-20b 

python -m src.rel_classify \
--input /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/input_data.json \
--output /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/output_nemotron_8b-gptoss20b.json \
--embed_model_path /home/jovyan/nfs_share/models/huggingface/hub/nvidia/llama-embed-nemotron-8b \
--nli_model_id /home/jovyan/nfs_share/models/huggingface/hub/microsoft/deberta-v3-large \
--llm_model_path /home/jovyan/nfs_share/models/huggingface/hub/openai/gpt-oss-20b 



python -m src.rel_classify \
--input /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/input_data.json \
--output /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/output_qwen8b-qwen3-235b.json \
--embed_model_path /home/jovyan/nfs_share/models/huggingface/hub/qwen/qwen_embed_8b \
--nli_model_id /home/jovyan/nfs_share/models/huggingface/hub/microsoft/deberta-v3-large \
--llm_model_path /home/jovyan/nfs_share/models/huggingface/hub/qwen/qwen3-235b-a22b-inst-2507

python -m src.rel_classify \
--input /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/input_data.json \
--output /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/output_qwen4b-qwen3-235b.json \
--embed_model_path /home/jovyan/nfs_share/models/huggingface/hub/qwen/qwen_embed_4b \
--nli_model_id /home/jovyan/nfs_share/models/huggingface/hub/microsoft/deberta-v3-large \
--llm_model_path /home/jovyan/nfs_share/models/huggingface/hub/qwen/qwen3-235b-a22b-inst-2507


python -m src.rel_classify \
--input /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/input_data.json \
--output /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/output_qwen06b-qwen3-235b.json \
--embed_model_path /home/jovyan/nfs_share/models/huggingface/hub/qwen/qwen_embed_06b \
--nli_model_id /home/jovyan/nfs_share/models/huggingface/hub/microsoft/deberta-v3-large \
--llm_model_path /home/jovyan/nfs_share/models/huggingface/hub/qwen/qwen3-235b-a22b-inst-2507


python -m src.rel_classify \
--input /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/input_data.json \
--output /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/output_nv_embed7b-qwen3-235b.json \
--embed_model_path /home/jovyan/nfs_share/models/huggingface/hub/NV-Embed \
--nli_model_id /home/jovyan/nfs_share/models/huggingface/hub/MoritzLaurer/deberta-v3-large-zeroshot-v2.0 \
--llm_model_path /home/jovyan/nfs_share/models/huggingface/hub/qwen/qwen3-235b-a22b-inst-2507 

python -m src.rel_classify \
--input /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/input_data.json \
--output /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/output_mpnet-v2-qwen3-235b.json \
--embed_model_path /home/jovyan/nfs_share/models/huggingface/hub/all-mpnet-base-v2 \
--nli_model_id /home/jovyan/nfs_share/models/huggingface/hub/MoritzLaurer/deberta-v3-large-zeroshot-v2.0 \
--llm_model_path /home/jovyan/nfs_share/models/huggingface/hub/qwen/qwen3-235b-a22b-inst-2507


python -m src.rel_classify \
--input /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/input_data.json \
--output /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/output_gemma300m-qwen3-235b.json \
--embed_model_path /home/jovyan/nfs_share/models/huggingface/hub/google/embeddinggemma-300m \
--nli_model_id /home/jovyan/nfs_share/models/huggingface/hub/microsoft/deberta-v3-large \
--llm_model_path /home/jovyan/nfs_share/models/huggingface/hub/qwen/qwen3-235b-a22b-inst-2507


python -m src.rel_classify \
--input /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/input_data.json \
--output /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/output_nemotron_8b-qwen3-235b.json \
--embed_model_path /home/jovyan/nfs_share/models/huggingface/hub/nvidia/llama-embed-nemotron-8b \
--nli_model_id /home/jovyan/nfs_share/models/huggingface/hub/microsoft/deberta-v3-large \
--llm_model_path /home/jovyan/nfs_share/models/huggingface/hub/qwen/qwen3-235b-a22b-inst-2507


python -m src.rel_classify \
--input /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/input_data.json \
--output /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/output_qwen8b-Qwen25-7b.json \
--embed_model_path /home/jovyan/nfs_share/models/huggingface/hub/qwen/qwen_embed_8b \
--nli_model_id /home/jovyan/nfs_share/models/huggingface/hub/microsoft/deberta-v3-large \
--llm_model_path /home/jovyan/nfs_share/models/huggingface/hub/models--Qwen-Qwen2.5-Instruct


python -m src.rel_classify \
--input /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/input_data.json \
--output /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/output_qwen4b-Qwen25-7b.json \
--embed_model_path /home/jovyan/nfs_share/models/huggingface/hub/qwen/qwen_embed_4b \
--nli_model_id /home/jovyan/nfs_share/models/huggingface/hub/microsoft/deberta-v3-large \
--llm_model_path /home/jovyan/nfs_share/models/huggingface/hub/models--Qwen-Qwen2.5-Instruct

python -m src.rel_classify \
--input /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/input_data.json \
--output /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/output_qwen06b-Qwen25-7b.json \
--embed_model_path /home/jovyan/nfs_share/models/huggingface/hub/qwen/qwen_embed_06b \
--nli_model_id /home/jovyan/nfs_share/models/huggingface/hub/microsoft/deberta-v3-large \
--llm_model_path /home/jovyan/nfs_share/models/huggingface/hub/models--Qwen-Qwen2.5-Instruct

python -m src.rel_classify \
--input /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/input_data.json \
--output /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/output_nv_embed7b-qwen25-7b.json \
--embed_model_path /home/jovyan/nfs_share/models/huggingface/hub/NV-Embed \
--nli_model_id /home/jovyan/nfs_share/models/huggingface/hub/MoritzLaurer/deberta-v3-large-zeroshot-v2.0 \
--llm_model_path /home/jovyan/nfs_share/models/huggingface/hub/models--Qwen-Qwen2.5-Instruct 

python -m src.rel_classify \
--input /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/input_data.json \
--output /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/output_mpnet-v2-qwen25-7b.json \
--embed_model_path /home/jovyan/nfs_share/models/huggingface/hub/all-mpnet-base-v2 \
--nli_model_id /home/jovyan/nfs_share/models/huggingface/hub/MoritzLaurer/deberta-v3-large-zeroshot-v2.0 \
--llm_model_path /home/jovyan/nfs_share/models/huggingface/hub/models--Qwen-Qwen2.5-Instruct

python -m src.rel_classify \
--input /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/input_data.json \
--output /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/output_gemma300m-Qwen25-7b.json \
--embed_model_path /home/jovyan/nfs_share/models/huggingface/hub/google/embeddinggemma-300m \
--nli_model_id /home/jovyan/nfs_share/models/huggingface/hub/microsoft/deberta-v3-large \
--llm_model_path /home/jovyan/nfs_share/models/huggingface/hub/models--Qwen-Qwen2.5-Instruct


python -m src.rel_classify \
--input /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/input_data.json \
--output /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/output_nemotron_8b-Qwen25-7b.json \
--embed_model_path /home/jovyan/nfs_share/models/huggingface/hub/nvidia/llama-embed-nemotron-8b \
--nli_model_id /home/jovyan/nfs_share/models/huggingface/hub/microsoft/deberta-v3-large \
--llm_model_path /home/jovyan/nfs_share/models/huggingface/hub/models--Qwen-Qwen2.5-Instruct


python -m src.rel_classify \
--input /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/input_data.json \
--output /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/output_qwen8b-Qwen3-30b.json \
--embed_model_path /home/jovyan/nfs_share/models/huggingface/hub/qwen/qwen_embed_8b \
--nli_model_id /home/jovyan/nfs_share/models/huggingface/hub/microsoft/deberta-v3-large \
--llm_model_path /home/jovyan/nfs_share/models/huggingface/hub/qwen/Qwen3-30B-A3B-Instruct-2507



python -m src.rel_classify \
--input /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/input_data.json \
--output /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/output_qwen4b-Qwen3-30b.json \
--embed_model_path /home/jovyan/nfs_share/models/huggingface/hub/qwen/qwen_embed_4b \
--nli_model_id /home/jovyan/nfs_share/models/huggingface/hub/microsoft/deberta-v3-large \
--llm_model_path /home/jovyan/nfs_share/models/huggingface/hub/qwen/Qwen3-30B-A3B-Instruct-2507


python -m src.rel_classify \
--input /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/input_data.json \
--output /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/output_qwen06b-Qwen3-30b.json \
--embed_model_path /home/jovyan/nfs_share/models/huggingface/hub/qwen/qwen_embed_06b \
--nli_model_id /home/jovyan/nfs_share/models/huggingface/hub/microsoft/deberta-v3-large \
--llm_model_path /home/jovyan/nfs_share/models/huggingface/hub/qwen/Qwen3-30B-A3B-Instruct-2507

python -m src.rel_classify \
--input /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/input_data.json \
--output /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/output_nv_embed7b-qwen3-30b.json \
--embed_model_path /home/jovyan/nfs_share/models/huggingface/hub/NV-Embed \
--nli_model_id /home/jovyan/nfs_share/models/huggingface/hub/MoritzLaurer/deberta-v3-large-zeroshot-v2.0 \
--llm_model_path /home/jovyan/nfs_share/models/huggingface/hub/qwen/Qwen3-30B-A3B-Instruct-2507

python -m src.rel_classify \
--input /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/input_data.json \
--output /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/output_mpnet-v2-qwen3-30b.json \
--embed_model_path /home/jovyan/nfs_share/models/huggingface/hub/all-mpnet-base-v2 \
--nli_model_id /home/jovyan/nfs_share/models/huggingface/hub/MoritzLaurer/deberta-v3-large-zeroshot-v2.0 \
--llm_model_path /home/jovyan/nfs_share/models/huggingface/hub/qwen/Qwen3-30B-A3B-Instruct-2507


python -m src.rel_classify \
--input /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/input_data.json \
--output /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/output_gemma300m-Qwen3-30b.json \
--embed_model_path /home/jovyan/nfs_share/models/huggingface/hub/google/embeddinggemma-300m \
--nli_model_id /home/jovyan/nfs_share/models/huggingface/hub/microsoft/deberta-v3-large \
--llm_model_path /home/jovyan/nfs_share/models/huggingface/hub/qwen/Qwen3-30B-A3B-Instruct-2507


python -m src.rel_classify \
--input /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/input_data.json \
--output /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/output_nemotron_8b-Qwen3-30b.json \
--embed_model_path /home/jovyan/nfs_share/models/huggingface/hub/nvidia/llama-embed-nemotron-8b \
--nli_model_id /home/jovyan/nfs_share/models/huggingface/hub/microsoft/deberta-v3-large \
--llm_model_path /home/jovyan/nfs_share/models/huggingface/hub/qwen/Qwen3-30B-A3B-Instruct-2507




python -m src.rel_classify \
--input /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/input_data.json \
--output /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/output_qwen8b-llama31-8b.json \
--embed_model_path /home/jovyan/nfs_share/models/huggingface/hub/qwen/qwen_embed_8b \
--nli_model_id /home/jovyan/nfs_share/models/huggingface/hub/microsoft/deberta-v3-large \
--llm_model_path /home/jovyan/nfs_share/models/huggingface/hub/meta-llama/Meta-Llama-3.1-8B-Instruct

python -m src.rel_classify \
--input /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/input_data.json \
--output /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/output_qwen4b-llama31-8b.json \
--embed_model_path /home/jovyan/nfs_share/models/huggingface/hub/qwen/qwen_embed_4b \
--nli_model_id /home/jovyan/nfs_share/models/huggingface/hub/microsoft/deberta-v3-large \
--llm_model_path /home/jovyan/nfs_share/models/huggingface/hub/meta-llama/Meta-Llama-3.1-8B-Instruct


python -m src.rel_classify \
--input /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/input_data.json \
--output /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/output_qwen06b-llama31-8b.json \
--embed_model_path /home/jovyan/nfs_share/models/huggingface/hub/qwen/qwen_embed_06b \
--nli_model_id /home/jovyan/nfs_share/models/huggingface/hub/microsoft/deberta-v3-large \
--llm_model_path /home/jovyan/nfs_share/models/huggingface/hub/meta-llama/Meta-Llama-3.1-8B-Instruct

python -m src.rel_classify \
--input /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/input_data.json \
--output /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/output_nv_embed7b-llama31-8b.json \
--embed_model_path /home/jovyan/nfs_share/models/huggingface/hub/NV-Embed \
--nli_model_id /home/jovyan/nfs_share/models/huggingface/hub/MoritzLaurer/deberta-v3-large-zeroshot-v2.0 \
--llm_model_path /home/jovyan/nfs_share/models/huggingface/hub/meta-llama/Meta-Llama-3.1-8B-Instruct

python -m src.rel_classify \
--input /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/input_data.json \
--output /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/output_mpnet-v2-llama31-8b.json \
--embed_model_path /home/jovyan/nfs_share/models/huggingface/hub/all-mpnet-base-v2 \
--nli_model_id /home/jovyan/nfs_share/models/huggingface/hub/MoritzLaurer/deberta-v3-large-zeroshot-v2.0 \
--llm_model_path /home/jovyan/nfs_share/models/huggingface/hub/meta-llama/Meta-Llama-3.1-8B-Instruct


python -m src.rel_classify \
--input /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/input_data.json \
--output /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/output_gemma300m-llama31-8b.json \
--embed_model_path /home/jovyan/nfs_share/models/huggingface/hub/google/embeddinggemma-300m \
--nli_model_id /home/jovyan/nfs_share/models/huggingface/hub/microsoft/deberta-v3-large \
--llm_model_path /home/jovyan/nfs_share/models/huggingface/hub/meta-llama/Meta-Llama-3.1-8B-Instruct

python -m src.rel_classify \
--input /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/input_data.json \
--output /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/output_nemotron_8b-llama31-8b.json \
--embed_model_path /home/jovyan/nfs_share/models/huggingface/hub/nvidia/llama-embed-nemotron-8b \
--nli_model_id /home/jovyan/nfs_share/models/huggingface/hub/microsoft/deberta-v3-large \
--llm_model_path /home/jovyan/nfs_share/models/huggingface/hub/meta-llama/Meta-Llama-3.1-8B-Instruct



python -m src.rel_classify \
--input /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/input_data.json \
--output /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/output_qwen8b-llama31-70b.json \
--embed_model_path /home/jovyan/nfs_share/models/huggingface/hub/qwen/qwen_embed_8b \
--nli_model_id /home/jovyan/nfs_share/models/huggingface/hub/microsoft/deberta-v3-large \
--llm_model_path /home/jovyan/nfs_share/models/huggingface/hub/meta-llama/Meta-Llama-3.1-70B-Instruct

python -m src.rel_classify \
--input /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/input_data.json \
--output /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/output_qwen4b-llama31-70b.json \
--embed_model_path /home/jovyan/nfs_share/models/huggingface/hub/qwen/qwen_embed_4b \
--nli_model_id /home/jovyan/nfs_share/models/huggingface/hub/microsoft/deberta-v3-large \
--llm_model_path /home/jovyan/nfs_share/models/huggingface/hub/meta-llama/Meta-Llama-3.1-70B-Instruct

python -m src.rel_classify \
--input /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/input_data.json \
--output /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/output_qwen06b-llama31-70b.json \
--embed_model_path /home/jovyan/nfs_share/models/huggingface/hub/qwen/qwen_embed_06b \
--nli_model_id /home/jovyan/nfs_share/models/huggingface/hub/microsoft/deberta-v3-large \
--llm_model_path /home/jovyan/nfs_share/models/huggingface/hub/meta-llama/Meta-Llama-3.1-70B-Instruct


python -m src.rel_classify \
--input /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/input_data.json \
--output /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/output_nv_embed7b-llama31-70b.json \
--embed_model_path /home/jovyan/nfs_share/models/huggingface/hub/NV-Embed \
--nli_model_id /home/jovyan/nfs_share/models/huggingface/hub/MoritzLaurer/deberta-v3-large-zeroshot-v2.0 \
--llm_model_path /home/jovyan/nfs_share/models/huggingface/hub/meta-llama/Meta-Llama-3.1-70B-Instruct

python -m src.rel_classify \
--input /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/input_data.json \
--output /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/output_mpnet-v2-llama31-70b.json \
--embed_model_path /home/jovyan/nfs_share/models/huggingface/hub/all-mpnet-base-v2 \
--nli_model_id /home/jovyan/nfs_share/models/huggingface/hub/MoritzLaurer/deberta-v3-large-zeroshot-v2.0 \
--llm_model_path /home/jovyan/nfs_share/models/huggingface/hub/qwen/meta-llama/Meta-Llama-3.1-70B-Instruct


python -m src.rel_classify \
--input /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/input_data.json \
--output /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/output_gemma300m-llama31-70b.json \
--embed_model_path /home/jovyan/nfs_share/models/huggingface/hub/google/embeddinggemma-300m \
--nli_model_id /home/jovyan/nfs_share/models/huggingface/hub/microsoft/deberta-v3-large \
--llm_model_path /home/jovyan/nfs_share/models/huggingface/hub/meta-llama/Meta-Llama-3.1-70B-Instruct

python -m src.rel_classify \
--input /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/input_data.json \
--output /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/output_nemotron_8b-llama31-70b.json \
--embed_model_path /home/jovyan/nfs_share/models/huggingface/hub/nvidia/llama-embed-nemotron-8b \
--nli_model_id /home/jovyan/nfs_share/models/huggingface/hub/microsoft/deberta-v3-large \
--llm_model_path /home/jovyan/nfs_share/models/huggingface/hub/meta-llama/Meta-Llama-3.1-70B-Instruct


python -m src.rel_classify \
--input /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/input_data.json \
--output /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/output_qwen8b-mistral_large123b.json \
--embed_model_path /home/jovyan/nfs_share/models/huggingface/hub/qwen/qwen_embed_8b \
--nli_model_id /home/jovyan/nfs_share/models/huggingface/hub/microsoft/deberta-v3-large \
--llm_model_path /home/jovyan/nfs_share/models/huggingface/hub/mistral_models/mistral_large2411

python -m src.rel_classify \
--input /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/input_data.json \
--output /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/output_qwen4b-mistral_large123b.json \
--embed_model_path /home/jovyan/nfs_share/models/huggingface/hub/qwen/qwen_embed_4b \
--nli_model_id /home/jovyan/nfs_share/models/huggingface/hub/microsoft/deberta-v3-large \
--llm_model_path /home/jovyan/nfs_share/models/huggingface/hub/mistral_models/mistral_large2411

python -m src.rel_classify \
--input /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/input_data.json \
--output /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/output_qwen06b-mistral_large123b.json \
--embed_model_path /home/jovyan/nfs_share/models/huggingface/hub/qwen/qwen_embed_06b \
--nli_model_id /home/jovyan/nfs_share/models/huggingface/hub/microsoft/deberta-v3-large \
--llm_model_path /home/jovyan/nfs_share/models/huggingface/hub/mistral_models/mistral_large2411


python -m src.rel_classify \
--input /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/input_data.json \
--output /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/output_nv_embed7b-mistral_large123b.json \
--embed_model_path /home/jovyan/nfs_share/models/huggingface/hub/NV-Embed \
--nli_model_id /home/jovyan/nfs_share/models/huggingface/hub/MoritzLaurer/deberta-v3-large-zeroshot-v2.0 \
--llm_model_path /home/jovyan/nfs_share/models/huggingface/hub/mistral_models/mistral_large2411

python -m src.rel_classify \
--input /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/input_data.json \
--output /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/output_mpnet-v2-mistral_large123b.json \
--embed_model_path /home/jovyan/nfs_share/models/huggingface/hub/all-mpnet-base-v2 \
--nli_model_id /home/jovyan/nfs_share/models/huggingface/hub/MoritzLaurer/deberta-v3-large-zeroshot-v2.0 \
--llm_model_path /home/jovyan/nfs_share/models/huggingface/hub/mistral_models/mistral_large2411

python -m src.rel_classify \
--input /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/input_data.json \
--output /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/output_gemma300m-mistral_large123b.json \
--embed_model_path /home/jovyan/nfs_share/models/huggingface/hub/google/embeddinggemma-300m \
--nli_model_id /home/jovyan/nfs_share/models/huggingface/hub/microsoft/deberta-v3-large \
--llm_model_path /home/jovyan/nfs_share/models/huggingface/hub/mistral_models/mistral_large2411

python -m src.rel_classify \
--input /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/input_data.json \
--output /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/output_nemotron_8b-mistral_large123b.json \
--embed_model_path /home/jovyan/nfs_share/models/huggingface/hub/nvidia/llama-embed-nemotron-8b \
--nli_model_id /home/jovyan/nfs_share/models/huggingface/hub/microsoft/deberta-v3-large \
--llm_model_path /home/jovyan/nfs_share/models/huggingface/hub/mistral_models/mistral_large2411



python -m src.rel_classify \
--input /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/input_data.json \
--output /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/output_qwen8b-mistral_small24b.json \
--embed_model_path /home/jovyan/nfs_share/models/huggingface/hub/qwen/qwen_embed_8b \
--nli_model_id /home/jovyan/nfs_share/models/huggingface/hub/microsoft/deberta-v3-large \
--llm_model_path /home/jovyan/nfs_share/models/huggingface/hub/mistral_models/mistral_small2501

python -m src.rel_classify \
--input /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/input_data.json \
--output /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/output_qwen4b-mistral_small24b.json \
--embed_model_path /home/jovyan/nfs_share/models/huggingface/hub/qwen/qwen_embed_4b \
--nli_model_id /home/jovyan/nfs_share/models/huggingface/hub/microsoft/deberta-v3-large \
--llm_model_path /home/jovyan/nfs_share/models/huggingface/hub/mistral_models/mistral_small2501


python -m src.rel_classify \
--input /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/input_data.json \
--output /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/output_qwen06b-mistral_small24b.json \
--embed_model_path /home/jovyan/nfs_share/models/huggingface/hub/qwen/qwen_embed_06b \
--nli_model_id /home/jovyan/nfs_share/models/huggingface/hub/microsoft/deberta-v3-large \
--llm_model_path /home/jovyan/nfs_share/models/huggingface/hub/mistral_models/mistral_small2501

python -m src.rel_classify \
--input /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/input_data.json \
--output /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/output_nv_embed7b-mistral_small24b.json \
--embed_model_path /home/jovyan/nfs_share/models/huggingface/hub/NV-Embed \
--nli_model_id /home/jovyan/nfs_share/models/huggingface/hub/MoritzLaurer/deberta-v3-large-zeroshot-v2.0 \
--llm_model_path /home/jovyan/nfs_share/models/huggingface/hub/mistral_models/mistral_small2501

python -m src.rel_classify \
--input /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/input_data.json \
--output /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/output_mpnet-v2-mistral_small24b.json \
--embed_model_path /home/jovyan/nfs_share/models/huggingface/hub/all-mpnet-base-v2 \
--nli_model_id /home/jovyan/nfs_share/models/huggingface/hub/MoritzLaurer/deberta-v3-large-zeroshot-v2.0 \
--llm_model_path /home/jovyan/nfs_share/models/huggingface/hub/mistral_models/mistral_small2501


python -m src.rel_classify \
--input /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/input_data.json \
--output /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/output_gemma300m-mistral_small24b.json \
--embed_model_path /home/jovyan/nfs_share/models/huggingface/hub/google/embeddinggemma-300m \
--nli_model_id /home/jovyan/nfs_share/models/huggingface/hub/microsoft/deberta-v3-large \
--llm_model_path /home/jovyan/nfs_share/models/huggingface/hub/mistral_models/mistral_small2501

python -m src.rel_classify \
--input /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/input_data.json \
--output /home/jovyan/nfs_share/Simon/PROJECTS/relevance_judge/data/output_nemotron_8b-mistral_small24b.json \
--embed_model_path /home/jovyan/nfs_share/models/huggingface/hub/nvidia/llama-embed-nemotron-8b \
--nli_model_id /home/jovyan/nfs_share/models/huggingface/hub/microsoft/deberta-v3-large \
--llm_model_path /home/jovyan/nfs_share/models/huggingface/hub/mistral_models/mistral_small2501

