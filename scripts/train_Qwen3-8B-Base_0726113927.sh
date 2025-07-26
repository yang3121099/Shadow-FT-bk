# ##### Auto-generated 2025-07-26 11:39:27 #####
# # Model     : Qwen3-8B-Base
# # LoRA mode : true
# # Template  : qwen3

# ##### Environment #####
# export VLLM_WORKER_MULTIPROC_METHOD=spawn
# export HF_HUB_OFFLINE=0
# export HF_DATASETS_OFFLINE=0
# export HF_DATASETS_TRUST_REMOTE_CODE=1
# export TRUST_REMOTE_CODE=True
# export HF_ALLOW_CODE_EVAL=1

# ##### Training #####
# ###### B  max=2000  lr=0.000005 ######
# mkdir -p "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/B-2k-lora-rank128-lr0.000005-Shadow_2k"
# cd "/home/ubuntu/Shadow"
# llamafactory-cli train \
#   --model_name_or_path "/home/ubuntu/models//Qwen3-8B-Base" \
#   --stage sft \
#   --do_train true \
#   --finetuning_type lora --lora_rank 128 \
#   --dataset "Shadow_2k" \
#   --template "qwen3" \
#   --cutoff_len 4096 \
#   --max_samples 2000 \
#   --output_dir "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/B-2k-lora-rank128-lr0.000005-Shadow_2k" \
#   --per_device_train_batch_size 2 \
#   --gradient_accumulation_steps 16 \
#   --learning_rate 0.000005 \
#   --num_train_epochs 1 \
#   --logging_steps 1 \
#   --save_steps 1000 \
#   --plot_loss true \
#   --lr_scheduler_type cosine \
#   --warmup_ratio 0.1 \
#   --bf16 true \
#   --val_size 0.01 \
#   --per_device_eval_batch_size 1 \
#   --eval_strategy steps \
#   --eval_steps 10000 \
#   --trust_remote_code True \
#   --flash_attn fa2 \
#   --overwrite_cache false

# ###### I  max=2000  lr=0.000005 ######
# mkdir -p "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/I-2k-lora-rank128-lr0.000005-Shadow_2k"
# cd "/home/ubuntu/Shadow"
# llamafactory-cli train \
#   --model_name_or_path "/home/ubuntu/models//Qwen3-8B" \
#   --stage sft \
#   --do_train true \
#   --finetuning_type lora --lora_rank 128 \
#   --dataset "Shadow_2k" \
#   --template "qwen3" \
#   --cutoff_len 4096 \
#   --max_samples 2000 \
#   --output_dir "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/I-2k-lora-rank128-lr0.000005-Shadow_2k" \
#   --per_device_train_batch_size 2 \
#   --gradient_accumulation_steps 16 \
#   --learning_rate 0.000005 \
#   --num_train_epochs 1 \
#   --logging_steps 1 \
#   --save_steps 1000 \
#   --plot_loss true \
#   --lr_scheduler_type cosine \
#   --warmup_ratio 0.1 \
#   --bf16 true \
#   --val_size 0.01 \
#   --per_device_eval_batch_size 1 \
#   --eval_strategy steps \
#   --eval_steps 10000 \
#   --trust_remote_code True \
#   --flash_attn fa2 \
#   --overwrite_cache false

# ###### B  max=2000  lr=0.00001 ######
# mkdir -p "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/B-2k-lora-rank128-lr0.00001-Shadow_2k"
# cd "/home/ubuntu/Shadow"
# llamafactory-cli train \
#   --model_name_or_path "/home/ubuntu/models//Qwen3-8B-Base" \
#   --stage sft \
#   --do_train true \
#   --finetuning_type lora --lora_rank 128 \
#   --dataset "Shadow_2k" \
#   --template "qwen3" \
#   --cutoff_len 4096 \
#   --max_samples 2000 \
#   --output_dir "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/B-2k-lora-rank128-lr0.00001-Shadow_2k" \
#   --per_device_train_batch_size 2 \
#   --gradient_accumulation_steps 16 \
#   --learning_rate 0.00001 \
#   --num_train_epochs 1 \
#   --logging_steps 1 \
#   --save_steps 1000 \
#   --plot_loss true \
#   --lr_scheduler_type cosine \
#   --warmup_ratio 0.1 \
#   --bf16 true \
#   --val_size 0.01 \
#   --per_device_eval_batch_size 1 \
#   --eval_strategy steps \
#   --eval_steps 10000 \
#   --trust_remote_code True \
#   --flash_attn fa2 \
#   --overwrite_cache false

# ###### I  max=2000  lr=0.00001 ######
# mkdir -p "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/I-2k-lora-rank128-lr0.00001-Shadow_2k"
# cd "/home/ubuntu/Shadow"
# llamafactory-cli train \
#   --model_name_or_path "/home/ubuntu/models//Qwen3-8B" \
#   --stage sft \
#   --do_train true \
#   --finetuning_type lora --lora_rank 128 \
#   --dataset "Shadow_2k" \
#   --template "qwen3" \
#   --cutoff_len 4096 \
#   --max_samples 2000 \
#   --output_dir "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/I-2k-lora-rank128-lr0.00001-Shadow_2k" \
#   --per_device_train_batch_size 2 \
#   --gradient_accumulation_steps 16 \
#   --learning_rate 0.00001 \
#   --num_train_epochs 1 \
#   --logging_steps 1 \
#   --save_steps 1000 \
#   --plot_loss true \
#   --lr_scheduler_type cosine \
#   --warmup_ratio 0.1 \
#   --bf16 true \
#   --val_size 0.01 \
#   --per_device_eval_batch_size 1 \
#   --eval_strategy steps \
#   --eval_steps 10000 \
#   --trust_remote_code True \
#   --flash_attn fa2 \
#   --overwrite_cache false

# ###### B  max=2000  lr=0.00002 ######
# mkdir -p "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/B-2k-lora-rank128-lr0.00002-Shadow_2k"
# cd "/home/ubuntu/Shadow"
# llamafactory-cli train \
#   --model_name_or_path "/home/ubuntu/models//Qwen3-8B-Base" \
#   --stage sft \
#   --do_train true \
#   --finetuning_type lora --lora_rank 128 \
#   --dataset "Shadow_2k" \
#   --template "qwen3" \
#   --cutoff_len 4096 \
#   --max_samples 2000 \
#   --output_dir "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/B-2k-lora-rank128-lr0.00002-Shadow_2k" \
#   --per_device_train_batch_size 2 \
#   --gradient_accumulation_steps 16 \
#   --learning_rate 0.00002 \
#   --num_train_epochs 1 \
#   --logging_steps 1 \
#   --save_steps 1000 \
#   --plot_loss true \
#   --lr_scheduler_type cosine \
#   --warmup_ratio 0.1 \
#   --bf16 true \
#   --val_size 0.01 \
#   --per_device_eval_batch_size 1 \
#   --eval_strategy steps \
#   --eval_steps 10000 \
#   --trust_remote_code True \
#   --flash_attn fa2 \
#   --overwrite_cache false

# ###### I  max=2000  lr=0.00002 ######
# mkdir -p "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/I-2k-lora-rank128-lr0.00002-Shadow_2k"
# cd "/home/ubuntu/Shadow"
# llamafactory-cli train \
#   --model_name_or_path "/home/ubuntu/models//Qwen3-8B" \
#   --stage sft \
#   --do_train true \
#   --finetuning_type lora --lora_rank 128 \
#   --dataset "Shadow_2k" \
#   --template "qwen3" \
#   --cutoff_len 4096 \
#   --max_samples 2000 \
#   --output_dir "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/I-2k-lora-rank128-lr0.00002-Shadow_2k" \
#   --per_device_train_batch_size 2 \
#   --gradient_accumulation_steps 16 \
#   --learning_rate 0.00002 \
#   --num_train_epochs 1 \
#   --logging_steps 1 \
#   --save_steps 1000 \
#   --plot_loss true \
#   --lr_scheduler_type cosine \
#   --warmup_ratio 0.1 \
#   --bf16 true \
#   --val_size 0.01 \
#   --per_device_eval_batch_size 1 \
#   --eval_strategy steps \
#   --eval_steps 10000 \
#   --trust_remote_code True \
#   --flash_attn fa2 \
#   --overwrite_cache false

# ###### B  max=2000  lr=0.00005 ######
# mkdir -p "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/B-2k-lora-rank128-lr0.00005-Shadow_2k"
# cd "/home/ubuntu/Shadow"
# llamafactory-cli train \
#   --model_name_or_path "/home/ubuntu/models//Qwen3-8B-Base" \
#   --stage sft \
#   --do_train true \
#   --finetuning_type lora --lora_rank 128 \
#   --dataset "Shadow_2k" \
#   --template "qwen3" \
#   --cutoff_len 4096 \
#   --max_samples 2000 \
#   --output_dir "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/B-2k-lora-rank128-lr0.00005-Shadow_2k" \
#   --per_device_train_batch_size 2 \
#   --gradient_accumulation_steps 16 \
#   --learning_rate 0.00005 \
#   --num_train_epochs 1 \
#   --logging_steps 1 \
#   --save_steps 1000 \
#   --plot_loss true \
#   --lr_scheduler_type cosine \
#   --warmup_ratio 0.1 \
#   --bf16 true \
#   --val_size 0.01 \
#   --per_device_eval_batch_size 1 \
#   --eval_strategy steps \
#   --eval_steps 10000 \
#   --trust_remote_code True \
#   --flash_attn fa2 \
#   --overwrite_cache false

# ###### I  max=2000  lr=0.00005 ######
# mkdir -p "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/I-2k-lora-rank128-lr0.00005-Shadow_2k"
# cd "/home/ubuntu/Shadow"
# llamafactory-cli train \
#   --model_name_or_path "/home/ubuntu/models//Qwen3-8B" \
#   --stage sft \
#   --do_train true \
#   --finetuning_type lora --lora_rank 128 \
#   --dataset "Shadow_2k" \
#   --template "qwen3" \
#   --cutoff_len 4096 \
#   --max_samples 2000 \
#   --output_dir "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/I-2k-lora-rank128-lr0.00005-Shadow_2k" \
#   --per_device_train_batch_size 2 \
#   --gradient_accumulation_steps 16 \
#   --learning_rate 0.00005 \
#   --num_train_epochs 1 \
#   --logging_steps 1 \
#   --save_steps 1000 \
#   --plot_loss true \
#   --lr_scheduler_type cosine \
#   --warmup_ratio 0.1 \
#   --bf16 true \
#   --val_size 0.01 \
#   --per_device_eval_batch_size 1 \
#   --eval_strategy steps \
#   --eval_steps 10000 \
#   --trust_remote_code True \
#   --flash_attn fa2 \
#   --overwrite_cache false

# ###### B  max=2000  lr=0.0001 ######
# mkdir -p "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/B-2k-lora-rank128-lr0.0001-Shadow_2k"
# cd "/home/ubuntu/Shadow"
# llamafactory-cli train \
#   --model_name_or_path "/home/ubuntu/models//Qwen3-8B-Base" \
#   --stage sft \
#   --do_train true \
#   --finetuning_type lora --lora_rank 128 \
#   --dataset "Shadow_2k" \
#   --template "qwen3" \
#   --cutoff_len 4096 \
#   --max_samples 2000 \
#   --output_dir "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/B-2k-lora-rank128-lr0.0001-Shadow_2k" \
#   --per_device_train_batch_size 2 \
#   --gradient_accumulation_steps 16 \
#   --learning_rate 0.0001 \
#   --num_train_epochs 1 \
#   --logging_steps 1 \
#   --save_steps 1000 \
#   --plot_loss true \
#   --lr_scheduler_type cosine \
#   --warmup_ratio 0.1 \
#   --bf16 true \
#   --val_size 0.01 \
#   --per_device_eval_batch_size 1 \
#   --eval_strategy steps \
#   --eval_steps 10000 \
#   --trust_remote_code True \
#   --flash_attn fa2 \
#   --overwrite_cache false

# ###### I  max=2000  lr=0.0001 ######
# mkdir -p "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/I-2k-lora-rank128-lr0.0001-Shadow_2k"
# cd "/home/ubuntu/Shadow"
# llamafactory-cli train \
#   --model_name_or_path "/home/ubuntu/models//Qwen3-8B" \
#   --stage sft \
#   --do_train true \
#   --finetuning_type lora --lora_rank 128 \
#   --dataset "Shadow_2k" \
#   --template "qwen3" \
#   --cutoff_len 4096 \
#   --max_samples 2000 \
#   --output_dir "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/I-2k-lora-rank128-lr0.0001-Shadow_2k" \
#   --per_device_train_batch_size 2 \
#   --gradient_accumulation_steps 16 \
#   --learning_rate 0.0001 \
#   --num_train_epochs 1 \
#   --logging_steps 1 \
#   --save_steps 1000 \
#   --plot_loss true \
#   --lr_scheduler_type cosine \
#   --warmup_ratio 0.1 \
#   --bf16 true \
#   --val_size 0.01 \
#   --per_device_eval_batch_size 1 \
#   --eval_strategy steps \
#   --eval_steps 10000 \
#   --trust_remote_code True \
#   --flash_attn fa2 \
#   --overwrite_cache false

# ###### B  max=2000  lr=0.0002 ######
# mkdir -p "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/B-2k-lora-rank128-lr0.0002-Shadow_2k"
# cd "/home/ubuntu/Shadow"
# llamafactory-cli train \
#   --model_name_or_path "/home/ubuntu/models//Qwen3-8B-Base" \
#   --stage sft \
#   --do_train true \
#   --finetuning_type lora --lora_rank 128 \
#   --dataset "Shadow_2k" \
#   --template "qwen3" \
#   --cutoff_len 4096 \
#   --max_samples 2000 \
#   --output_dir "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/B-2k-lora-rank128-lr0.0002-Shadow_2k" \
#   --per_device_train_batch_size 2 \
#   --gradient_accumulation_steps 16 \
#   --learning_rate 0.0002 \
#   --num_train_epochs 1 \
#   --logging_steps 1 \
#   --save_steps 1000 \
#   --plot_loss true \
#   --lr_scheduler_type cosine \
#   --warmup_ratio 0.1 \
#   --bf16 true \
#   --val_size 0.01 \
#   --per_device_eval_batch_size 1 \
#   --eval_strategy steps \
#   --eval_steps 10000 \
#   --trust_remote_code True \
#   --flash_attn fa2 \
#   --overwrite_cache false

# ###### I  max=2000  lr=0.0002 ######
# mkdir -p "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/I-2k-lora-rank128-lr0.0002-Shadow_2k"
# cd "/home/ubuntu/Shadow"
# llamafactory-cli train \
#   --model_name_or_path "/home/ubuntu/models//Qwen3-8B" \
#   --stage sft \
#   --do_train true \
#   --finetuning_type lora --lora_rank 128 \
#   --dataset "Shadow_2k" \
#   --template "qwen3" \
#   --cutoff_len 4096 \
#   --max_samples 2000 \
#   --output_dir "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/I-2k-lora-rank128-lr0.0002-Shadow_2k" \
#   --per_device_train_batch_size 2 \
#   --gradient_accumulation_steps 16 \
#   --learning_rate 0.0002 \
#   --num_train_epochs 1 \
#   --logging_steps 1 \
#   --save_steps 1000 \
#   --plot_loss true \
#   --lr_scheduler_type cosine \
#   --warmup_ratio 0.1 \
#   --bf16 true \
#   --val_size 0.01 \
#   --per_device_eval_batch_size 1 \
#   --eval_strategy steps \
#   --eval_steps 10000 \
#   --trust_remote_code True \
#   --flash_attn fa2 \
#   --overwrite_cache false

# ###### B  max=2000  lr=0.0005 ######
# mkdir -p "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/B-2k-lora-rank128-lr0.0005-Shadow_2k"
# cd "/home/ubuntu/Shadow"
# llamafactory-cli train \
#   --model_name_or_path "/home/ubuntu/models//Qwen3-8B-Base" \
#   --stage sft \
#   --do_train true \
#   --finetuning_type lora --lora_rank 128 \
#   --dataset "Shadow_2k" \
#   --template "qwen3" \
#   --cutoff_len 4096 \
#   --max_samples 2000 \
#   --output_dir "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/B-2k-lora-rank128-lr0.0005-Shadow_2k" \
#   --per_device_train_batch_size 2 \
#   --gradient_accumulation_steps 16 \
#   --learning_rate 0.0005 \
#   --num_train_epochs 1 \
#   --logging_steps 1 \
#   --save_steps 1000 \
#   --plot_loss true \
#   --lr_scheduler_type cosine \
#   --warmup_ratio 0.1 \
#   --bf16 true \
#   --val_size 0.01 \
#   --per_device_eval_batch_size 1 \
#   --eval_strategy steps \
#   --eval_steps 10000 \
#   --trust_remote_code True \
#   --flash_attn fa2 \
#   --overwrite_cache false

# ###### I  max=2000  lr=0.0005 ######
# mkdir -p "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/I-2k-lora-rank128-lr0.0005-Shadow_2k"
# cd "/home/ubuntu/Shadow"
# llamafactory-cli train \
#   --model_name_or_path "/home/ubuntu/models//Qwen3-8B" \
#   --stage sft \
#   --do_train true \
#   --finetuning_type lora --lora_rank 128 \
#   --dataset "Shadow_2k" \
#   --template "qwen3" \
#   --cutoff_len 4096 \
#   --max_samples 2000 \
#   --output_dir "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/I-2k-lora-rank128-lr0.0005-Shadow_2k" \
#   --per_device_train_batch_size 2 \
#   --gradient_accumulation_steps 16 \
#   --learning_rate 0.0005 \
#   --num_train_epochs 1 \
#   --logging_steps 1 \
#   --save_steps 1000 \
#   --plot_loss true \
#   --lr_scheduler_type cosine \
#   --warmup_ratio 0.1 \
#   --bf16 true \
#   --val_size 0.01 \
#   --per_device_eval_batch_size 1 \
#   --eval_strategy steps \
#   --eval_steps 10000 \
#   --trust_remote_code True \
#   --flash_attn fa2 \
#   --overwrite_cache false

# ###### B  max=2000  lr=0.001 ######
# mkdir -p "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/B-2k-lora-rank128-lr0.001-Shadow_2k"
# cd "/home/ubuntu/Shadow"
# llamafactory-cli train \
#   --model_name_or_path "/home/ubuntu/models//Qwen3-8B-Base" \
#   --stage sft \
#   --do_train true \
#   --finetuning_type lora --lora_rank 128 \
#   --dataset "Shadow_2k" \
#   --template "qwen3" \
#   --cutoff_len 4096 \
#   --max_samples 2000 \
#   --output_dir "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/B-2k-lora-rank128-lr0.001-Shadow_2k" \
#   --per_device_train_batch_size 2 \
#   --gradient_accumulation_steps 16 \
#   --learning_rate 0.001 \
#   --num_train_epochs 1 \
#   --logging_steps 1 \
#   --save_steps 1000 \
#   --plot_loss true \
#   --lr_scheduler_type cosine \
#   --warmup_ratio 0.1 \
#   --bf16 true \
#   --val_size 0.01 \
#   --per_device_eval_batch_size 1 \
#   --eval_strategy steps \
#   --eval_steps 10000 \
#   --trust_remote_code True \
#   --flash_attn fa2 \
#   --overwrite_cache false

# ###### I  max=2000  lr=0.001 ######
# mkdir -p "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/I-2k-lora-rank128-lr0.001-Shadow_2k"
# cd "/home/ubuntu/Shadow"
# llamafactory-cli train \
#   --model_name_or_path "/home/ubuntu/models//Qwen3-8B" \
#   --stage sft \
#   --do_train true \
#   --finetuning_type lora --lora_rank 128 \
#   --dataset "Shadow_2k" \
#   --template "qwen3" \
#   --cutoff_len 4096 \
#   --max_samples 2000 \
#   --output_dir "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/I-2k-lora-rank128-lr0.001-Shadow_2k" \
#   --per_device_train_batch_size 2 \
#   --gradient_accumulation_steps 16 \
#   --learning_rate 0.001 \
#   --num_train_epochs 1 \
#   --logging_steps 1 \
#   --save_steps 1000 \
#   --plot_loss true \
#   --lr_scheduler_type cosine \
#   --warmup_ratio 0.1 \
#   --bf16 true \
#   --val_size 0.01 \
#   --per_device_eval_batch_size 1 \
#   --eval_strategy steps \
#   --eval_steps 10000 \
#   --trust_remote_code True \
#   --flash_attn fa2 \
#   --overwrite_cache false

# ###### B  max=2000  lr=0.002 ######
# mkdir -p "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/B-2k-lora-rank128-lr0.002-Shadow_2k"
# cd "/home/ubuntu/Shadow"
# llamafactory-cli train \
#   --model_name_or_path "/home/ubuntu/models//Qwen3-8B-Base" \
#   --stage sft \
#   --do_train true \
#   --finetuning_type lora --lora_rank 128 \
#   --dataset "Shadow_2k" \
#   --template "qwen3" \
#   --cutoff_len 4096 \
#   --max_samples 2000 \
#   --output_dir "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/B-2k-lora-rank128-lr0.002-Shadow_2k" \
#   --per_device_train_batch_size 2 \
#   --gradient_accumulation_steps 16 \
#   --learning_rate 0.002 \
#   --num_train_epochs 1 \
#   --logging_steps 1 \
#   --save_steps 1000 \
#   --plot_loss true \
#   --lr_scheduler_type cosine \
#   --warmup_ratio 0.1 \
#   --bf16 true \
#   --val_size 0.01 \
#   --per_device_eval_batch_size 1 \
#   --eval_strategy steps \
#   --eval_steps 10000 \
#   --trust_remote_code True \
#   --flash_attn fa2 \
#   --overwrite_cache false

# ###### I  max=2000  lr=0.002 ######
# mkdir -p "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/I-2k-lora-rank128-lr0.002-Shadow_2k"
# cd "/home/ubuntu/Shadow"
# llamafactory-cli train \
#   --model_name_or_path "/home/ubuntu/models//Qwen3-8B" \
#   --stage sft \
#   --do_train true \
#   --finetuning_type lora --lora_rank 128 \
#   --dataset "Shadow_2k" \
#   --template "qwen3" \
#   --cutoff_len 4096 \
#   --max_samples 2000 \
#   --output_dir "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/I-2k-lora-rank128-lr0.002-Shadow_2k" \
#   --per_device_train_batch_size 2 \
#   --gradient_accumulation_steps 16 \
#   --learning_rate 0.002 \
#   --num_train_epochs 1 \
#   --logging_steps 1 \
#   --save_steps 1000 \
#   --plot_loss true \
#   --lr_scheduler_type cosine \
#   --warmup_ratio 0.1 \
#   --bf16 true \
#   --val_size 0.01 \
#   --per_device_eval_batch_size 1 \
#   --eval_strategy steps \
#   --eval_steps 10000 \
#   --trust_remote_code True \
#   --flash_attn fa2 \
#   --overwrite_cache false

# ##### LoRA delta-merge #####
# python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
#   --adapter_path "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/B-2k-lora-rank128-lr0.000005-Shadow_2k" \
#   --target_base "/home/ubuntu/models//Qwen3-8B" \
#   --merge_tag "B2I" \
#   --template "qwen3"

# python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
#   --adapter_path "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/I-2k-lora-rank128-lr0.000005-Shadow_2k" \
#   --target_base "/home/ubuntu/models//Qwen3-8B" \
#   --merge_tag "I2I" \
#   --template "qwen3"

# #python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
# #  --adapter_path "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/I-2k-lora-rank128-lr0.000005-Shadow_2k" \
# #  --target_base "/home/ubuntu/models//Qwen3-8B-Base" \
# #  --merge_tag "I2B" \
# #  --template "qwen3"

# #python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
# #  --adapter_path "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/B-2k-lora-rank128-lr0.000005-Shadow_2k" \
# #  --target_base "/home/ubuntu/models//Qwen3-8B-Base" \
# #  --merge_tag "B2B" \
# #  --template "qwen3"

# python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
#   --adapter_path "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/B-2k-lora-rank128-lr0.00001-Shadow_2k" \
#   --target_base "/home/ubuntu/models//Qwen3-8B" \
#   --merge_tag "B2I" \
#   --template "qwen3"

# python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
#   --adapter_path "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/I-2k-lora-rank128-lr0.00001-Shadow_2k" \
#   --target_base "/home/ubuntu/models//Qwen3-8B" \
#   --merge_tag "I2I" \
#   --template "qwen3"

# #python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
# #  --adapter_path "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/I-2k-lora-rank128-lr0.00001-Shadow_2k" \
# #  --target_base "/home/ubuntu/models//Qwen3-8B-Base" \
# #  --merge_tag "I2B" \
# #  --template "qwen3"

# #python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
# #  --adapter_path "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/B-2k-lora-rank128-lr0.00001-Shadow_2k" \
# #  --target_base "/home/ubuntu/models//Qwen3-8B-Base" \
# #  --merge_tag "B2B" \
# #  --template "qwen3"

# python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
#   --adapter_path "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/B-2k-lora-rank128-lr0.00002-Shadow_2k" \
#   --target_base "/home/ubuntu/models//Qwen3-8B" \
#   --merge_tag "B2I" \
#   --template "qwen3"

# python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
#   --adapter_path "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/I-2k-lora-rank128-lr0.00002-Shadow_2k" \
#   --target_base "/home/ubuntu/models//Qwen3-8B" \
#   --merge_tag "I2I" \
#   --template "qwen3"

# #python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
# #  --adapter_path "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/I-2k-lora-rank128-lr0.00002-Shadow_2k" \
# #  --target_base "/home/ubuntu/models//Qwen3-8B-Base" \
# #  --merge_tag "I2B" \
# #  --template "qwen3"

# #python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
# #  --adapter_path "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/B-2k-lora-rank128-lr0.00002-Shadow_2k" \
# #  --target_base "/home/ubuntu/models//Qwen3-8B-Base" \
# #  --merge_tag "B2B" \
# #  --template "qwen3"

# python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
#   --adapter_path "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/B-2k-lora-rank128-lr0.00005-Shadow_2k" \
#   --target_base "/home/ubuntu/models//Qwen3-8B" \
#   --merge_tag "B2I" \
#   --template "qwen3"

# python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
#   --adapter_path "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/I-2k-lora-rank128-lr0.00005-Shadow_2k" \
#   --target_base "/home/ubuntu/models//Qwen3-8B" \
#   --merge_tag "I2I" \
#   --template "qwen3"

# #python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
# #  --adapter_path "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/I-2k-lora-rank128-lr0.00005-Shadow_2k" \
# #  --target_base "/home/ubuntu/models//Qwen3-8B-Base" \
# #  --merge_tag "I2B" \
# #  --template "qwen3"

# #python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
# #  --adapter_path "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/B-2k-lora-rank128-lr0.00005-Shadow_2k" \
# #  --target_base "/home/ubuntu/models//Qwen3-8B-Base" \
# #  --merge_tag "B2B" \
# #  --template "qwen3"

# python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
#   --adapter_path "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/B-2k-lora-rank128-lr0.0001-Shadow_2k" \
#   --target_base "/home/ubuntu/models//Qwen3-8B" \
#   --merge_tag "B2I" \
#   --template "qwen3"

# python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
#   --adapter_path "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/I-2k-lora-rank128-lr0.0001-Shadow_2k" \
#   --target_base "/home/ubuntu/models//Qwen3-8B" \
#   --merge_tag "I2I" \
#   --template "qwen3"

# #python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
# #  --adapter_path "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/I-2k-lora-rank128-lr0.0001-Shadow_2k" \
# #  --target_base "/home/ubuntu/models//Qwen3-8B-Base" \
# #  --merge_tag "I2B" \
# #  --template "qwen3"

# #python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
# #  --adapter_path "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/B-2k-lora-rank128-lr0.0001-Shadow_2k" \
# #  --target_base "/home/ubuntu/models//Qwen3-8B-Base" \
# #  --merge_tag "B2B" \
# #  --template "qwen3"

# python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
#   --adapter_path "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/B-2k-lora-rank128-lr0.0002-Shadow_2k" \
#   --target_base "/home/ubuntu/models//Qwen3-8B" \
#   --merge_tag "B2I" \
#   --template "qwen3"

# python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
#   --adapter_path "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/I-2k-lora-rank128-lr0.0002-Shadow_2k" \
#   --target_base "/home/ubuntu/models//Qwen3-8B" \
#   --merge_tag "I2I" \
#   --template "qwen3"

# #python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
# #  --adapter_path "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/I-2k-lora-rank128-lr0.0002-Shadow_2k" \
# #  --target_base "/home/ubuntu/models//Qwen3-8B-Base" \
# #  --merge_tag "I2B" \
# #  --template "qwen3"

# #python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
# #  --adapter_path "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/B-2k-lora-rank128-lr0.0002-Shadow_2k" \
# #  --target_base "/home/ubuntu/models//Qwen3-8B-Base" \
# #  --merge_tag "B2B" \
# #  --template "qwen3"

# python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
#   --adapter_path "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/B-2k-lora-rank128-lr0.0005-Shadow_2k" \
#   --target_base "/home/ubuntu/models//Qwen3-8B" \
#   --merge_tag "B2I" \
#   --template "qwen3"

# python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
#   --adapter_path "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/I-2k-lora-rank128-lr0.0005-Shadow_2k" \
#   --target_base "/home/ubuntu/models//Qwen3-8B" \
#   --merge_tag "I2I" \
#   --template "qwen3"

# #python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
# #  --adapter_path "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/I-2k-lora-rank128-lr0.0005-Shadow_2k" \
# #  --target_base "/home/ubuntu/models//Qwen3-8B-Base" \
# #  --merge_tag "I2B" \
# #  --template "qwen3"

# #python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
# #  --adapter_path "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/B-2k-lora-rank128-lr0.0005-Shadow_2k" \
# #  --target_base "/home/ubuntu/models//Qwen3-8B-Base" \
# #  --merge_tag "B2B" \
# #  --template "qwen3"

# python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
#   --adapter_path "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/B-2k-lora-rank128-lr0.001-Shadow_2k" \
#   --target_base "/home/ubuntu/models//Qwen3-8B" \
#   --merge_tag "B2I" \
#   --template "qwen3"

# python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
#   --adapter_path "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/I-2k-lora-rank128-lr0.001-Shadow_2k" \
#   --target_base "/home/ubuntu/models//Qwen3-8B" \
#   --merge_tag "I2I" \
#   --template "qwen3"

# #python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
# #  --adapter_path "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/I-2k-lora-rank128-lr0.001-Shadow_2k" \
# #  --target_base "/home/ubuntu/models//Qwen3-8B-Base" \
# #  --merge_tag "I2B" \
# #  --template "qwen3"

# #python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
# #  --adapter_path "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/B-2k-lora-rank128-lr0.001-Shadow_2k" \
# #  --target_base "/home/ubuntu/models//Qwen3-8B-Base" \
# #  --merge_tag "B2B" \
# #  --template "qwen3"

# python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
#   --adapter_path "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/B-2k-lora-rank128-lr0.002-Shadow_2k" \
#   --target_base "/home/ubuntu/models//Qwen3-8B" \
#   --merge_tag "B2I" \
#   --template "qwen3"

# python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
#   --adapter_path "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/I-2k-lora-rank128-lr0.002-Shadow_2k" \
#   --target_base "/home/ubuntu/models//Qwen3-8B" \
#   --merge_tag "I2I" \
#   --template "qwen3"

# #python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
# #  --adapter_path "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/I-2k-lora-rank128-lr0.002-Shadow_2k" \
# #  --target_base "/home/ubuntu/models//Qwen3-8B-Base" \
# #  --merge_tag "I2B" \
# #  --template "qwen3"

# #python3 /home/ubuntu/Shadow/src/shadow/merge_lora.py \
# #  --adapter_path "/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/B-2k-lora-rank128-lr0.002-Shadow_2k" \
# #  --target_base "/home/ubuntu/models//Qwen3-8B-Base" \
# #  --merge_tag "B2B" \
# #  --template "qwen3"

# ##### Evaluation list #####
# # ('0726/result-Qwen3-8B-Base-0726/merged-B2I-lr0.000005','/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/merged-B2I-lr0.000005'),
# # ('0726/result-Qwen3-8B-Base-0726/merged-I2I-lr0.000005','/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/merged-I2I-lr0.000005'),
# ## ('0726/result-Qwen3-8B-Base-0726/merged-I2B-lr0.000005','/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/merged-I2B-lr0.000005'),
# ## ('0726/result-Qwen3-8B-Base-0726/merged-B2B-lr0.000005','/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/merged-B2B-lr0.000005'),
# # ('0726/result-Qwen3-8B-Base-0726/merged-B2I-lr0.00001','/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/merged-B2I-lr0.00001'),
# # ('0726/result-Qwen3-8B-Base-0726/merged-I2I-lr0.00001','/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/merged-I2I-lr0.00001'),
# ## ('0726/result-Qwen3-8B-Base-0726/merged-I2B-lr0.00001','/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/merged-I2B-lr0.00001'),
# ## ('0726/result-Qwen3-8B-Base-0726/merged-B2B-lr0.00001','/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/merged-B2B-lr0.00001'),
# # ('0726/result-Qwen3-8B-Base-0726/merged-B2I-lr0.00002','/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/merged-B2I-lr0.00002'),
# # ('0726/result-Qwen3-8B-Base-0726/merged-I2I-lr0.00002','/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/merged-I2I-lr0.00002'),
# ## ('0726/result-Qwen3-8B-Base-0726/merged-I2B-lr0.00002','/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/merged-I2B-lr0.00002'),
# ## ('0726/result-Qwen3-8B-Base-0726/merged-B2B-lr0.00002','/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/merged-B2B-lr0.00002'),
# # ('0726/result-Qwen3-8B-Base-0726/merged-B2I-lr0.00005','/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/merged-B2I-lr0.00005'),
# # ('0726/result-Qwen3-8B-Base-0726/merged-I2I-lr0.00005','/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/merged-I2I-lr0.00005'),
# ## ('0726/result-Qwen3-8B-Base-0726/merged-I2B-lr0.00005','/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/merged-I2B-lr0.00005'),
# ## ('0726/result-Qwen3-8B-Base-0726/merged-B2B-lr0.00005','/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/merged-B2B-lr0.00005'),
# # ('0726/result-Qwen3-8B-Base-0726/merged-B2I-lr0.0001','/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/merged-B2I-lr0.0001'),
# # ('0726/result-Qwen3-8B-Base-0726/merged-I2I-lr0.0001','/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/merged-I2I-lr0.0001'),
# ## ('0726/result-Qwen3-8B-Base-0726/merged-I2B-lr0.0001','/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/merged-I2B-lr0.0001'),
# ## ('0726/result-Qwen3-8B-Base-0726/merged-B2B-lr0.0001','/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/merged-B2B-lr0.0001'),
# # ('0726/result-Qwen3-8B-Base-0726/merged-B2I-lr0.0002','/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/merged-B2I-lr0.0002'),
# # ('0726/result-Qwen3-8B-Base-0726/merged-I2I-lr0.0002','/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/merged-I2I-lr0.0002'),
# ## ('0726/result-Qwen3-8B-Base-0726/merged-I2B-lr0.0002','/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/merged-I2B-lr0.0002'),
# ## ('0726/result-Qwen3-8B-Base-0726/merged-B2B-lr0.0002','/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/merged-B2B-lr0.0002'),
# # ('0726/result-Qwen3-8B-Base-0726/merged-B2I-lr0.0005','/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/merged-B2I-lr0.0005'),
# # ('0726/result-Qwen3-8B-Base-0726/merged-I2I-lr0.0005','/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/merged-I2I-lr0.0005'),
# ## ('0726/result-Qwen3-8B-Base-0726/merged-I2B-lr0.0005','/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/merged-I2B-lr0.0005'),
# ## ('0726/result-Qwen3-8B-Base-0726/merged-B2B-lr0.0005','/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/merged-B2B-lr0.0005'),
# # ('0726/result-Qwen3-8B-Base-0726/merged-B2I-lr0.001','/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/merged-B2I-lr0.001'),
# # ('0726/result-Qwen3-8B-Base-0726/merged-I2I-lr0.001','/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/merged-I2I-lr0.001'),
# ## ('0726/result-Qwen3-8B-Base-0726/merged-I2B-lr0.001','/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/merged-I2B-lr0.001'),
# ## ('0726/result-Qwen3-8B-Base-0726/merged-B2B-lr0.001','/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/merged-B2B-lr0.001'),
# # ('0726/result-Qwen3-8B-Base-0726/merged-B2I-lr0.002','/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/merged-B2I-lr0.002'),
# # ('0726/result-Qwen3-8B-Base-0726/merged-I2I-lr0.002','/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/merged-I2I-lr0.002'),
# ## ('0726/result-Qwen3-8B-Base-0726/merged-I2B-lr0.002','/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/merged-I2B-lr0.002'),
# ## ('0726/result-Qwen3-8B-Base-0726/merged-B2B-lr0.002','/home/ubuntu/Shadow/results/0726/result-Qwen3-8B-Base-0726/merged-B2B-lr0.002'),

# # please copy this eval_config to opencompass/examples/eval_shadow_202505.py and then run

# cd ./opencompass
# python3 ./run.py ./examples/eval_shadow_202505.py

