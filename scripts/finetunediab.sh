# python qlora.py \
#     --model_name_or_path /home/sonia/llama-qlora/models/10/mhllama \
#     --num_heads 10 \
#     --use_auth \
#     --output_dir /mnt/data/sonia/ckpts/diabetes-new \
#     --lora_r 64 \
#     --lora_alpha 16 \
#     --report_to wandb \
#     --logging_steps 2 \
#     --save_strategy steps \
#     --data_seed 42 \
#     --save_steps 5 \
#     --save_total_limit 40 \
#     --eval_dataset_size 5 \
#     --max_eval_samples 100 \
#     --per_device_eval_batch_size 1 \
#     --max_new_tokens 100 \
#     --dataloader_num_workers 1 \
#     --group_by_length \
#     --logging_strategy steps \
#     --remove_unused_columns False \
#     --do_train True \
#     --do_generate False \
#     --eval_samples False \
#     --do_eval False \
#     --do_mmlu_eval False \
#     --diversity False \
#     --divdist manhattan \
#     --lora_modules all \
#     --double_quant \
#     --quant_type nf4 \
#     --bf16 \
#     --bits 4 \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type constant \
#     --gradient_checkpointing \
#     --dataset /mnt/data/sonia/datasets/diabetes/may10.dat \
#     --max_column_len 4 \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 16 \
#     --max_steps 60 \
#     --eval_steps 2 \
#     --learning_rate 0.0002 \
#     --adam_beta2 0.999 \
#     --max_grad_norm 0.3 \
#     --lora_dropout 0.1 \
#     --weight_decay 0.0 \
#     --seed 0 
CUDA_VISIBLE_DEVICES=-1 python sample.py \
    --model_name_or_path /home/sonia/llama-qlora/models/10/mhllama \
    --num_heads 10 \
    --use_auth \
    --output_dir /mnt/data/sonia/ckpts/diabetes-new \
    --lora_r 64 \
    --lora_alpha 16 \
    --report_to wandb \
    --logging_steps 2 \
    --save_strategy steps \
    --data_seed 42 \
    --save_steps 5 \
    --save_total_limit 40 \
    --eval_dataset_size 5 \
    --max_eval_samples 100 \
    --per_device_eval_batch_size 1 \
    --max_new_tokens 100 \
    --dataloader_num_workers 1 \
    --group_by_length \
    --logging_strategy steps \
    --remove_unused_columns False \
    --do_train True \
    --do_generate False \
    --eval_samples False \
    --do_eval False \
    --do_mmlu_eval False \
    --diversity False \
    --divdist manhattan \
    --lora_modules all \
    --double_quant \
    --quant_type nf4 \
    --bf16 \
    --bits 4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type constant \
    --gradient_checkpointing \
    --dataset /mnt/data/sonia/datasets/diabetes/may10.dat \
    --max_column_len 4 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_steps 60 \
    --eval_steps 2 \
    --learning_rate 0.0002 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.1 \
    --weight_decay 0.0 \
    --seed 0 

#     --evaluation_strategy steps \