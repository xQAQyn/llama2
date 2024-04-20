python -m llama_recipes.finetuning \
--dataset "custom_dataset" \
--custom_dataset.file "fintune/dataset.py" \
--use_peft \
--peft_method lora \
--model_name ./llama-2-7b-chat-hf \
--output_dir ./finetune/output