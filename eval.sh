#!/bin/bash

# Initialize Conda for script use
eval "$(conda shell.bash hook)"

# Activate the environment
conda activate harness

# Your commands here
echo "Conda environment activated."

#export HF_DATASETS_CACHE=/content/.cache_datasets
#export TRANSFORMERS_CACHE=/content/.cache_transformers

# export HF_TOKEN=
# echo $HF_TOKEN | huggingface-cli login
export NUMEXPR_MAX_THREADS=16
export LANG="en_US.UTF-8"
export LC_ALL="en_US.UTF-8"

#pretrained_model=TinyLlama/TinyLlama-1.1B-Chat-v1.0
pretrained_model=/home/iapp/llama2-7b-finetune-hf
# pretrained_model=/content/.cache_transformers/qwen-7b/checkpoint-21000
# pretrained_model=/content/.cache_transformers/mistral-7b/checkpoint-21000
# pretrained_model=/content/.cache_transformers/llama-2-7b/checkpoint-21000
# pretrained_model=/content/.cache_transformers/sealion-7b/checkpoint-21000 #TODO : MODEL_TYPE

model_args="dtype=bfloat16"
#,load_in_8bit=True ,attn_implementation=flash_attention_2
# gen_kwargs="num_beams=5"

# Done
# tasks=hellaswag,hellaswag_th
# tasks=xcopa_th_nllb,xcopa_th_original,xcopa_th_superai,xcopa_th_scb,xcopa_th_opus

tasks=ted_tran_th_en

# tasks=hellaswag,hellaswag_th,xcopa_th,xquad_th,xnli_th,belebele_tha_Thai

# tasks=xquad_th
# tasks=xnli_th
# tasks=belebele_tha_Thai
# tasks=mgsm_th_native_cot_original,mgsm_th_native_cot_superai,mgsm_th_native_cot_opus,mgsm_th_native_cot_nllb,mgsm_th_native_cot_scb
# tasks=xcopa_th_nllb,xcopa_th_original,xcopa_th_superai,xcopa_th_scb,xcopa_th_opus
# tasks='belebele_tha_Thai_nllb_sent','belebele_tha_Thai_scb_sent','belebele_tha_Thai_opus_sent','belebele_tha_Thai_superai_sent','belebele_tha_Thai_original'
# tasks='belebele_tha_Thai_nllb','belebele_tha_Thai_scb','belebele_tha_Thai_opus','belebele_tha_Thai_superai','belebele_tha_Thai_original'
# tasks='belebele_zho_Hans_nllb_sent','belebele_zho_Hans_scb_sent','belebele_zho_Hans_opus_sent','belebele_zho_Hans_superai_sent','belebele_zho_Hans_original'
# tasks='xnli_th_nllb','xnli_th_scb','xnli_th_opus','xnli_th_superai','xnli_th_original'

# tasks=multiple_choice_defualt_th
# tasks=multiple_choice_defualt_en
# tasks=multiple_choice_defualt_zh
# tasks=multiple_choice_defualt_de
# tasks=generate_default_th
# tasks=generate_default_en
# tasks=generate_default_zh
# tasks=generate_default_de
# tasks=xcopa_default
# tasks=xquad_default
# tasks=belebele_origin

# tasks=xnli_default
# tasks=belebele_default
# tasks=mgsm_default
# tasks='xnli_default','mgsm_default','belebele_default'

model_path_segments=(${pretrained_model//\// })
model_name="${model_path_segments[-2]}-${model_path_segments[-1]}"

num_fewshot=0
output_name=${model_name}_Y$(date +%Y)_${tasks}_${num_fewshot}shot.out
echo "Writing to $output_name"
rm -f ${output_name}

# lm_eval \
accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=${pretrained_model},${model_args},trust_remote_code=True \
    --tasks ${tasks} \
    --num_fewshot ${num_fewshot} \
    --output output/${tasks}/${model_name} \
    --batch_size auto \
    --log_samples \
    &>> ${output_name}
