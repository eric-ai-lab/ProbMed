#!/bin/bash
set -e

model_name=$1

if [ "${model_name}" == "llavamed" ]; then
    cd ../LLaVA-Med
elif [ "${model_name}" == "llava_v1" ]; then
    cd ../LLaVA
elif [ "${model_name}" == "llava_v1.6" ]; then
    cd ../LLaVA
elif [ "${model_name}" == "minigptv2" ]; then
    cd ../MiniGPT-4
elif [ "${model_name}" == "chexagent" ]; then
    cd ../CheXagent
elif [ "${model_name}" == "gpt4v" ]; then
    cd ../gpt4V
elif [ "${model_name}" == "gemini" ]; then
    cd ../gemini
elif [ "${model_name}" == "gpt4o" ]; then
    cd ../gpt4V
elif [ "${model_name}" == "med-flamingo" ]; then
    cd ../med-flamingo
elif [ "${model_name}" == "biomedgpt" ]; then
    cd ../BiomedGPT
fi

echo "=========================================="

# inference for probmed results
question_file="path to question file" # */probmed.json
answer_file="./response_file/${model_name}"
answer_file_json="./response_file/${model_name}.json"

# uncomment the following block if you are running inference for ablation study
# question_file="/data3/qianqi/medHVL/vqa/ablation/ablation_question.json"
# answer_file="/data3/qianqi/medHVL/vqa/ablation/${model_name}"
# answer_file_json="/data3/qianqi/medHVL/vqa/ablation/${model_name}.json"

if [ "${model_name}" == "llavamed" ]; then  
    python llava/eval/run_med_datasets_eval_batch.py \
    --num-chunks 4 \
    --model-name /model_weights/llavamed/llava_med_in_text_60k \
    --question-file ${question_file} \
    --answers-file ${answer_file}

    rm ${answer_file}-*

elif [ "${model_name}" == "llava_v1" ]; then
    python llava/eval/run_eval_batch.py \
    --num-chunks 4 \
    --model-name /model_weights/llava/llava_v1 \
    --image-folder ${image_foler} \
    --question-file ${question_file} \
    --answers-file ${answer_file}

    rm ${answer_file}-*

elif [ "${model_name}" == "llava_v1.6" ]; then
    python llava/eval/run_eval_batch.py \
    --num-chunks 4 \
    --model-name /model_weights/llava/llava-v1.6-vicuna-7b \
    --image-folder ${image_foler} \
    --question-file ${question_file} \
    --answers-file ${answer_file}

    rm ${answer_file}-*

elif [ "${model_name}" == "minigptv2" ]; then
    python run_eval_batch.py \
    --num-chunks 4 \
    --cfg-path eval_configs/minigptv2_eval.yaml \
    --image-folder ${image_foler} \
    --question-file ${question_file} \
    --answers-file ${answer_file}

    rm ${answer_file}-*

elif [ "${model_name}" == "chexagent" ]; then
    python run_eval_batch.py \
    --num-chunks 4 \
    --image-folder ${image_foler} \
    --question-file ${question_file} \
    --answers-file ${answer_file}

    rm ${answer_file}-*

elif [ "${model_name}" == "gpt4v" ]; then
    python gpt4v.py \
    --image-folder ${image_foler} \
    --question-file ${question_file} \
    --answers-file ${answer_file_json}

elif [ "${model_name}" == "gemini" ]; then
    python run_eval_batch.py \
    --num-chunks 4 \
    --image-folder ${image_foler} \
    --question-file ${question_file} \
    --answers-file ${answer_file}

    rm ${answer_file}-*
    rm ${answer_file}_*

elif [ "${model_name}" == "gpt4o" ]; then
    python gpt4v.py \
    --image-folder ${image_foler} \
    --question-file ${question_file} \
    --answers-file ${answer_file_json}

elif [ "${model_name}" == "med-flamingo" ]; then
    python scripts/run_eval_batch.py \
    --num-chunks 3 \
    --question-file ${question_file} \
    --answers-file ${answer_file}

    rm ${answer_file}-*

elif [ "${model_name}" == "biomedgpt" ]; then
    python evaluate.py \
    ablation.tsv \
    --path /model_weights/biomedgpt_base.pt \
    --user-dir module \
    --task vqa_gen \
    --batch-size 64 \
    --log-format simple --log-interval 10 \
    --seed 7 \
    --gen-subset ablation \
    --results-path ../ablation \
    --fp16 \
    --beam-search-vqa-eval \
    --ema-eval \
    --unnormalized \
    --temperature 1.0 \
    --num-workers 0 \
    --model-overrides "{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}"

fi

echo "=========================================="
