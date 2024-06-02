#!/bin/bash
set -e

image_types=(ct-scan mri x-ray)
body_parts=(abdomen brain chest spine)

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
fi

for image_type in "${image_types[@]}"; do
for body_part in "${body_parts[@]}"; do


echo "=========================================="
echo "${model_name} begin running for ${body_part} ${image_type}"

question_file=set/this/to/question/file/path
answer_file=set/this/to/answer/file/path
answer_file_json=set/this/to/answer/file/path.json
image_foler=set/this/to/image/folder/path

if [ "${model_name}" == "llavamed" ]; then  
    python llava/eval/run_med_datasets_eval_batch.py \
    --num-chunks 4 \
    --model-name set/this/to/model/checkpoint \
    --image-folder ${image_foler} \
    --question-file ${question_file} \
    --answers-file ${answer_file}

    rm ${answer_file}-*

elif [ "${model_name}" == "llava_v1" ]; then
    python llava/eval/run_eval_batch.py \
    --num-chunks 4 \
    --model-name set/this/to/model/checkpoint \
    --image-folder ${image_foler} \
    --question-file ${question_file} \
    --answers-file ${answer_file}

    rm ${answer_file}-*

elif [ "${model_name}" == "llava_v1.6" ]; then
    python llava/eval/run_eval_batch.py \
    --num-chunks 4 \
    --model-name set/this/to/model/checkpoint \
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
fi

echo "${model_name} end running for ${body_part} ${image_type}"
echo "=========================================="

done
done