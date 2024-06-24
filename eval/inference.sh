#!/bin/bash
set -e

(
source activate llava-med || conda activate llava-med
./model_inference.sh llavamed
conda deactivate
)

(
source activate llava || conda activate llava
./model_inference.sh llava_v1
conda deactivate
)

(
source activate llava || conda activate llava
./model_inference.sh llava_v1.6
conda deactivate
)

(
source activate minigptv || conda activate minigptv
./model_inference.sh minigptv2
conda deactivate
)

(
source activate llama || conda activate llama
./model_inference.sh chexagent
conda deactivate
)

(
source activate llama || conda activate llama
./model_inference.sh gpt4v
conda deactivate
)

(
source activate llama || conda activate llama
./model_inference.sh gemini
conda deactivate
)

(
source activate llama || conda activate llama
./model_inference.sh gpt4o
conda deactivate
)

(
source activate med-flamingo || conda activate med-flamingo
./model_inference.sh med-flamingo
conda deactivate
)

(
source activate biomedgpt || conda activate biomedgpt
./model_inference.sh biomedgpt
conda deactivate
)
