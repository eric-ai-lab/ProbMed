# Welcome to the codebase for: Worse than Random? An Embarrassingly Simple Probing Evaluation of Large Multimodal Models in Medical VQA!


# Get Started

Clone the official repo of open-sourced models into the following folder:
* LLaVA [[repo]](https://github.com/haotian-liu/LLaVA)
* LLaVA-Med [[repo]](https://github.com/microsoft/LLaVA-Med)
* MiniGPTv2 [[repo]](https://github.com/Vision-CAIR/MiniGPT-4)
* CheXagent [[repo]](https://github.com/Stanford-AIMI/CheXagent)
Move the inference scripts under the **./scripts/inference** to their corresponding path under those folders. (refer to path in model_inference.sh)

For API-based models, GPT-4V and Gemini, set up your API key in the provided inference scripts.

## Provided Dataset

**ProbMed_vqa.json** : 57,132 question-answer pairs spanning three modalities (X-ray, MRI, and CT scan) and four organs (abdomen, brain, chest, and spine). Before using, reorganize the QA pairs according to their image modality and organs.
├── ct-scan_abdomen
│   └── question.json
├── ct-scan_abdomen
│   └── question.json
**ablation_RAD-VQA_vqa.json**: 236 question-answer pairs constructed from 118 test instances where the answer is "yes" out of 272 closed-ended question-answers pairs within RAD-VQA test set.

## Model Inference

Set up the environment for each open-sourced model as instructed by their original repo. Run model_inference.sh to get model outputs on provided question files.

After getting the output, use the functions in util.py to get categorical scores.