# Evaluation Guidelines
We provide detailed instructions for evaluation. 
To execute our evaluation script, please ensure that the structure of your model outputs is the same as ours.

# Get Started

Clone the official repo of open-sourced models into the following folder:
* LLaVA [[repo]](https://github.com/haotian-liu/LLaVA)
* LLaVA-Med [[repo]](https://github.com/microsoft/LLaVA-Med)
* MiniGPTv2 [[repo]](https://github.com/Vision-CAIR/MiniGPT-4)
* CheXagent [[repo]](https://github.com/Stanford-AIMI/CheXagent)
Move the inference scripts under the **./scripts/inference** to their corresponding path under those folders. (refer to path in model_inference.sh)

For API-based models, GPT-4V and Gemini, set up your API key in the provided inference scripts.

## Categorized Evaluation

After downloading our [dataset](https://huggingface.co/datasets/rippleripple/ProbMed), reorganize the questions by their modality-organ combination under different folders.

```
ct-scan_abdomen
├── question.json
ct-scan_brain
├── question.json
ct-scan_chest
├── question.json
ct-scan_spine
├── question.json
mri_abdomen
├── question.json
...
```

## Model Inference

Set up the environment for each open-sourced model as instructed by their original repo. Run model_inference.sh to get model outputs on provided question files.

After getting the output, use the functions in util.py to get categorical scores.
