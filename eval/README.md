# Evaluation Guidelines
We provide detailed instructions for evaluation. To execute our evaluation script, please ensure that the structure of your model outputs is the same as ours.

## Categorized Evaluation

After downloading our [dataset](https://huggingface.co/datasets/rippleripple/ProbMed), reorganize the questions by their modality-organ combination under different folders. Refer to data statistics (Table 2) in the paper to confirm the correct number of questions per modality-organ.

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

Clone the official repo of open-sourced models into the following folder:
* LLaVAv1, v1.6 [[repo]](https://github.com/haotian-liu/LLaVA)
* LLaVA-Med [[repo]](https://github.com/microsoft/LLaVA-Med)
* MiniGPTv2 [[repo]](https://github.com/Vision-CAIR/MiniGPT-4)
* CheXagent [[repo]](https://github.com/Stanford-AIMI/CheXagent)

Set up the environment for each open-sourced model as instructed by their original repo. Move the inference scripts to their corresponding folders under the **./scripts/inference** folder. (refer to path in model_inference.sh). For API-based models, GPT-4V and Gemini, set up your API key in the provided inference scripts.

Run model_inference.sh to get model outputs on the question files you get from the previous step.


## Evaluation results

After getting the output, use the functions in util.py to get categorical scores.
