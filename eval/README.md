# Evaluation Guidelines
We provide detailed instructions for evaluation. To execute our evaluation script, please ensure that the structure of your model outputs is the same as ours.

## Model Inference

Download our [dataset](https://huggingface.co/datasets/rippleripple/ProbMed) from huggingface.

Clone the official repo of open-sourced models into the following folder:
* LLaVAv1, v1.6 [[repo]](https://github.com/haotian-liu/LLaVA)
* LLaVA-Med [[repo]](https://github.com/microsoft/LLaVA-Med)
* MiniGPTv2 [[repo]](https://github.com/Vision-CAIR/MiniGPT-4)
* CheXagent [[repo]](https://github.com/Stanford-AIMI/CheXagent)
* BiomedGPT [[repo]](https://github.com/taokz/BiomedGPT)
* Med-Flamingo [[repo]](https://github.com/snap-stanford/med-flamingo)

Set up the environment for each open-sourced model as instructed by their original repo and run inference. For API-based models: GPT-4o, GPT-4V, and Gemini Pro set up your API key in the provided scripts under the /inference folder.

For the open-source models, we also provide our inference scripts for your reference. To utilize those, move the inference scripts under the /inference folder to the corresponding folders you clone from the original repos by referring to the path in model_inference.sh.

After setting up, run inference.sh to get model outputs on the question files.


## Get Evaluation results and scores

After getting the output, run calculate_score.py to get scores for all models.

Your folder structure should look like this:

    .
    ├── ...
    ├── docs                    # Documentation files (alternatively `doc`)
    │   ├── TOC.md              # Table of contents
    │   ├── faq.md              # Frequently asked questions
    │   ├── misc.md             # Miscellaneous information
    │   ├── usage.md            # Getting started guide
    │   └── ...                 # etc.
    └── ...
    
project-root\
├── LLaVA\
│   └── ...\
├── LLaVA-Med\
│   └── ...\
└── ...\
│\
├── probmed.json\
├── response_file\
│   └── llava_v1.json\
│   └── llavamed.json\
│   └── xxx.json\
├── ablation\
│   └── ablation.json\
│   └── llava_v1.json\
│   └── llavamed.json\
│   └── xxx.json\
