# ProbMed

[**🌐 Homepage**](https://jackie-2000.github.io/probmed.github.io/) | [**🤗 Dataset**](https://huggingface.co/datasets/rippleripple/ProbMed) | [**🤗 Paper**](https://github.com/eric-ai-lab/ProbMed/) | [**📖 arXiv**](https://github.com/eric-ai-lab/ProbMed/) | [**GitHub**](https://github.com/eric-ai-lab/ProbMed/)


This repo contains the evaluation code for the paper "[Worse than Random? An Embarrassingly Simple Probing Evaluation of Large Multimodal Models in Medical VQA]([https://arxiv.org/pdf/2311.16502.pdf](https://github.com/eric-ai-lab/ProbMed/))"


## Introduction
We introduce the <b>Probing Evaluation for Medical Diagnosis (ProbMed)</b> dataset to rigorously assess LMM performance in medical imaging through <b>probing evaluation</b> and <b>procedural diagnosis</b>. Particularly, probing evaluation features pairing original questions with negation questions with hallucinated attributes, while procedural diagnosis requires reasoning across various diagnostic dimensions for each image, including modality recognition, organ identification, clinical findings, abnormalities, and positional grounding. ProbMed draws from two comprehensive biomedical datasets MedICaT and ChestX-ray14 to compile a diverse set of <b>6,303 images</b>. These images span three modalities (X-ray, MRI, and CT scan) and four organs (abdomen, brain, chest, and spine). After preprocessing, we generated a diverse set of high-quality questions for each image, covering various diagnostic dimensions. This process resulted in a total of <b>57,132 question-answer pairs</b>, averaging 9 pairs per image.

![Alt text](image.png)

## Dataset Creation

MMMU was created to challenge multimodal models with tasks that demand college-level subject knowledge and deliberate reasoning, pushing the boundaries of what these models can achieve in terms of expert-level perception and reasoning. Please refer to our huggingface [**🤗 Dataset**](https://huggingface.co/datasets/MMMU/MMMU/) for more details.

## Evaluation
Please refer to our [eval](eval)
 folder for more details.

## 🏆 Mini-Leaderboard
| Model                          | Val (900) | Test (10.5K) |
|--------------------------------|:---------:|:------------:|
| Expert (Best)                  |   88.6    |      -       |
| Expert (Medium)                |   82.6    |      -       |
| Expert (Worst)                 |   76.2    |      -       |
| GPT-4o*                        | **69.1**  |      -       |
| Gemini 1.5 Pro*                |   59.4    |      -       |
| Gemini 1.0 Ultra*              |   59.4    |      -       |
| Claude 3 Opus*                 |   59.4    |      -       |
| GPT-4V(ision) (Playground)     |   56.8    |   **55.7**   |
| Reka Core*                     |   56.3    |      -       |
| Gemini 1.5 Flash*              |   56.1    |      -       |
| SenseChat-Vision-0423-Preview* |   54.6    |     50.3     |
| Reka Flash*                    |   53.3    |      -       |
| Claude 3 Sonnet*               |   53.1    |      -       |
| HPT Pro*                       |   52.0    |      -       |
| VILA1.5*                       |   51.9    |     46.9     |
| Qwen-VL-MAX*                   |   51.4    |     46.8     |
| InternVL-Chat-V1.2*            |   51.6    |     46.2     |
| Skywork-VL*                    |   51.4    |     46.2     |
| LLaVA-1.6-34B*                 |   51.1    |     44.7     |
| Claude 3 Haiku*                |   50.2    |      -       |
| Adept Fuyu-Heavy*              |   48.3    |      -       |
| Gemini 1.0 Pro*                |   47.9    |      -       |
| Marco-VL-Plus*                 |   46.2    |     44.3     |
| Yi-VL-34B*                     |   45.9    |     41.6     |
| Qwen-VL-PLUS*                  |   45.2    |     40.8     |
| HPT Air*                       |   44.0    |      -       |
| Reka Edge*                     |   42.8    |      -       |
| Marco-VL*                      |   41.2    |     40.4     |
| OmniLMM-12B*                   |   41.1    |     40.4     |
| Weitu-VL-1.0-15B*              |     -     |     38.4     |
| InternLM-XComposer2-VL*        |   43.0    |     38.2     |
| Yi-VL-6B*                      |   39.1    |     37.8     |
| InfiMM-Zephyr-7B*              |   39.4    |     35.5     |
| InternVL-Chat-V1.1*            |   39.1    |     35.3     |
| SVIT*                          |   38.0    |     34.1     |
| MiniCPM-V*                     |   37.2    |     34.1     |
| MiniCPM-V-2*                   |   37.1    |      -       |
| Emu2-Chat*                     |   36.3    |     34.1     |
| BLIP-2 FLAN-T5-XXL             |   35.4    |     34.0     |
| InstructBLIP-T5-XXL            |   35.7    |     33.8     |
| LLaVA-1.5-13B                  |   36.4    |     33.6     |
| Bunny-3B*                      |   38.2    |     33.0     |
| Qwen-VL-7B-Chat                |   35.9    |     32.9     |
| SPHINX*                        |   32.9    |     32.9     |
| mPLUG-OWL2*                    |   32.7    |     32.1     |
| BLIP-2 FLAN-T5-XL              |   34.4    |     31.0     |
| InstructBLIP-T5-XL             |   32.9    |     30.6     |
| Gemini Nano2*                  |   32.6    |      -       |
| CogVLM                         |   32.1    |     30.1     |
| Otter                          |   32.2    |     29.1     |
| LLaMA-Adapter2-7B              |   29.8    |     27.7     |
| MiniGPT4-Vicuna-13B            |   26.8    |     27.6     |
| Adept Fuyu-8B                  |   27.9    |     27.4     |
| Kosmos2                        |   24.4    |     26.6     |
| OpenFlamingo2-9B               |   28.7    |     26.3     |
| Frequent Choice                |   22.1    |     23.9     |
| Random Choice                  |   26.8    |     25.8     |

*: results provided by the authors.


🎯 **We have released a full suite comprising 150 development samples and 900 validation samples. However, the 10,500 test questions are available without their answers.** Use the development set for few-shot/in-context learning, and the validation set for debugging models, selecting hyperparameters, and quick evaluations. The answers and explanations for the test set questions are withheld. You can submit your model's predictions for the **test set** on **[EvalAI](https://eval.ai/web/challenges/challenge-page/2179/overview)**.

## Disclaimers
The guidelines for the annotators emphasized strict compliance with copyright and licensing rules from the initial data source, specifically avoiding materials from websites that forbid copying and redistribution. 
Should you encounter any data samples potentially breaching the copyright or licensing regulations of any site, we encourage you to [contact](#contact) us. Upon verification, such samples will be promptly removed.

## Contact
- Xiang Yue: xiangyue.work@gmail.com
- Yu Su: su.809@osu.edu
- Wenhu Chen: wenhuchen@uwaterloo.ca

## Citation

**BibTeX:**
```bibtex
@inproceedings{yue2023mmmu,
  title={MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI},
  author={Xiang Yue and Yuansheng Ni and Kai Zhang and Tianyu Zheng and Ruoqi Liu and Ge Zhang and Samuel Stevens and Dongfu Jiang and Weiming Ren and Yuxuan Sun and Cong Wei and Botao Yu and Ruibin Yuan and Renliang Sun and Ming Yin and Boyuan Zheng and Zhenzhu Yang and Yibo Liu and Wenhao Huang and Huan Sun and Yu Su and Wenhu Chen},
  booktitle={Proceedings of CVPR},
  year={2024},
}
```
