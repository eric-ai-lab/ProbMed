# ProbMed

[**🌐 Homepage**](https://jackie-2000.github.io/probmed.github.io/) | [**🤗 Dataset**](https://huggingface.co/datasets/rippleripple/ProbMed) | [**🤗 Paper**](https://arxiv.org/pdf/2405.20421) | [**📖 arXiv**](https://arxiv.org/abs/2405.20421) | [**GitHub**](https://github.com/eric-ai-lab/ProbMed/)


This repo contains the evaluation code for the paper "[Worse than Random? An Embarrassingly Simple Probing Evaluation of Large Multimodal Models in Medical VQA]([https://arxiv.org/pdf/2311.16502.pdf](https://github.com/eric-ai-lab/ProbMed/))"


## Introduction
We introduce the <b>Probing Evaluation for Medical Diagnosis (ProbMed)</b> dataset to rigorously assess LMM performance in medical imaging through <b>probing evaluation</b> and <b>procedural diagnosis</b>. Particularly, probing evaluation features pairing original questions with negation questions with hallucinated attributes, while procedural diagnosis requires reasoning across various diagnostic dimensions for each image, including modality recognition, organ identification, clinical findings, abnormalities, and positional grounding. ProbMed draws from two comprehensive biomedical datasets MedICaT and ChestX-ray14 to compile a diverse set of <b>6,303 images</b>. These images span three modalities (X-ray, MRI, and CT scan) and four organs (abdomen, brain, chest, and spine). After preprocessing, we generated a diverse set of high-quality questions for each image, covering various diagnostic dimensions. This process resulted in a total of <b>57,132 question-answer pairs</b>, averaging 9 pairs per image.

![Alt text](image.png)

## Dataset Creation

ProbMed was created to rigorously evaluate LMMs’ readiness for real-life diagnostic tasks, particularly under adversarial conditions. Please refer to our huggingface [**🤗 Dataset**](https://huggingface.co/datasets/rippleripple/ProbMed) for more details.

## Evaluation
Please refer to our [eval](eval)
 folder for more details.

## 🏆 Leaderboard
| Model           | Modality  | Organ     | Abnormality | Condition/Finding | Position | Overall |
|-----------------|:---------:|:---------:|:-----------:|:-----------------:|:--------:|:-------:|
| Random Choice   | 25.00	    | 25.00	    | 50.00	      | **35.67**	        | **36.48**| 32.13   |
| GPT-4o          | **97.42**	| 69.46     | 61.79	      | 29.30	            | 24.06    | 55.60   |
| GPT-4V          | 92.51	    | 71.73	    | 53.30	      | 35.19	            | 22.40    | 55.28   |
| Gemini 1.5 Pro  | 96.47     | 75.69	    | 62.59	      | 27.93	            | 17.54    | 55.08   |
| Med-Flamingo    | 44.15     | 61.39	    | 50.00	      | 26.33	            | 5.65     | 35.66   |
| CheXagent       | 37.25	    | 33.95	    | **73.31**	  | 28.52	            | 7.48     | 30.61   |
| BiomedGPT       | 60.25	    | 46.81	    | 50.31	      | 14.13	            | 6.11     | 33.34   |
| LLaVA-Med       | 5.48	     | 32.96	    | 38.76	      | 20.38	            | 5.33     | 17.90   |
| MiniGPT-v2      | 3.25	     | 76.26	    | 50.08	      | 15.23	            | 7.96     | 27.67   |
| LLaVA-v1.6 (7B) | 6.77	     | **80.70**	| 46.18	      | 3.56	             | 1.21     | 24.96   |
| LLaVA-v1 (7B)   | 25.27	    | 40.53	    | 50.00	      | 0.34		            | 0.11     | 19.30   |

## Contact
- Qianqi Yan: qyan79@ucsc.edu
- Xin Eric Wang: xwang366@ucsc.edu

## Citation

**BibTeX:**
```bibtex
@misc{yan2024worse,
      title={Worse than Random? An Embarrassingly Simple Probing Evaluation of Large Multimodal Models in Medical VQA}, 
      author={Qianqi Yan and Xuehai He and Xiang Yue and Xin Eric Wang},
      year={2024},
      eprint={2405.20421},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```
