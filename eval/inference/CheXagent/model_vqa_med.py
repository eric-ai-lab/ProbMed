import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import os
import json
from tqdm import tqdm
import io

import requests
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

from PIL import Image
import random
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def eval_model(args):
    # Model
    # step 1: Setup constant
    device = "cuda"
    dtype = torch.float16

    # step 2: Load Processor and Model
    processor = AutoProcessor.from_pretrained("path/to/CheXagent", trust_remote_code=True)
    generation_config = GenerationConfig.from_pretrained("path/to/CheXagent")
    model = AutoModelForCausalLM.from_pretrained("path/to/CheXagent", torch_dtype=dtype, trust_remote_code=True)
    model = model.cuda().half()

    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    # questions = get_chunk(questions, args.num_chunks, args.chunk_idx - 1)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(answers_file), "images"), exist_ok=True)
    ans_file = open(answers_file, "w")
    save_image_folder = os.path.join(os.path.dirname(os.path.expanduser(args.answers_file)), "images")
    for i, line in enumerate(tqdm(questions)):
        idx = line["id"]
        qa_type = line["qa_type"]
        answer = line["answer"]
        qs = line["question"]

        qs = qs.replace('<image>', '').strip()
        cur_prompt = qs

        image_file = line["image"]
        image = Image.open(os.path.join(args.image_folder, image_file))
        inputs = processor(images=image, text=f" USER: <s>{cur_prompt} ASSISTANT: <s>", return_tensors="pt").to(device=device, dtype=dtype)

        output = model.generate(**inputs, generation_config=generation_config)[0]
        response = processor.tokenizer.decode(output, skip_special_tokens=True)

        ans_file.write(json.dumps({"id": idx,
                                   "qa_type": qa_type,
                                   "question": cur_prompt,
                                   "gt_ans": answer,
                                   "response": response}) + "\n")
        # ans_file.write(json.dumps({"id": idx,
        #                            "prompt": cur_prompt,
        #                            "text": outputs,
        #                            "answer_id": ans_id,
        #                            "model_id": model_name,
        #                            "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--mm-projector", type=str, default=None)
    parser.add_argument("--vision-tower", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="simple")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--answer-prompter", action="store_true")
    args = parser.parse_args()

    eval_model(args)
