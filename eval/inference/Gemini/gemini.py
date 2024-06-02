import json
from typing import Optional
import fire
import os
import asyncio
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm
import google.generativeai as genai
import PIL.Image

class Gemini():
    def __init__(self, image_folder):
        self.image_folder = image_folder
        api_key = "your api key"
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro-vision')
    
    async def process(self, data, i):
        idx = data["id"]
        gpt_idx = data["gpt_idx"]
        qa_type = data["qa_type"]
        answer = data["answer"]
        qs = data["question"]
        image_file = data["image"]
        try:
            response = await self.model.generate_content_async([qs, PIL.Image.open(self.image_folder + image_file)])
            return {
                    "i": i, 
                    "data" : {
                        "id": idx,
                        "gpt_idx": gpt_idx,
                        "qa_type": qa_type,
                        "question": qs,
                        "gt_ans": answer,
                        "response": response.text
                        }
                    }
        except Exception as e:
                print(f"An error occurred: {str(e)}")
                return None
        
    def label(self, meta_data: list[dict]) -> list[dict]:
        return asyncio.run(self.label_async(meta_data))
        
    async def label_async(self, meta_data):
        results = []

        tasks = [self.process(data, i) for i, data in enumerate(meta_data)]
        for task in async_tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"generate responses"):
            result = await task
            if result is not None:
                results.append(result)
        
        results = sorted(results, key=lambda x: x['i'])
        return results

def main(
    question_file: Optional[str] = "xx.json",
    answers_file: Optional[str] = "xx.json",
    image_folder: Optional[str] = "image/folder"
):

    labeler = Gemini(image_folder)

    with open(question_file, 'r') as f:
        question_data = json.load(f)

    indices = list(range(len(question_data)))

    # assign global index
    for i, _ in enumerate(question_data):
        question_data[i]['gpt_idx'] = i

    results = labeler.label(meta_data=question_data)
    results = [r['data'] for r in results]

    # indices
    for data in results:
        indices.remove(data['gpt_idx'])
    
    no_effect = 0
    while (len(indices) > 0):
        before_count = len(indices)
        print(f"There are {len(indices)} left")
        meta_data = [question_data[i] for i in indices]
        tmp_results = labeler.label(meta_data=meta_data)
        tmp_results = [r['data'] for r in tmp_results]
        results.extend(tmp_results)
        # indices
        for data in tmp_results:
            indices.remove(data['gpt_idx'])
        after_count = len(indices)
        if after_count == before_count:
            no_effect += 1
        else:
            no_effect = 0
        if no_effect >= 3:
            break
    
    results = sorted(results, key=lambda x: x['gpt_idx'])
    
    with open(answers_file, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    fire.Fire(main)
    