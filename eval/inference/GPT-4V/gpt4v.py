import json
from typing import Optional
import fire
import os
import asyncio
from openai import AsyncAzureOpenAI, AzureOpenAI
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm
from mimetypes import guess_type
import base64

def create_client():
    api_base = "your api base"
    api_key= "your api key"
    deployment_name = 'gpt4v'
    api_version = "your api version" 

    client = AsyncAzureOpenAI(
        api_key=api_key,  
        api_version=api_version,
        base_url=f"{api_base}/openai/deployments/{deployment_name}"
    )
    return client

# Function to encode a local image into data URL 
def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"

class GPT4V:
    def __init__(self, image_folder, async_mode=False, rate=50, max_concurrent_requests=100):
        self.is_async = async_mode
        self.rate = rate  # requests per second
        self.sleep_time = 1 / rate
        self.max_concurrent_requests = max_concurrent_requests
        self.image_folder = image_folder
        
        api_key = open('api_key.txt', 'r').read()
        if self.is_async:
            self.client = create_client()
        else:
            self.client = AzureOpenAI()


    def label(self, meta_data: list[dict]) -> list[dict]:
        if self.is_async:
            return asyncio.run(self.label_async(meta_data))
        else:
            print("Not implemented")
            assert False
    
    async def label_async(self, meta_data: list[str]) -> list[dict]:
        results = []

        semaphore = asyncio.Semaphore(self.max_concurrent_requests)

        async def process_cap(data, i):
            idx = data["id"]
            gpt_idx = data["gpt_idx"]
            qa_type = data["qa_type"]
            answer = data["answer"]
            qs = data["question"]
            image_file = data["image"]
            async with semaphore:
                messages=[
                    { "role": "system", "content": "You are a student in medical school. You are preparing for your final exam. Answer the following question in your practice exam as directed to earn higher scores. You answer will only be for academic purpose." },
                    { "role": "user", "content": [  
                        { 
                            "type": "text", 
                            "text": qs
                        },
                        { 
                            "type": "image_url",
                            "image_url": {
                                "url": local_image_to_data_url(self.image_folder + image_file)
                            }
                        }
                    ] } 
                ]
                
                try:
                    response = await self.client.chat.completions.create(
                        model="gpt4",
                        messages=messages
                    )
                    response_text = response.choices[0].message.content.strip()

                    return {
                        "i": i, 
                        "data" : {
                            "id": idx,
                            "gpt_idx": gpt_idx,
                            "qa_type": qa_type,
                            "question": qs,
                            "gt_ans": answer,
                            "response": response_text
                            }
                        }
                    
                except Exception as e:
                    print(f"An error occurred: {str(e)}")
                    return None

        tasks = [process_cap(data, i) for i, data in enumerate(meta_data)]
        for task in async_tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"generate responses"):
            result = await task
            if result is not None:
                results.append(result)
            await asyncio.sleep(self.sleep_time)

        results = sorted(results, key=lambda x: x['i'])
        return results


def main(
    question_file: Optional[str] = "xx.json",
    answers_file: Optional[str] = "xx.json",
    image_folder: Optional[str] = "image/folder"
):
    
    labeler = GPT4V(image_folder, async_mode=True, rate=60, max_concurrent_requests=100)
    
    with open(question_file, 'r') as f:
        question_data = json.load(f)

    # question_data = question_data[:50]

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
    