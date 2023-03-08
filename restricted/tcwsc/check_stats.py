import json
import numpy as np
import pandas as pd
prompt_lens = []
answer_lens = []

def create_prompt(sample: dict):
        """
        Given an example in dataset format, create the prompt and the list of
        correct references.
        """
        prompt = "請閱讀以下內容，並回答問題：\n\n"
        prompt += f"內容：{sample['context']}\n\n"
        prompt += f"問：{sample['question']}\n\n"
        prompt += f"答："
       

        answers = sample["answers"]
      
        return prompt, answers

with open("restricted/tcwsc/valid.json", encoding="utf-8") as f:
    lines = f.readlines()
    all_samples = []
    for line in lines:
        if len(line) > 0:
            all_samples.append(json.loads(line))

    clean_samples = all_samples

    for sample in clean_samples:
        sample_ = {
            'context': sample['text'],
            'question': f"請問{sample['target']['span2_text']}是否指{sample['target']['span1_text']}，請回答是或否。",
            'answers': ["是","對","正確","是的"] if sample['label']=='true' else ["不對","不","非","錯","否","錯的"],
        }
        prompt, answers = create_prompt(sample_)
    
        prompt_lens.append(len(prompt))
        answer_lens.append(np.max(np.array([len(answer) for answer in answers])))

print(len(prompt_lens))  
print(sum(prompt_lens)/len(prompt_lens))
print(np.max(np.array(prompt_lens)))
print(sum(answer_lens)/len(answer_lens))
print(np.max(np.array(answer_lens)))