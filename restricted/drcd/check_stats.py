import json
import numpy as np

prompt_lens = []
answer_lens = []

def create_prompt(sample: dict) :
        """
        Given an example in dataset format, create the prompt and the list of
        correct references.
        """
        prompt = "答案在上下文內容中，請回答以下問題：\n\n"
        prompt += f"內容：{sample['context']}\n\n"
        prompt += f"問：{sample['question']}\n\n"
        prompt += f"答："
        answers = [x['text'] for x in sample["answers"]]

        return prompt, answers

with open("restricted/drcd/test.json", encoding="utf-8") as f:
    all_samples = []
    lines = f.readlines()
    for line in lines:
        if len(line) > 0:
            all_samples.append(json.loads(line))

    clean_samples = all_samples

    for sample in clean_samples:
        prompt, answers = create_prompt(sample)
        prompt_lens.append(len(prompt))
        answer_lens.append(np.max(np.array([len(answer) for answer in answers])))

print(len(prompt_lens))  
print(sum(prompt_lens)/len(prompt_lens))
print(np.max(np.array(prompt_lens)))
print(sum(answer_lens)/len(answer_lens))
print(np.max(np.array(answer_lens)))