import json
import numpy as np

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
       

        answers = [sample["answers"]]

        return prompt, answers

with open('restricted/fgc/test.json', "r", encoding="utf-8") as f:
    clean_samples = json.load(f)

    for sample in clean_samples:
        for subsample_question in sample['QUESTIONS']:
            if subsample_question['QTYPE'] == "申論":
                continue
            subsample = {
                'context': sample['DTEXT'],
                'question': subsample_question['QTEXT'],
                'answers': subsample_question['ANSWER'],
            }
            prompt, answers = create_prompt(subsample)
            prompt_lens.append(len(prompt))
            answer_lens.append(len(answers[0]))

print(len(prompt_lens))  
print(sum(prompt_lens)/len(prompt_lens))
print(np.max(np.array(prompt_lens)))
print(sum(answer_lens)/len(answer_lens))
print(np.max(np.array(answer_lens)))