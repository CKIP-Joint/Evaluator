import json
import numpy as np
import pandas as pd
prompt_lens = []
answer_lens = []

def process_flud_item(item):
    # Note: question concept field is not used: item["question"]["question_concept"]
    letter2idx = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    question = item["文章"] + "\n\n" + item["問題"]
    answers = [item[f"選項{i+1}"] for i in range(4) ]
    correct_answer = answers[int(item["正確答案"])-1]

    return question, answers, correct_answer

data = pd.read_csv("restricted/flud/test.csv").dropna()
instances = []

converted_data = []
for item_idx in range(len(data)):
    item = data.iloc[item_idx]
    item["選項1"] = str(item["選項1"])
    item["選項2"] = str(item["選項2"])
    item["選項3"] = str(item["選項3"])
    item["選項4"] = str(item["選項4"])
    if len(item["選項1"]) > 0 and len(item["選項2"]) > 0 and len(item["選項3"]) > 0 and len(item["選項4"]) > 0:
        converted_data.append(process_flud_item(data.iloc[item_idx]))
for _, (question, answers, correct_answer) in enumerate(converted_data):
    
    prompt_lens.append(len(question) + sum([len(answer) for answer in answers]))
    answer_lens.append(np.max(np.array([len(answer) for answer in answers])))

print(len(prompt_lens))  
print(sum(prompt_lens)/len(prompt_lens))
print(np.max(np.array(prompt_lens)))
print(sum(answer_lens)/len(answer_lens))
print(np.max(np.array(answer_lens)))