import json
import os
from typing import List
import pandas as pd
from helm.common.general import ensure_file_downloaded, ensure_directory_exists
from helm.common.hierarchical_logger import hlog
from .scenario import (
    Scenario,
    Instance,
    Reference,
    TRAIN_SPLIT,
    VALID_SPLIT,
    TEST_SPLIT,
    CORRECT_TAG,
    Input,
    Output,
)


class FLUDScenario(Scenario):
    name = "flud"
    description = "flud"
    tags = ["multiple_choice"]

    @staticmethod
    def process_flud_item(item):
        # Note: question concept field is not used: item["question"]["question_concept"]
        question = item["文章"] + "\n\n" + item["問題"]
        answers = [chr(ord("A") + i)+'. ' + item[f"選項{i+1}"] for i in range(4) ]
        correct_answer = answers[int(item["正確答案"])-1]

        return question, answers, correct_answer

    def get_instances(self) -> List[Instance]:
        data_path = os.path.join('restricted', self.name, 'test.csv')
        data = pd.read_csv(data_path).dropna()


        instances: List[Instance] = []

        def answer_to_reference(answer: str) -> Reference:
            return Reference(Output(text=answer), tags=[CORRECT_TAG] if answer == correct_answer else [])

        converted_data = []
        for item_idx in range(len(data)):
            item = data.iloc[item_idx]
            item["選項1"] = str(item["選項1"])
            item["選項2"] = str(item["選項2"])
            item["選項3"] = str(item["選項3"])
            item["選項4"] = str(item["選項4"])
            if len(item["選項1"]) > 0 and len(item["選項2"]) > 0 and len(item["選項3"]) > 0 and len(item["選項4"]) > 0:
                converted_data.append(self.process_flud_item(data.iloc[item_idx]))
        for _, (question, answers, correct_answer) in enumerate(converted_data):
            instance = Instance(
                input=Input(text=question),
                references=list(map(answer_to_reference, answers)),
                split=TEST_SPLIT,
            )
            instances.append(instance)
        return instances
