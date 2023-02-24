import json
import os
import random
from typing import List, Tuple
import pandas as pd
from helm.common.general import ensure_file_downloaded, ensure_directory_exists
from .scenario import Scenario, Instance, Reference, TRAIN_SPLIT, VALID_SPLIT, TEST_SPLIT, CORRECT_TAG, Input, Output


class FGCScenario(Scenario):
    """
    TBD
    """

    name = "fgc"
    description = "Question answering from Various sources."
    tags = ["question_answering"]

    def create_prompt(self, sample: dict) -> Tuple[str, List[str]]:
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

    def get_split_instances(self, target_file: str, split: str) -> List[Instance]:
        """
        Helper for generating instances for a split.
        Args:
            target_file (str): Data file.
            splits (dict): Which splits to partition the data into.
        Returns:
            List[Instance]: Instances from the file for the specified split.
        """
        file_instances: List[Instance] = []
        all_samples = []
        assert os.path.exists(target_file), f"File {target_file} not exists!"
        
        with open(target_file, "r", encoding="utf-8") as f:
            all_samples = json.load(f)

        clean_samples = all_samples

        for sample in clean_samples:
            for subsample_question in sample['QUESTIONS']:
                if subsample_question['QTYPE'] == "申論":
                    continue
                subsample = {
                    'context': sample['DTEXT'],
                    'question': subsample_question['QTEXT'],
                    'answers': subsample_question['ANSWER'],
                }
                prompt, answers = self.create_prompt(subsample)
                instance = Instance(
                    input=Input(text=prompt),
                    references=[Reference(Output(text=answer), tags=[CORRECT_TAG]) for answer in answers],
                    split=split,
                )
                file_instances.append(instance)
        return file_instances


    def get_instances(self) -> List[Instance]:
        data_path: str = os.path.join("restricted", self.name)
        assert os.path.exists(data_path)

        instances: List[Instance] = []
        #splits = {"train": TRAIN_SPLIT, "test": TEST_SPLIT}
        splits = {"test": TEST_SPLIT}
        random.seed(0)  # we pick a random dialogue point to query the model
        for split, split_tag in splits.items():
            split_path: str = os.path.join(data_path, f"{split}.json")
            instances.extend(self.get_split_instances(split_path, split=split_tag))

        return instances




class FGC_1_Shot_Scenario(Scenario):
    """
    TBD
    """

    name = "fgc"
    description = "Question answering from Various sources."
    tags = ["question_answering"]

    def create_prompt(self, sample: dict) -> Tuple[str, List[str]]:
        """
        Given an example in dataset format, create the prompt and the list of
        correct references.
        """
        if sample["answers"] in ["否", "是"]:
            few_shot_example = sample['few_shot_example']['yesno']
        elif '、' in sample["answers"] and ( \
            '以及' in sample["answers"] or \
            '與' in sample["answers"] or \
            '及' in sample["answers"] or \
            '和' in sample["answers"]):
            few_shot_example = sample['few_shot_example']['comma']
        else:
            few_shot_example = sample['few_shot_example']['general']

        prompt = few_shot_example+"\n\n"
        prompt += "請閱讀以下內容，並回答問題：\n\n"
        prompt += f"內容：{sample['context']}\n\n"
        prompt += f"問：{sample['question']}\n\n"
        prompt += f"答："
       

        answers = [sample["answers"]]

        return prompt, answers

    def get_split_instances(self, target_file: str, split: str) -> List[Instance]:
        """
        Helper for generating instances for a split.
        Args:
            target_file (str): Data file.
            splits (dict): Which splits to partition the data into.
        Returns:
            List[Instance]: Instances from the file for the specified split.
        """
        file_instances: List[Instance] = []
        all_samples = []
        assert os.path.exists(target_file), f"File {target_file} not exists!"
        
        with open(target_file, "r", encoding="utf-8") as f:
            all_samples = json.load(f)

        clean_samples = all_samples

        # get few-shot example:
        with open(os.path.join(os.path.dirname(target_file), "1_shot_examples.txt"), encoding="utf-8") as f:
            few_shot_example = ''.join(f.readlines())

        few_shot_example_dict = {
            'general': few_shot_example.split("general")[1].split("yesno")[0].strip(), 
            'yesno': few_shot_example.split("yesno")[1].split('comma')[0].strip(), 
            'comma': few_shot_example.split("yesno")[1].split('comma')[1].strip(), 
        }

        for sample in clean_samples:
            for subsample_question in sample['QUESTIONS']:
                if subsample_question['QTYPE'] == "申論":
                    continue
                subsample = {
                    'context': sample['DTEXT'],
                    'question': subsample_question['QTEXT'],
                    'answers': subsample_question['ANSWER'],
                    'few_shot_example':few_shot_example_dict
                }
                prompt, answers = self.create_prompt(subsample)
                instance = Instance(
                    input=Input(text=prompt),
                    references=[Reference(Output(text=answer), tags=[CORRECT_TAG]) for answer in answers],
                    split=split,
                )
                file_instances.append(instance)
        return file_instances


    def get_instances(self) -> List[Instance]:
        data_path: str = os.path.join("restricted", self.name)
        assert os.path.exists(data_path)

        instances: List[Instance] = []
        #splits = {"train": TRAIN_SPLIT, "test": TEST_SPLIT}
        splits = {"test": TEST_SPLIT}
        random.seed(0)  # we pick a random dialogue point to query the model
        for split, split_tag in splits.items():
            split_path: str = os.path.join(data_path, f"{split}.json")
            instances.extend(self.get_split_instances(split_path, split=split_tag))

        return instances
