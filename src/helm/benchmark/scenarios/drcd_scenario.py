import json
import os
import random
from typing import Dict, List, Tuple

from helm.common.general import ensure_file_downloaded, ensure_directory_exists
from .scenario import Scenario, Instance, Reference, TRAIN_SPLIT, VALID_SPLIT, TEST_SPLIT, CORRECT_TAG, Input, Output


class DRCDScenario(Scenario):
    """
    TBD
    """

    name = "drcd"
    description = "Question answering from Wikipedia."
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
        answers = [x['text'] for x in sample["answers"]]

        return prompt, answers

    def get_file_instances(self, target_file: str) -> List[Instance]:
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
        with open(target_file, encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            if len(line) > 0:
                all_samples.append(json.loads(line))

        clean_samples = all_samples

        for sample in clean_samples:
            prompt, answers = self.create_prompt(sample)
            instance = Instance(
                input=Input(text=prompt),
                references=[Reference(Output(text=answer), tags=[CORRECT_TAG]) for answer in answers],
                split=TEST_SPLIT,
            )
            file_instances.append(instance)
        return file_instances


    def get_instances(self) -> List[Instance]:
        file_path: str = os.path.join("restricted", self.name, "test.json")
        assert os.path.exists(file_path)
        random.seed(0)  # randomness needed to pick question at random
        
        instances: List[Instance] = []
        instances: List[Instance] = self.get_file_instances(target_file=file_path)
        return instances




class DRCD_Extractive_Scenario(Scenario):
    """
    TBD
    """

    name = "drcd_extractive"
    description = "Question answering from Wikipedia."
    tags = ["question_answering"]

    def create_prompt(self, sample: dict) -> Tuple[str, List[str]]:
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

    def get_file_instances(self, target_file: str) -> List[Instance]:
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
        with open(target_file, encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            if len(line) > 0:
                all_samples.append(json.loads(line))

        clean_samples = all_samples

        for sample in clean_samples:
            prompt, answers = self.create_prompt(sample)
            instance = Instance(
                input=Input(text=prompt),
                references=[Reference(Output(text=answer), tags=[CORRECT_TAG]) for answer in answers],
                split=TEST_SPLIT,
            )
            file_instances.append(instance)
        return file_instances


    def get_instances(self) -> List[Instance]:
        file_path: str = os.path.join("restricted", self.name, "test.json")
        assert os.path.exists(file_path)
        random.seed(0)  # randomness needed to pick question at random
        
        instances: List[Instance] = []
        instances: List[Instance] = self.get_file_instances(target_file=file_path)
        return instances

