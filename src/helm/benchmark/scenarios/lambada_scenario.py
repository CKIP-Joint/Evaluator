import json
import os
import random
from typing import List, Tuple
import pandas as pd
from helm.common.general import ensure_file_downloaded, ensure_directory_exists
from .scenario import Scenario, Instance, Reference, TRAIN_SPLIT, VALID_SPLIT, TEST_SPLIT, CORRECT_TAG, Input, Output


class LAMBADAScenario(Scenario):
    """
    TBD
    """

    name = "lambada"
    description = "Lambada test"
    tags = ["question_answering"]

    def create_prompt(self, sample: dict) -> Tuple[str, List[str]]:
        """
        Given an example in dataset format, create the prompt and the list of
        correct references.
        """
        prompt = " ".join(sample['text'].split()[:-1])

        answers = [sample['text'].split()[-1]]

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
            split_path: str = os.path.join(data_path, f"{split}.jsonl")
            instances.extend(self.get_split_instances(split_path, split=split_tag))

        return instances

