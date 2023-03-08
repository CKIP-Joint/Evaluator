import json
import os
import random
from typing import List, Tuple
import pandas as pd
from helm.common.general import ensure_file_downloaded, ensure_directory_exists
from .scenario import Scenario, Instance, Reference, TRAIN_SPLIT, VALID_SPLIT, TEST_SPLIT, CORRECT_TAG, Input, Output


class TCWSCScenario(Scenario):
    """
    TBD
    """

    name = "tcwsc"
    description = "Winograd Schema Challenge"
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
       

        answers = sample["answers"]
      
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
            sample_ = {
                'context': sample['text'],
                'question': f"請問{sample['target']['span2_text']}是否指{sample['target']['span1_text']}，請回答是或否。",
                'answers': ["是","對","正確","是的"] if sample['label']=='true' else ["不對","不","非","錯","否","錯的"],
            }
            prompt, answers = self.create_prompt(sample_)
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
        splits = {"train": TRAIN_SPLIT, "valid": VALID_SPLIT}
        random.seed(0)  # we pick a random dialogue point to query the model
        for split, split_tag in splits.items():
            split_path: str = os.path.join(data_path, f"{split}.json")
            instances.extend(self.get_split_instances(split_path, split=split_tag))

        return instances



class TCWSC_5_Shot_Scenario(Scenario):
    """
    TBD
    """

    name = "tcwsc_5_shot"
    description = "Winograd Schema Challenge"
    tags = ["question_answering"]

    def create_prompt(self, sample: dict) -> Tuple[str, List[str]]:
        """
        Given an example in dataset format, create the prompt and the list of
        correct references.
        """
        prompt = sample["few_shot_example"]+"\n"
        prompt += "請閱讀以下內容，並回答問題：\n\n"
        prompt += f"內容：{sample['context']}\n\n"
        prompt += f"問：{sample['question']}\n\n"
        prompt += f"答："
        

        answers = sample["answers"]
       
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


        # get few-shot example:
        with open(os.path.join(os.path.dirname(target_file), "5_shot_examples.txt"), encoding="utf-8") as f:
            few_shot_example = ''.join(f.readlines())

        for sample in clean_samples:
            sample_ = {
                'context': sample['text'],
                'question': f"請問{sample['target']['span2_text']}是否指{sample['target']['span1_text']}，請回答是或否。",
                'answers': ["是","對","正確","是的"] if sample['label']=='true' else ["不對","不","非","錯","否","錯的"],
                'few_shot_example':few_shot_example
            }
            prompt, answers = self.create_prompt(sample_)
            instance = Instance(
                input=Input(text=prompt),
                references=[Reference(Output(text=answer), tags=[CORRECT_TAG]) for answer in answers],
                split=split,
            )
            file_instances.append(instance)
        return file_instances


    def get_instances(self) -> List[Instance]:
        data_path: str = os.path.join("restricted", self.name.split('_')[0])
        assert os.path.exists(data_path)

        instances: List[Instance] = []
        #splits = {"train": TRAIN_SPLIT, "test": TEST_SPLIT}
        splits = {"train": TRAIN_SPLIT, "valid": VALID_SPLIT}
        random.seed(0)  # we pick a random dialogue point to query the model
        
        for split, split_tag in splits.items():
            split_path: str = os.path.join(data_path, f"{split}.json")
            instances.extend(self.get_split_instances(split_path, split=split_tag))

        return instances

