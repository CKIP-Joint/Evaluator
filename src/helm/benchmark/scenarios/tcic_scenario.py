import json
import os
import random
from typing import List, Tuple
import pandas as pd
from helm.common.general import ensure_file_downloaded, ensure_directory_exists
from .scenario import Scenario, Instance, Reference, TRAIN_SPLIT, VALID_SPLIT, TEST_SPLIT, CORRECT_TAG, Input, Output


class TCICScenario(Scenario):
    """
    TBD
    """

    name = "tcic"
    description = "Idiom cloze answering."
    tags = ["question_answering"]

    def create_prompt(self, sample: dict) -> Tuple[str, List[str]]:
        """
        Given an example in dataset format, create the prompt and the list of
        correct references.
        """
        symbol = "____"
        prompt = "請閱讀以下短文，並在空格內填入正確的成語：\n\n"
        content = sample['examples'].replace("____", symbol)
        prompt += f"內容：{content}"
        prompt += f"空格中應填入成語\n"
        prompt += "1."
        

        answers = sample["ans"].split(',')
        

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
        splits = {"valid": VALID_SPLIT}
        random.seed(0)  # we pick a random dialogue point to query the model
        for split, split_tag in splits.items():
            split_path: str = os.path.join(data_path, f"{split}.json")
            instances.extend(self.get_split_instances(split_path, split=split_tag))

        return instances
        
class TCIC_MC_Scenario(Scenario):
    """
    TBD
    """

    name = "tcic"
    description = "Idiom cloze answering."
    tags = ["question_answering"]

    def create_prompt(self, sample: dict) -> Tuple[str, List[str]]:
        """
        Given an example in dataset format, create the prompt and the list of
        correct references.
        """
        symbol = "____"
        #prompt = "請閱讀以下短文，並在空格內填入正確的成語：\n\n"
        content = sample['passage'].replace("____", symbol)
        prompt = f"{content}"
        prompt += f"\n\n空格中應填入成語："
        #prompt += 'A. {}\nB. {}\nC. {}\nD. {}\nAnswer:'.format(sample['A'],sample['B'],sample['C'],sample['D'])
        answers = [ c+'. '+sample[c] for c in ['A','B','C','D']]
        #answers = [sample['A'], sample['B'], sample['C'], sample['D']]
        correct_answer = answers[ord(sample["mc_answer"])-65]
        print(correct_answer)

        return prompt, answers, correct_answer

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
            samples = json.load(f)
            
        def answer_to_reference(answer: str) -> Reference:
            return Reference(Output(text=answer), tags=[CORRECT_TAG] if answer == correct_answer else [])
            

        for sample in samples.items():
            sample = sample[1] #???
            question, answers, correct_answer = self.create_prompt(sample)
            instance = Instance(
                input=Input(text=question),
                references=list(map(answer_to_reference, answers)),
                split=split,
            )
            file_instances.append(instance)
        return file_instances


    def get_instances(self) -> List[Instance]:
        data_path: str = os.path.join("restricted", self.name)
        assert os.path.exists(data_path)

        instances: List[Instance] = []
        
        
        splits = {"valid": VALID_SPLIT}
        random.seed(0)  # we pick a random dialogue point to query the model
        for split, split_tag in splits.items():
            split_path: str = os.path.join(data_path, f"TCIC_mc_1.2.0.json")
            instances.extend(self.get_split_instances(split_path, split=split_tag))

        return instances
        
    

