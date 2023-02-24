import json
import os
import random
from typing import Dict, List, Tuple

from helm.common.general import ensure_file_downloaded, ensure_directory_exists
from .scenario import Scenario, Instance, Reference, TRAIN_SPLIT, VALID_SPLIT, TEST_SPLIT, CORRECT_TAG, Input, Output


class TTQAScenario(Scenario):
    """
    TBD
    """

    name = "ttqa"
    description = "Taiwan trivia question answering"
    tags = ["cloze"]

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
        with open(target_file, encoding="utf-8") as f:
            samples = json.load(f)

        for sample in samples:
            prompt, answers = sample['examples'], sample['answer']
            answers = answers.split(',')
            instance = Instance(
                input=Input(text=prompt),
                references=[Reference(Output(text=answer), tags=[CORRECT_TAG]) for answer in answers],
                split=TEST_SPLIT,
            )
            #print(instance)
            file_instances.append(instance)
        return file_instances


    def get_instances(self) -> List[Instance]:
        file_path: str = os.path.join("restricted", self.name, "TTQA_1.0.0.json")
        assert os.path.exists(file_path)
        random.seed(0)  # randomness needed to pick question at random
        
        instances: List[Instance] = []
        instances: List[Instance] = self.get_file_instances(target_file=file_path)
        return instances
       
class TTQA_MC_Scenario(Scenario):

    name = "ttqa"
    description = "Taiwan trivia question answering"
    tags = ["cloze"]
    
    #def permutation():
    def create_prompt(self, sample: dict) -> Tuple[str, str]:
        template = '{}\n{}\nA. {}\nB. {}\nC. {}\nD. {}\nAnswer:'
        prompt   = template.format(sample['passage'],sample['question'], \
        			    sample['A'],sample['B'],sample['C'],sample['D'])
        return prompt, sample['answer']

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
        with open(target_file, encoding="utf-8") as f:
            samples = json.load(f)
        
        for i,sample in samples.items():
            #for p in permutation():
            prompt, answer = self.create_prompt(sample)
            #prompt, answer = sample['prompt'], sample['mc_answer']
            instance = Instance(
                input=Input(text=prompt),
                references=Reference(Output(text=answer), tags=[CORRECT_TAG]),
                split=TEST_SPLIT,
            )
            file_instances.append(instance)
            #print(instance)
        return file_instances


    def get_instances(self) -> List[Instance]:
        file_path: str = os.path.join("restricted", self.name, "TTQA_mc_1.0.0.json")
        assert os.path.exists(file_path)
        random.seed(0)  # randomness needed to pick question at random
        
        instances: List[Instance] = []
        instances: List[Instance] = self.get_file_instances(target_file=file_path)
        return instances

class TTQA_5_Shot_Scenario(Scenario):
    """
    TBD
    """

    name = "ttqa"
    description = "Taiwan trivia question answering"
    tags = ["cloze"]

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
        with open(target_file, encoding="utf-8") as f:
            samples = json.load(f)

        with open(os.path.join(os.path.dirname(target_file), "5_shot_examples.txt"), encoding="utf-8") as f:
            few_shot_example = ''.join(f.readlines())

        for sample in samples:
            prompt, answers = sample['examples'], sample['answer']
            prompt = few_shot_example + '\n\n' + few_shot_example
            answers = answers.split(',')
            instance = Instance(
                input=Input(text=prompt),
                references=[Reference(Output(text=answer), tags=[CORRECT_TAG]) for answer in answers],
                split=TEST_SPLIT,
            )
            file_instances.append(instance)
        return file_instances


    def get_instances(self) -> List[Instance]:
        file_path: str = os.path.join("restricted", self.name, "TTQA_1.0.0.json")
        assert os.path.exists(file_path)
        random.seed(0)  # randomness needed to pick question at random
        
        instances: List[Instance] = []
        instances: List[Instance] = self.get_file_instances(target_file=file_path)
        return instances
