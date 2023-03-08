import json
import os
import random
from typing import Dict, List, Tuple

from helm.common.general import ensure_file_downloaded, ensure_directory_exists
from .scenario import Scenario, Instance, Reference, TRAIN_SPLIT, VALID_SPLIT, TEST_SPLIT, CORRECT_TAG, Input, Output


class SLTPScenario(Scenario):
    """
    TBD
    """

    name = "sltp"
    description = "Taiwan-specific knowledge"
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
            file_instances.append(instance)
            print(file_instances)
        return file_instances


    def get_instances(self) -> List[Instance]:
        file_path: str = os.path.join("restricted", self.name, "SLTP_3.0.json")
        assert os.path.exists(file_path)
        random.seed(0)  # randomness needed to pick question at random
        
        instances: List[Instance] = []
        instances: List[Instance] = self.get_file_instances(target_file=file_path)
        return instances

