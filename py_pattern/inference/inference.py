from dataclasses import dataclass
from string import Template
from typing import List
from collections import Counter
from py_pattern.model_train.preprocessing_utils import Tokenizer
from py_pattern.model_train.models import GruBiDirNN
from py_pattern.model_train.train import DIMENSIONALITY, INPUT_PADDING, OUTPUT_PADDING, HIDDEN_SIZE
from datetime import datetime
import re


@dataclass
class PatternAttributes:
    python_pattern: str
    java_pattern: str
    validation: float
    number_of_predictions: int


class InferenceModel:

    def __init__(self, weight_path: str, batch_size):
        tokenizer = Tokenizer.from_json("../../train_data/char_to_idx.json")
        self.model = GruBiDirNN(INPUT_PADDING, DIMENSIONALITY, HIDDEN_SIZE, OUTPUT_PADDING, tokenizer=tokenizer,
                                batch_size=batch_size)
        self.model.load_weights(weight_path)

    def predict_formats(self, str_dates_list: List[str]):
        return self.model.predict(str_dates_list)

    def predict_best_format(self, str_dates_list: List[str]):
        pattern_list: List[str] = self.model.predict(str_dates_list)
        pattern_distrib = Counter(pattern_list)
        best_patterns = [x[0] for x in pattern_distrib.most_common()]
        validation = self._validate(str_dates_list, best_patterns)
        return sorted((PatternAttributes(pattern,
                                         self._python_to_java_simpledate(pattern),
                                         validation[pattern],
                                         pattern_distrib[pattern]) for pattern in best_patterns),
                      key=lambda x: x.validation, reverse=True)

    @staticmethod
    def _validate(str_dates_list: List[str], pattern_list: List[str]):
        validation = {pattern: 0 for pattern in pattern_list}
        n_elem = len(str_dates_list)
        for pattern in pattern_list:
            for date in str_dates_list:
                try:
                    datetime.strptime(date, pattern)
                    validation[pattern] += 1/n_elem
                except (re.error, ValueError):
                    pass
        return validation

    @staticmethod
    def _python_to_java_simpledate(python_pattern: str):
        mapping = {'Y': 'yyyy', 'm': 'MM', 'd': 'dd', 'H': 'HH', 'M': 'mm', 'S': 'ss', 'y': 'yy',
                   'b': 'MMM', '#m': 'M', '#d': 'd', '#H': 'H', '#M': 'm', '#S': 's'}
        try:
            return Template(python_pattern.replace('%', '$')).substitute(**mapping)
        except:
            return "error"


if __name__ == "__main__":
    input_dates = ["13-04-1984 12:00", "04/13/84", "01/21/2016 05:50", "01/01/2016 04:50",
                   "01/01/2016 06:50", "01/01/2016 07:50", "01/13/2016 07:50", '01/Aug/1995:00:00:01',
                   '01/Aug/1995:00:00:07', '01/Aug/1995:00:00:08', '01/Aug/1995:00:00:08', '01/Aug/1995:00:00:08',
                   '01/Aug/1995:00:00:09']
    weight_path = "../model_train/weights/model_003.pth"
    inference_model = InferenceModel(weight_path, len(input_dates))
    print(inference_model.predict_best_format(input_dates))