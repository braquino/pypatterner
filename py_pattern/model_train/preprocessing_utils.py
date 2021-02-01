import string
import json
from dataclasses import dataclass
from typing import Dict, List, Iterable
import pandas as pd
import numpy as np


def tokenize_chars_json(json_path: str):
    all_chars = string.printable
    char_to_idx = {char: idx for idx, char in enumerate(all_chars)}
    with open(json_path, mode="w") as f:
        json.dump(char_to_idx, f)

@dataclass
class Tokenizer:
    char_to_idx: Dict[str, int]
    idx_to_char: Dict[int, str]

    @staticmethod
    def from_json(json_path: str):
        with open(json_path) as f:
            char_to_idx = json.load(f)
            idx_to_char = {char_to_idx[k]: k for k in char_to_idx}
            return Tokenizer(char_to_idx, idx_to_char)

    @property
    def get_len(self):
        return len(self.char_to_idx)

def apply_tokenizer(data: Iterable[str], padding: int, tokenizer: Tokenizer) -> List[np.ndarray]:
    def tokenize_row(row) -> np.ndarray:
        padded_row = np.full((padding), 94)
        for i in range(min(padding, len(row))):
            padded_row[i] = tokenizer.char_to_idx[row[i]]
        return padded_row
    return [tokenize_row(row) for row in data]


def tokenize_dataset(df: pd.DataFrame, columns_padding: Dict[str, int], tokenizer: Tokenizer) -> pd.DataFrame:
    for column in columns_padding:
        df[column + "_tokenized"] = apply_tokenizer(df[column].values, columns_padding[column], tokenizer)
    return df