from py_pattern.model_train.preprocessing_utils import tokenize_dataset, Tokenizer
from py_pattern.model_train.models import GruBiDirNN
import pandas as pd
import numpy as np
import os

from py_pattern.train_creator.train_creator_main import create_train_file

DIMENSIONALITY = 20
INPUT_PADDING = 40
OUTPUT_PADDING = 30
HIDDEN_SIZE = 10
EPOCHS = 20

if __name__ == "__main__":
    tokenizer = Tokenizer.from_json("../../train_data/char_to_idx.json")
    model = GruBiDirNN(INPUT_PADDING, DIMENSIONALITY, HIDDEN_SIZE, OUTPUT_PADDING, tokenizer=tokenizer)
    model.load_weights("weights/model_003.pth")
    for i in range(EPOCHS):
        file_path = create_train_file()
        df = pd.read_csv(file_path)
        df = tokenize_dataset(df, {"date": INPUT_PADDING, "pattern": OUTPUT_PADDING}, tokenizer)
        df = df.sample(frac=1).reset_index(drop=True)
        model.train_all_data(np.stack(df["date_tokenized"].values, axis=0),
                             np.stack(df["pattern_tokenized"].values, axis=0))
        model.save_weights("weights/model_003.pth")
        os.remove(file_path)
    model.plot_loss_chart()

