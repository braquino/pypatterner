from typing import Tuple, Iterable, List
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from py_pattern.model_train.preprocessing_utils import apply_tokenizer

train_on_gpu = torch.cuda.is_available()


class GruBiDirNN(nn.Module):

    def __init__(self, input_padding, embedding_dimension, hidden_size, output_padding, tokenizer, n_layers=1,
                 learning_rate=0.015, batch_size=5000):
        super().__init__()
        self.input_padding = input_padding
        self.embedding_dimension = embedding_dimension
        self.output_padding = output_padding
        self.n_layers = n_layers
        self.vocab_size = tokenizer.get_len
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.encoder = nn.Embedding(self.vocab_size, self.embedding_dimension)
        self.gru = nn.GRU(self.embedding_dimension, self.hidden_size, self.n_layers,
                          batch_first=True, bidirectional=True)
        self.decoder = nn.Linear(self.hidden_size * self.input_padding * self.n_layers * 2,
                                 self.output_padding * self.vocab_size)
        self.losses = []
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, input_data: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        embedded = self.encoder(input_data)
        raw_output, _ = self.gru(embedded)
        output = self.decoder(raw_output.reshape(self.batch_size, -1))  # TODO: test reverse embedded here
        return output.view((self.batch_size, self.output_padding, self.vocab_size))

    def train_all_data(self, input_data: np.ndarray, target: np.ndarray):
        """
        :param input_data: Numpy Array 2D type int (date tokenized size x padding)
        :param target: Numpy Array 2D type int (pattern tokenized size x padding)
        """
        losses = []

        num_batches = int(len(input_data) / self.batch_size)
        for i in range(num_batches):  # FIXME: last batch
            self.zero_grad()

            input_batch: np.ndarray = input_data[i * self.batch_size: (i + 1) * self.batch_size]
            target_batch_cat: np.ndarray = target[i * self.batch_size: (i + 1) * self.batch_size]
            target_batch = np.zeros((self.batch_size, self.output_padding, self.vocab_size))
            for b in range(self.batch_size):
                for p in range(self.output_padding):
                    target_batch[b, p, target_batch_cat[b, p]] = 1
            input_tensor = torch.tensor(input_batch, dtype=torch.long)#.cuda()
            target_tensor = torch.tensor(target_batch, dtype=torch.float)#.cuda()
            output = self.forward(input_tensor)
            loss = self.criterion(output, target_tensor)
            loss.backward()
            self.optimizer.step()
            losses += [float(loss)]
            print("batch {}: loss: {}".format(i, losses[-1]))
        print("Epoch loss: " + str(np.mean(losses)))
        self.losses.append(np.mean(losses))

    def plot_loss_chart(self):
        plt.plot(self.losses)

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()

    def predict(self, input_date: Iterable[str]) -> List[str]:
        with torch.no_grad():
            tokenized = apply_tokenizer(input_date, self.input_padding, self.tokenizer)
            input_tensor = torch.tensor(tokenized, dtype=torch.long)
            output_tensor: torch.Tensor = self.forward(input_tensor)
            out_array: np.ndarray = output_tensor.numpy()
            out_cat = out_array.argmax(axis=2)
            return ["".join([self.tokenizer.idx_to_char[idx] for idx in out]).strip() for out in out_cat]






