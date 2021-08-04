"""RNN model in pytorch
References:
    [1] H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, Blaise Agueray Arcas.
    Communication-Efficient Learning of Deep Networks from Decentralized Data. AISTATS 2017.
    https://arxiv.org/abs/1602.05629
    [2] Reddi S, Charles Z, Zaheer M, et al.
    Adaptive Federated Optimization. ICML 2020.
    https://arxiv.org/pdf/2003.00295.pdf
"""
import torch.nn as nn
import torch
from fedlab_benchmarks.datasets.leaf_data_process.nlp_utils.vocab import Vocab


class RNN_Shakespeare(nn.Module):
    def __init__(self, vocab_size=90, embedding_dim=8, hidden_size=256):
        """Creates a RNN model using LSTM layers for Shakespeare language models (next character prediction task).

        Args:
            vocab_size (int, optional): the size of the vocabulary, used as a dimension in the input embedding,
                Defaults to 90.
            embedding_dim (int, optional): the size of embedding vector size, used as a dimension in the output embedding,
                Defaults to 8.
            hidden_size (int, optional): the size of hidden layer. Defaults to 256.

        Returns:
            A `torch.nn.Module`.
        """
        super(RNN_Shakespeare, self).__init__()
        self.embeddings = nn.Embedding(num_embeddings=vocab_size,
                                       embedding_dim=embedding_dim,
                                       padding_idx=0)
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_size,
                            num_layers=2,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_seq):
        embeds = self.embeddings(input_seq)  # (batch, seq_len, embedding_dim)
        lstm_out, _ = self.lstm(embeds)
        final_hidden_state = lstm_out[:, -1]
        output = self.fc(final_hidden_state)
        return output


class LSTMModel(nn.Module):
    def __init__(self,
                 vocab_size=90, embedding_dim=8, hidden_size=256, num_layers=2, output_dim=3,
                 using_pretrained=False, embedding_weights=None, bid=False, head_tail=False):
        """Creates a RNN model using LSTM layers for Shakespeare language models (next character prediction task).

        Args:
            vocab_size (int, optional): the size of the vocabulary, used as a dimension in the input embedding,
            embedding_dim (int, optional): the size of embedding vector size, used as a dimension in the output embedding
            hidden_size (int, optional): the size of hidden layer. Defaults to 256.

        Returns:
            A `torch.nn.Module`.
        """
        super(LSTMModel, self).__init__()
        self.embeddings = nn.Embedding(num_embeddings=vocab_size,
                                       embedding_dim=embedding_dim,
                                       padding_idx=0)
        if using_pretrained:
            assert embedding_weights.shape[0] == vocab_size
            assert embedding_weights.shape[1] == embedding_dim
            self.embedding.from_pretrained(embedding_weights)

        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bid,
            dropout=0.3,
        )

        # using bidrectional, *2
        if bid:
            hidden_size *= 2
        if head_tail:
            hidden_size *= 2
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_size, output_dim),
        )

    def forward(self, input_seq):
        embeds = self.embeddings(input_seq)  # (batch, seq_len, embedding_dim)
        embeds = self.dropout(embeds)
        outputs, _ = self.lstm(embeds)  # output, (hidden, memory)
        return self.fc(outputs[-1, :, :])
