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
        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_seq):
        embeds = self.embeddings(input_seq)  # (batch, seq_len, embedding_dim)
        lstm_out, _ = self.lstm(embeds)
        final_hidden_state = lstm_out[:, -1]
        output = self.fc(final_hidden_state)
        return output


class RNN_Sent140(nn.Module):

    def __init__(self, vocab_size=400000, embedding_dim=100, hidden_size=256, output_dim=2):
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
        super(RNN_Sent140, self).__init__()
        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, input_seq):
        embeds = self.embeddings(input_seq)  # (batch, seq_len, embedding_dim)
        lstm_out, hidden = self.lstm(embeds)  # lstm_out = [seq_len, batch, hidden_dim]
        # final_hidden_state = lstm_out[:, -1]
        assert torch.equal(lstm_out[-1, :, :], hidden.squeeze(0))
        assert torch.equal(lstm_out[:, -1], hidden.squeeze(0))
        return self.fc(hidden.squeeze(0))
        # output = self.fc(final_hidden_state)
        # return output
