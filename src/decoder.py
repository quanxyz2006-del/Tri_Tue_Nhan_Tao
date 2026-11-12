from torch import nn
from utils.constants import PAD


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=PAD)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, tokens, state):
        outputs, state = self.lstm(self.embedding(tokens), state)
        return self.output(outputs), state
