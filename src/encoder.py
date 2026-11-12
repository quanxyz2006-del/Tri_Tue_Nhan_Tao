import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils.constants import PAD


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, bidirectional=True):
        super().__init__()
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=PAD)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True,
                            bidirectional=bidirectional)
        if bidirectional:
            self.project_h = nn.Linear(2 * hidden_size, hidden_size)
            self.project_c = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, source):
        embedded = self.embedding(source)
        lengths = source.ne(PAD).sum(1).cpu()
        packed = pack_padded_sequence(embedded, lengths, batch_first=True,
                                      enforce_sorted=False)
        outputs, (h, c) = self.lstm(packed)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True,
                                         total_length=source.size(1))
        if self.bidirectional:
            h = torch.tanh(self.project_h(torch.cat((h[0], h[1]), -1))).unsqueeze(0)
            c = torch.tanh(self.project_c(torch.cat((c[0], c[1]), -1))).unsqueeze(0)
        return outputs, (h, c)
