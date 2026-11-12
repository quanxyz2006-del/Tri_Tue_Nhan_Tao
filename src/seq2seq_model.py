import torch
from torch import nn
from src.encoder import Encoder
from src.decoder import Decoder
from utils.constants import EOS, PAD, SOS


class Seq2Seq(nn.Module):
    def __init__(self, vocab_en, vocab_vi, embedding_size=128, hidden_size=256,
                 bidirectional=True):
        super().__init__()
        self.encoder = Encoder(vocab_en, embedding_size, hidden_size, bidirectional)
        self.decoder = Decoder(vocab_vi, embedding_size, hidden_size)

    def forward(self, source, decoder_input):
        _, state = self.encoder(source)
        logits, _ = self.decoder(decoder_input, state)
        return logits

    @torch.no_grad()
    def generate(self, source, max_length):
        self.eval()
        outputs, state = self.encoder(source)
        shapes = {"embedding": [*source.shape, self.encoder.embedding.embedding_dim],
                  "encoder": list(outputs.shape), "h_init": list(state[0].shape),
                  "c_init": list(state[1].shape)}
        token = torch.tensor([[SOS]], device=source.device)
        ids, steps = [], []
        for step in range(max_length):
            logits, state = self.decoder(token, state)
            probabilities = logits[0, -1].softmax(-1)
            scores = logits[0, -1].clone()
            scores[[PAD, SOS]] = -torch.inf
            index = int(scores.argmax())
            steps.append({"step": step + 1, "id": index,
                          "probability": float(probabilities[index])})
            if index == EOS:
                break
            ids.append(index)
            token = torch.tensor([[index]], device=source.device)
        return ids, shapes, steps
