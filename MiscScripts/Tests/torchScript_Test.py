import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import random


batch_size = 128
hidden_size = 256
input_features = 22
max_n_tracks = 12
n_layers = 3


class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_features,
            hidden_size=hidden_size,
            batch_first=False,
            num_layers=n_layers,
            dropout=0.2,
            bidirectional=False,
        )

    @torch.jit.script_method
    def forward(self, first_input, second_input):
        first_output = self.rnn(first_input)
        second_output = self.rnn(second_input)
        return first_output + second_output


rnn = RNN()
# padded version - following https://pytorch.org/blog/optimizing-cuda-rnn-with-torchscript/
track_info = []
for i in range(batch_size):
    this_track_length = random.randint(1, max_n_tracks)
    track_info.append(torch.randn(this_track_length, input_features))
padded_track_seq = pad_sequence(track_info)
track_length = [track.size(0) for track in track_info]

second_input = nn.Parameter(torch.zeros(max_n_tracks, batch_size, input_features))

traced_cell = torch.jit.trace(rnn, (padded_track_seq, second_input))
print(traced_cell)
traced_cell(padded_track_seq, second_input)
