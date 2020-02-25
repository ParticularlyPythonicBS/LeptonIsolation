import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F


class GRU_Model(nn.Module):
    def __init__(self, options):
        super().__init__(options)
        self.trk_rnn = nn.GRU(
            input_size=self.n_trk_features,
            hidden_size=self.hidden_size,
            batch_first=True,
            num_layers=self.n_layers,
            dropout=self.rnn_dropout,
            bidirectional=False,
        ).to(self.device)
        self.cal_rnn = nn.GRU(
            input_size=self.n_calo_features,
            hidden_size=self.hidden_size,
            batch_first=True,
            num_layers=self.n_layers,
            dropout=self.rnn_dropout,
            bidirectional=False,
        ).to(self.device)

    def forward(self, batch):
        return self.recurrent_forward(batch)


class Model(nn.Module):
    """
    GRU model class
    """

    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda")
        self.batch_size = 6
        self.hidden_size = 12
        self.n_trk_features = 6
        self.n_calo_features = 6
        self.n_lep_features = 10
        self.output_size = 2
        self.n_layers = 3
        self.rnn_dropout = 0.3
        self.h_0 = nn.Parameter(
            torch.zeros(self.n_layers, self.batch_size, self.hidden_size).to(
                self.device
            )
        )
        self.trk_rnn = nn.GRU(
            input_size=self.n_trk_features,
            hidden_size=self.hidden_size,
            batch_first=True,
            num_layers=self.n_layers,
            dropout=self.rnn_dropout,
            bidirectional=False,
        ).to(self.device)
        self.cal_rnn = nn.GRU(
            input_size=self.n_calo_features,
            hidden_size=self.hidden_size,
            batch_first=True,
            num_layers=self.n_layers,
            dropout=self.rnn_dropout,
            bidirectional=False,
        ).to(self.device)
        self.fc_pooled = nn.Linear(self.hidden_size * 3, self.hidden_size).to(
            self.device
        )
        self.output_layer = nn.Linear(self.hidden_size * 2, self.hidden_size).to(
            self.device
        )
        self.fc_trk_cal = nn.Linear(self.hidden_size * 2, self.hidden_size).to(
            self.device
        )

        self.fc_final = nn.Linear(
            self.hidden_size + self.n_lep_features, self.output_size
        ).to(self.device)
        self.relu_final = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.3)
        self.softmax = nn.Softmax(dim=1)
        self.loss_function = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def mock_prep_for_forward(self):
        """
        Preps dummy data for passing through the net

        Returns:
            prepared mock data
        """
        import dummy_data

        track_info = dummy_data.dummy_batch["track_info"]
        track_length = dummy_data.dummy_batch["track_length"]
        lepton_info = dummy_data.dummy_batch["lepton_info"]
        calo_info = dummy_data.dummy_batch["calo_info"]
        calo_length = dummy_data.dummy_batch["calo_length"]

        # sort and pack padded sequences for tracks and calo clusters
        sorted_n_tracks, sorted_indices_tracks = torch.sort(
            track_length, descending=True
        )
        sorted_tracks = track_info[sorted_indices_tracks].to(self.device)
        sorted_n_tracks = sorted_n_tracks.detach().cpu()
        sorted_n_cal, sorted_indices_cal = torch.sort(calo_length, descending=True)
        sorted_cal = calo_info[sorted_indices_cal].to(self.device)
        sorted_n_cal = sorted_n_cal.detach().cpu()

        torch.set_default_tensor_type(torch.FloatTensor)
        padded_track_seq = pack_padded_sequence(
            sorted_tracks, sorted_n_tracks, batch_first=True, enforce_sorted=True
        )
        padded_cal_seq = pack_padded_sequence(
            sorted_cal, sorted_n_cal, batch_first=True, enforce_sorted=True
        )
        if self.device == torch.device("cuda"):
            torch.set_default_tensor_type(torch.cuda.FloatTensor)

        padded_track_seq.to(self.device)
        padded_cal_seq.to(self.device)
        prepped_batch = padded_track_seq, padded_cal_seq, lepton_info.to(self.device)

        return prepped_batch

    # @torch.jit.script_method
    def forward(self, padded_track_seq, padded_cal_seq, lepton_info):

        self.trk_rnn.flatten_parameters()
        self.cal_rnn.flatten_parameters()

        output_track, hidden_track = self.trk_rnn(padded_track_seq, self.h_0)
        output_cal, hidden_cal = self.cal_rnn(padded_cal_seq, self.h_0)

        output_track, lengths_track = pad_packed_sequence(
            output_track, batch_first=False
        )
        output_cal, lengths_cal = pad_packed_sequence(output_cal, batch_first=False)

        out_cal = self.concat_pooling(output_cal, hidden_cal)
        out_tracks = self.concat_pooling(output_track, hidden_track)

        # combining rnn outputs
        out = self.fc_trk_cal(torch.cat([out_cal, out_tracks], dim=1))
        F.relu_(out)
        out = self.dropout(out)
        out = self.fc_final(torch.cat([out, lepton_info], dim=1))
        out = self.relu_final(out)
        out = self.softmax(out)

        return out

    def concat_pooling(self, output_rnn, hidden_rnn):
        """
        Pools and contatenates rnn output to suggest permutation invariance
        Concat pooling idea from: https://arxiv.org/pdf/1801.06146.pdf
        Args:
            pad_packed_sequence output and final hidden layer
        Returns: processed output
        """
        output_rnn = output_rnn.permute(
            1, 2, 0
        )  # converted to BxHxW, W=#words B=batch_size H=#neurons_hidden_layer
        # hidden_rnn already in form LxBxH, L=#layers
        avg_pool_rnn = F.adaptive_avg_pool1d(output_rnn, 1).view(-1, self.hidden_size)
        max_pool_rnn = F.adaptive_max_pool1d(output_rnn, 1).view(-1, self.hidden_size)
        concat_output = torch.cat([hidden_rnn[-1], avg_pool_rnn, max_pool_rnn], dim=1)
        out_rnns = self.fc_pooled(concat_output)
        return out_rnns


if __name__ == "__main__":
    # Testing
    model = Model()
    print(model(*model.mock_prep_for_forward()))

# TODO: code currently works for native execution, torch script requires debugging
