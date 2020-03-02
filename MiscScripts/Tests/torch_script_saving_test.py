# import torch

# import torch.nn as nn

"""
saving test
"""


# class Model(torch.jit.ScriptModule):
#     """dummy model """

#     def __init__(self):
#         super(Model, self).__init__()
#         # self.device = torch.device("cuda")
#         self.num_heads = 1
#         self.hidden_size = 12
#         self.n_trk_features = 6
#         self.n_calo_features = 6
#         self.n_lep_features = 10
#         self.output_size = 2
#         # self.trk_SetTransformer = SetTransformer(
#         #     self.n_trk_features, num_outputs=self.num_heads, dim_output=self.hidden_size
#         # ).to(self.device)
#         # self.calo_SetTransformer = SetTransformer(
#         #     self.n_calo_features, self.num_heads, self.hidden_size
#         # ).to(self.device)
#         # self.output_layer = nn.Linear(self.hidden_size * 2, self.hidden_size).to(
#         #     self.device
#         # )
#         # self.fc_final = nn.Linear(
#         #     self.hidden_size + self.n_lep_features, self.output_size
#         # ).to(self.device)
#         # self.relu_final = nn.ReLU(inplace=True)
#         # self.dropout = nn.Dropout(p=0.3)
#         # self.softmax = nn.Softmax(dim=1)

#     @torch.jit.script_method
#     def forward(self):
#         a = torch.rand([6, 1, 12])
#         b = torch.rand([6, 1, 12])
#         out = torch.cat([a, b], dim=2)
#         return out

#     def save_to_pytorch(self, output_path):
#         torch.jit.save(self, output_path)


# if __name__ == "__main__":
#     model = Model()
#     print(model())  # works
#     script = torch.jit.script(model)
#     model.to(torch.device("cuda"))
#     model.save_to_pytorch("test.zip")
#     loaded = torch.jit.load("test.zip")
#     print(loaded.code)

import torch
from torch.nn.utils.rnn import PackedSequence
import torch.nn._VF as torch_varfuncs
from torch._jit_internal import Optional


class MyModule(torch.nn.Module):
    def __init__(self, N, M):
        super(MyModule, self).__init__()
        # This parameter will be copied to the new ScriptModule
        self.weight = torch.nn.Parameter(torch.rand(N, M))

        # When this submodule is used, it will be compiled
        self.linear = torch.nn.Linear(N, M)

    def _pad(
        self,
        data,
        batch_first: bool,
        batch_sizes,
        pad_value: float,
        sorted_indices: Optional[torch.Tensor],
        unsorted_indices: Optional[torch.Tensor],
    ):
        packed_seq = PackedSequence(data, batch_sizes, sorted_indices, unsorted_indices)
        return torch.nn.utils.rnn.pad_packed_sequence(
            packed_seq, batch_first, pad_value
        )

    def forward(self, input, data_lengths):
        batch_first = True
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(
            input, batch_first=batch_first, lengths=data_lengths, enforce_sorted=False
        )
        output = self.weight.mv(packed_input.data)
        # This calls the `forward` method of the `nn.Linear` module, which will
        # cause the `self.linear` submodule to be compiled to a `ScriptModule` here
        output = self.linear(output)
        return self._pad(
            output,
            batch_first,
            packed_input.batch_sizes,
            -1.0,
            packed_input.sorted_indices,
            packed_input.unsorted_indices,
        )


class MyModuleVF(MyModule):
    def _pad(
        self,
        data,
        batch_first1: bool,
        batch_sizes,
        pad_value: float,
        sorted_indices: Optional[torch.Tensor],
        unsorted_indices: Optional[torch.Tensor],
    ):

        max_length = batch_sizes.size(0)
        padded_output, lengths = torch_varfuncs._pad_packed_sequence(
            data, batch_sizes, batch_first1, -1.0, max_length
        )
        if sorted_indices is not None:
            # had to invert permute specifically as pytorch method was giving errors in jit (arange is returning float type and not long, as expected)
            output = torch.empty_like(sorted_indices)
            output.scatter_(
                0,
                sorted_indices,
                torch.arange(
                    0, sorted_indices.numel(), device=sorted_indices.device
                ).long(),
            )
            batch_dim = 0 if batch_first1 else 1
            return padded_output.index_select(batch_dim, output), lengths[output]
        return padded_output, lengths


test_input = torch.tensor(
    [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, -1.0, -1.0], [8, 9, 10, -1.0]], dtype=torch.float
)
data_lengths = torch.tensor([4, 2, 3])
size_ = (test_input > 0).sum()
# works
mm = MyModule(20, size_.item())
result = mm(test_input, data_lengths)


# works
mmvf = MyModuleVF(20, size_.item())
result_vf = mmvf(test_input, data_lengths)


# works
mmvf_s = torch.jit.script(MyModuleVF(20, size_.item()))
result_vf_s = mmvf_s(test_input, data_lengths)

# does not work
mm_s = torch.jit.script(MyModule(20, size_.item()))
result_s = mm(test_input, data_lengths)
