import torch

# import torch.nn as nn

"""
saving test
"""


class Model(torch.jit.ScriptModule):
    """dummy model """

    def __init__(self):
        super(Model, self).__init__()
        # self.device = torch.device("cuda")
        self.num_heads = 1
        self.hidden_size = 12
        self.n_trk_features = 6
        self.n_calo_features = 6
        self.n_lep_features = 10
        self.output_size = 2
        # self.trk_SetTransformer = SetTransformer(
        #     self.n_trk_features, num_outputs=self.num_heads, dim_output=self.hidden_size
        # ).to(self.device)
        # self.calo_SetTransformer = SetTransformer(
        #     self.n_calo_features, self.num_heads, self.hidden_size
        # ).to(self.device)
        # self.output_layer = nn.Linear(self.hidden_size * 2, self.hidden_size).to(
        #     self.device
        # )
        # self.fc_final = nn.Linear(
        #     self.hidden_size + self.n_lep_features, self.output_size
        # ).to(self.device)
        # self.relu_final = nn.ReLU(inplace=True)
        # self.dropout = nn.Dropout(p=0.3)
        # self.softmax = nn.Softmax(dim=1)

    @torch.jit.script_method
    def forward(self):
        a = torch.rand([6, 1, 12])
        b = torch.rand([6, 1, 12])
        out = torch.cat([a, b], dim=2)
        return out

    def save_to_pytorch(self, output_path):
        torch.jit.save(self, output_path)


if __name__ == "__main__":
    model = Model()
    print(model())  # works
    script = torch.jit.script(model)  # throws error
    model.save_to_pytorch("test.zip")
    loaded = torch.jit.load("test.zip")
    print(loaded.code)
