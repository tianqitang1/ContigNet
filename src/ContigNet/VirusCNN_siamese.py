import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import itertools


class VirusCNN(nn.Module):
    def __init__(self, channel="both", rev_comp=False, share_weight=False) -> None:
        super().__init__()

        self.channel = channel
        self.rev_comp = rev_comp
        self.share_weight = share_weight

        if self.channel == "both":
            fc_layer_input_dim = 1024
        else:
            fc_layer_input_dim = 512
        self.fc = nn.Sequential(
            nn.Linear(fc_layer_input_dim, 128), nn.ReLU(inplace=True), nn.Dropout(p=0.5), nn.Linear(128, 1)
        )

        self.codon_channel_num = 64

        self.base_channel1 = nn.Sequential(
            nn.Conv2d(1, 64, (6, 4)),
            nn.ReLU(inplace=True),
            nn.Flatten(start_dim=2),
            nn.MaxPool1d(3),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 256, 3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        if self.share_weight:
            self.base_channel2 = self.base_channel1
        else:
            self.base_channel2 = nn.Sequential(
                nn.Conv2d(1, 64, (6, 4)),
                nn.ReLU(inplace=True),
                nn.Flatten(start_dim=2),
                nn.MaxPool1d(3),
                nn.BatchNorm1d(64),
                nn.Conv1d(64, 128, 3),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(3),
                nn.BatchNorm1d(128),
                nn.Conv1d(128, 256, 3),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
            )

        self.codon_channel1 = nn.Sequential(
            nn.Conv2d(1, 64, (6, self.codon_channel_num)),
            nn.ReLU(inplace=True),
            nn.Flatten(start_dim=2),
            nn.MaxPool1d(3),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 256, 3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        if self.share_weight:
            self.codon_channel2 = self.codon_channel1
        else:
            self.codon_channel2 = nn.Sequential(
                nn.Conv2d(1, 64, (6, self.codon_channel_num)),
                nn.ReLU(inplace=True),
                nn.Flatten(start_dim=2),
                nn.MaxPool1d(3),
                nn.BatchNorm1d(64),
                nn.Conv1d(64, 128, 3),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(3),
                nn.BatchNorm1d(128),
                nn.Conv1d(128, 256, 3),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
            )

        self.dropout = nn.Dropout(p=0.5, inplace=True)

        self.codon_transformer = CodonTransformer()

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten())

        self.global_max_pool = nn.Sequential(nn.AdaptiveMaxPool1d(1), nn.Flatten())

    @staticmethod
    def build_rev_comp(x: torch.Tensor):
        return torch.flip(x, [-1, -2])

    def forward(self, x, y):
        if self.rev_comp:
            x_rev_comp = self.build_rev_comp(x)
            x = torch.cat((x, x_rev_comp), dim=-2)
            y_rev_comp = self.build_rev_comp(y)
            y = torch.cat((y, y_rev_comp), dim=-2)
        if self.channel == "both":
            x1 = self.base_channel1(x)
            y1 = self.base_channel2(y)

            x = self.codon_transformer(x)
            y = self.codon_transformer(y)

            x = self.codon_channel1(x)
            y = self.codon_channel2(y)

            z = torch.cat((x, x1, y, y1), dim=1)
            # z = self.dropout(z)
            z = self.fc(z)
        elif self.channel == "base":
            x = self.base_channel1(x)
            y = self.base_channel2(y)

            z = torch.cat((x, y), dim=1)
            z = self.fc(z)
        elif self.channel == "codon":
            x = self.codon_transformer(x)
            y = self.codon_transformer(y)

            x = self.codon_channel1(x)
            y = self.codon_channel2(y)

            z = torch.cat((x, y), dim=1)
            z = self.fc(z)

        return z


class CodonTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.codon_channel_num = 64
        self.codon_transformer = torch.zeros(64, 1, 3, 4)
        indicies = itertools.product(range(4), repeat=3)
        for i in range(self.codon_transformer.shape[0]):
            index = next(indicies)
            for j in range(3):
                self.codon_transformer[i, 0, j, index[j]] = 1
        self.codon_transformer = nn.Parameter(self.codon_transformer, requires_grad=False)
        self.padding_layers = [nn.ZeroPad2d((0, 0, 0, 2)), nn.ZeroPad2d((0, 0, 0, 1))]

    def forward(self, x):
        mod_len = int(x.shape[2] % 3)
        if mod_len != 2:
            x = self.padding_layers[mod_len](x)
        x = F.conv2d(x, self.codon_transformer) - 2
        x = F.relu(x)
        x = x.flatten(start_dim=2)

        x = x.view(-1, self.codon_channel_num, int(x.shape[2] // 3), 3)
        x = x.transpose(2, 3)
        x = x.reshape(-1, 1, self.codon_channel_num, x.shape[-1] * 3).transpose(2, 3)
        return x
