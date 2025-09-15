import torch
import torch.nn as nn
from torch.nn import Conv1d, MaxPool1d, Linear


class ConvKRegionCNNLSTM(nn.Module):
    """
    先用卷积提取局部模式，再用LSTM提取时序依赖
    - 设计为“按通道独立抽取特征”：输入可以是 [B, k, T]，也可由上游展平为 [B*k, 1, T]
    - 我们在 forward 中将 [B, k, T] 展平为 [B*k, 1, T]，因此第一层卷积的 in_channels=1
    - 输出形状为 [B*k, out_size]，便于还原为 [B, k, out_size]
    """

    def __init__(self, k=1, out_size=16, time_series=100,
                 conv_channels=[32, 32, 16], kernel_sizes=[8, 8, 8],
                 pool_size=16, lstm_hidden=32, lstm_layers=1, bidirectional=True):
        super().__init__()
        self.k = k
        self.time_series = time_series
        self.bidirectional = bidirectional
        self.layer_norm = nn.LayerNorm(self.time_series)

        layers = []
        # 按通道独立抽取特征 -> 第一层卷积输入通道为 1
        in_channels = 1
        for i, (out_channels, ks) in enumerate(zip(conv_channels, kernel_sizes)):
            layers.append(Conv1d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=ks, stride=2 if i==0 else 1))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.LeakyReLU(0.2))
            in_channels = out_channels

        layers.append(MaxPool1d(pool_size))
        self.conv = nn.Sequential(*layers)

        dummy = torch.zeros(1, 1, self.time_series)
        conv_out = self.conv(dummy)        # [1, C, L]
        self.conv_out_channels = conv_out.shape[1]
        self.conv_out_len = conv_out.shape[2]

        self.lstm = nn.LSTM(
            input_size=self.conv_out_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        # fully connected projection
        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)
        self.linear = nn.Sequential(
            Linear(lstm_out_dim, 64),
            nn.LeakyReLU(0.2),
            Linear(64, out_size)
        )

    def forward(self, x):
        """
        x: [B, k, T]
        return: [B*k, out_size]
        """
        B, k, T = x.shape
        x = self.layer_norm(x)

        x = x.reshape(B * k, 1, T)

        x = self.conv(x)

        x = x.transpose(1, 2)

        lstm_out, _ = self.lstm(x)

        lstm_feat = lstm_out[:, -1, :]  

        out = self.linear(lstm_feat)    

        return out
        
class ConvOnlyFeatureExtractor(nn.Module):
    def __init__(self, out_size=16, conv_channels=[32, 32, 16], kernel_sizes=[8, 8, 8]):
        super().__init__()
        self.out_size = out_size
        layers = []
        in_channels = 1
        for i, (out_channels, ks) in enumerate(zip(conv_channels, kernel_sizes)):
            layers.append(nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=ks, stride=2 if i == 0 else 1))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.LeakyReLU(0.2))
            in_channels = out_channels

        self.conv = nn.Sequential(*layers)
        self.last_conv_channels = conv_channels[-1]

        self.linear = nn.Sequential(
            nn.Linear(self.last_conv_channels, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, self.out_size)
        )

    def forward(self, x):
        """
        x: [B, k, T]
        return: [B*k, out_size]
        """
        B, k, T = x.shape
        x = x.reshape(B * k, 1, T)
        x = self.conv(x)
        x = x.mean(dim=-1)  # [B*k, C_out]
        out = self.linear(x)  # [B*k, out_size]
        return out
#“包装器”把 [B, T, C] 变成 [B*k, 1, T]
class NodeFeatureExtractor(nn.Module):
    def __init__(self,conv:nn.Module,out_size:int):
        super().__init__()
        self.conv = conv
        self.out_size = out_size
    def forward(self, batch_data):
        """
        batch_data: [B, T, C]
        return: [B, C, out_size]
        """
        B,T,C=batch_data.shape
        x_nodes=batch_data.permute(0,2,1)           
        x_reshape=x_nodes.reshape(B*C,1,T)         
        feats_flat=self.conv(x_reshape)             
        feats=feats_flat.view(B,C,self.out_size)    
        return feats