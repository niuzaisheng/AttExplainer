
# %%
import torch
from torch import nn

# %%

class DQNNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.channel_mix = nn.Conv2d(in_channels=12*12, out_channels=64, kernel_size=1, stride=1, padding=1)

        self.down_conv_x = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=48, kernel_size=5, stride=(1, 2), padding=1),
            nn.BatchNorm2d(num_features=48),
            nn.ReLU(),

            nn.Conv2d(in_channels=48, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
        )

        self.down_conv_y = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=48, kernel_size=5, stride=(2, 1), padding=1),
            nn.BatchNorm2d(num_features=48),
            nn.ReLU(),

            nn.Conv2d(in_channels=48, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
        )

        self.out = nn.Linear(128+1, 1)
        self.dueling = nn.Linear(128+1, 1)
        
        # self.down_conv2 = nn.Sequential(
        #     nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(num_features=256),
        #     nn.ReLU(),

        #     nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(num_features=256),
        #     nn.ReLU(),

        #     nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(num_features=256),
        #     nn.ReLU()
        # )
        
        # self.flat_conv = nn.Sequential(    
        #     nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(num_features=256),
        #     nn.ReLU(),

        #     nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(num_features=512),
        #     nn.ReLU(),
            
        #     nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(num_features=1024),
        #     nn.ReLU(),
            
        #     nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(num_features=1024),
        #     nn.ReLU(),
        # ) 
        

    def forward(self, x, s):
        """_summary_

        Args:
            x (Tensor): attention matrix with size of [Batch x (head*layer_num) x seq_len x seq_len]
            s (Tensor): game status with size of [Batch x seq_len]

        Returns:
            Tensor: Q value
        """

        batch_size, matrix_num, max_seq_len, _ = x.size()

        x = self.channel_mix(x) # [B, 64, seq, seq]
        xx = self.down_conv_x(x) # [B, 128, seq, ?]
        xy = self.down_conv_y(x) # [B, 128, ?, seq]
        x = torch.cat([xx, xy.transpose(-1,-2)], dim = -1) # [B, 128, seq, ?]
        x = x.mean(-1) # average pooling
        # x, _ = x.max(-1) # max pooling
        x = torch.cat([x, s.unsqueeze(1)], dim=1) # [B, 129, seq]
        x = x.transpose(1,2) # [B, seq, 129]
        v = self.out(x).squeeze(-1) # [B, seq]
        adv = self.dueling(x).squeeze(-1) # [B, seq]
        adv_mean = adv.mean(1, keepdim=True) # [B, 1]

        return v + (adv - adv_mean)

net = DQNNet()
x = torch.zeros((2,12*12,10,10))
s = torch.zeros((2,10))
out = net(x,s)
print(out.size())
# %%
