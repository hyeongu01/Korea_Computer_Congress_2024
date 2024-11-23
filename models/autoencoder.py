import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        self.relu = nn.LeakyReLU(0.2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            self.relu,
            
            nn.Conv2d(16,  32, kernel_size=3, padding=1),    # (1,1,28,28) -> (1,32,28,28)
            self.pool,                                      # (1,32,14,14)
            self.relu,
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),    # (1,32,14,14) -> (1,64,14,14)
            self.pool,                                      # (1,64,7,7)
            self.relu
        )
        
        self.linear = nn.Sequential(
            nn.Linear(3136, 1024),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 3136),
            nn.Dropout(p=0.3),
        )
        
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),    # (1,64,7,7) -> (1,64,14,14)
            # concat encoder[4] 32+64 = 96                  # (1,96,14,14)
            nn.Conv2d(96, 32, kernel_size=3, padding=1),                              # (1,32,14,14)
            self.relu,
            
            nn.Upsample(scale_factor=2, mode='nearest'),    # (1,32,28,28)
            # concat encoder[0] 32+16 = 48                  # (1,48,28,28)
            nn.Conv2d(48, 16, kernel_size=3, padding=1),    # (1,16,28,28)
            
            nn.Conv2d(16, 1, kernel_size=3, padding=1),     # (1,1,28,28)
            nn.Sigmoid()
        )
        
    # x: (1,1,28,28)
    def forward(self, x):
        
        # encoder
        encoder_1 = self.encoder[0:2](x)
        encoder_2 = self.encoder[2:5](encoder_1)
        x = self.encoder[5:](encoder_2)
        
        # linear
        batch, channel = x.shape[0], x.shape[1]
        x = x.view(batch, -1)
        x = self.linear(x)
        x = x.view(batch, channel, 7, 7)
        
        # decoder with skip-connection
        x = self.decoder[0](x)
        x = torch.cat((x, encoder_2), dim=1)
        x = self.decoder[1:4](x)
        x = torch.cat((x, encoder_1), dim=1)
        x = self.decoder[4:](x)
        
        return x

def load_model(pretrained=None):
    model = Autoencoder()

    if pretrained:
        checkpoint = torch.load(pretrained, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])

    return model
