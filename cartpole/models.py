import torch
import torch.nn as nn
from config import device, lat

class AE(nn.Module):
    def __init__(self, image_channels=3, h_dim=216, h3_dim=16, z_dim=lat):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 4, kernel_size=5, stride=2),
            nn.Mish(),
            nn.Conv2d(4, 6, kernel_size=5, stride=2),
            nn.Mish(),
            nn.Conv2d(6, 12, kernel_size=5, stride=2),
            nn.Mish(),
            nn.Conv2d(12, 16, kernel_size=5, stride=2),
            nn.Mish(),
            nn.Conv2d(16, 24, kernel_size=5, stride=2),
            nn.Mish(),
            nn.Flatten(),
            nn.LayerNorm(h_dim),
            nn.Linear(h_dim, z_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.Dropout(0.2),
            nn.Mish(),
            nn.Unflatten(1, (h_dim, 1, 1)),
            nn.ConvTranspose2d(h_dim, 24, kernel_size=7, stride=2),
            nn.Mish(),
            nn.ConvTranspose2d(24, 16, kernel_size=7, stride=2),
            nn.Mish(),
            nn.ConvTranspose2d(16, 8, kernel_size=7, stride=2),
            nn.Mish(),
            nn.ConvTranspose2d(8, 4, kernel_size=9, stride=2),
            nn.Mish(),
            nn.ConvTranspose2d(4, image_channels, kernel_size=8, stride=2),
            nn.Sigmoid()
        )

    def encode(self, x):
        z = self.encoder(x)
        return z

    def decode(self, z):
        if self.training:
            mult = torch.randn(z.shape).to(device) * 0.10 + 1
            z_noise = z * mult
        else:
            z_noise = z
        z_noise = self.decoder(z_noise)
        return z_noise

    def forward(self, x):
        z = self.encode(x)
        z = self.decode(z)
        return z
    
class GameLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm_layers = 3
        self.lstm_size = 128
        self.lstm1 = nn.LSTM(lat+1, self.lstm_size, num_layers=self.lstm_layers, dropout=0.1, batch_first=True)
        self.batch_size = 1
        self.zero_hidden()
        self.remaining = nn.Sequential(
            nn.Linear(128, 512),
            nn.Dropout(0.1),
            nn.Mish(),
            nn.Linear(512, 1024),
            nn.Dropout(0.1),
            nn.Mish(),
            nn.Linear(1024, 512),
            nn.Dropout(0.1),
            nn.Mish(),
            nn.Linear(512, lat),
        )

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.zero_hidden()

    def zero_hidden(self):
        self.lstm_state1 = (torch.zeros(self.lstm_layers, self.batch_size, self.lstm_size).to(device), torch.zeros(self.lstm_layers, self.batch_size, self.lstm_size).to(device))
    
    def forward(self, data):
        data = torch.cat((data[:,:,:lat], data[:,:,lat:]), dim=-1)
        lstm_out_1, self.lstm_state1 = self.lstm1(data, self.lstm_state1)
        out = nn.Dropout(0.1)(lstm_out_1)#.view(-1, 24))
        out = nn.Mish()(out)
        out = self.remaining(out)
        out = torch.cat((out[:,:,:lat], out[:,:,lat:]), dim=-1)
        return out
    
class GameLSTM_ONNX(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm_layers = 3
        self.lstm_size = 128
        self.lstm1 = nn.LSTM(lat+1, self.lstm_size, num_layers=self.lstm_layers)
        self.batch_size = 1
        self.zero_hidden()
        self.remaining = nn.Sequential(
            nn.Linear(128, 512),
            nn.Dropout(0.1),
            nn.Mish(),
            nn.Linear(512, 1024),
            nn.Dropout(0.1),
            nn.Mish(),
            nn.Linear(1024, 512),
            nn.Dropout(0.1),
            nn.Mish(),
            nn.Linear(512, lat),
        )
    
    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.zero_hidden()
    
    def zero_hidden(self):
        self.lstm_state1 = (torch.zeros(self.lstm_layers, self.batch_size, self.lstm_size).to(device), torch.zeros(self.lstm_layers, self.batch_size, self.lstm_size).to(device))
    
    def forward(self, data, h0, c0):
        lstm_out_1, (hn, cn) = self.lstm1(data, (h0, c0))
        out = nn.Dropout(0.0)(lstm_out_1)
        out = nn.Mish()(out)
        out = self.remaining(out)
        return out, hn, cn

def load_model(path):
    return torch.load(path, map_location=device)

def load_ae(path):
    model = AE().to(device, memory_format=torch.channels_last)
    model.load_state_dict(torch.load(path, weights_only=True, map_location=device))
    return model

def load_lstm(path):
    model = GameLSTM().to(device, memory_format=torch.channels_last)
    model.load_state_dict(torch.load(path, weights_only=True, map_location=device))
    return model

def load_lstm_onnx(path):
    model = GameLSTM_ONNX().to(device, memory_format=torch.channels_last)
    model.load_state_dict(torch.load(path, weights_only=True, map_location=device))
    return model

def save_model(model, path):
    torch.save(model, path)

def new_ae():
    return AE().to(device, memory_format=torch.channels_last)

def new_lstm():
    return GameLSTM()