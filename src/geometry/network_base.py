import torch

class Network_Base(torch.nn.Module):
    def __init__(self):
        """A base class for networks"""
        super().__init__()

    def forward(self, x:torch.Tensor):
        raise NotImplementedError
    

class MLP(Network_Base):
    def __init__(self, input_dim=3, hidden=512, layers=3, outdim=3):
        """A simple MLP network that takes points as input"""
        super().__init__()
        mlp = []
        for i in range(layers - 1):
            mlp += [torch.nn.Linear(input_dim if i == 0 else hidden, hidden), torch.nn.LeakyReLU()]
        self.head = torch.nn.Linear(hidden, outdim)

        self.mlp = torch.nn.Sequential(*mlp)

        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor):
        # points: (B, P, 3)
        h = self.mlp(x)  # (B, P, hidden)
        displacement = self.head(h)  # (B, P, outdim)
        return displacement
    
class Encoder_Decoder(Network_Base):
    def __init__(self, encoder_layers=3, decoder_layers=3, input_dim=3, outdim=3):
        """A wrapper for an encoder-decoder architecture"""
        super().__init__()
        encoder = []
        for i in range(encoder_layers):
            encoder += [torch.nn.Conv1d(input_dim if i == 0 else 128*i, 128*(i+1), kernel_size=3, stride=2), torch.nn.LeakyReLU()]
        self.encoder = torch.nn.Sequential(*encoder)
        decoder = []
        for i in range(decoder_layers-1,0,-1):
            decoder += [torch.nn.ConvTranspose1d( 128*(i+1), 128*(i), kernel_size=3, stride=2), torch.nn.LeakyReLU()]
        self.decoder = torch.nn.Sequential(*decoder)
        self.head = torch.nn.ConvTranspose1d( 128, outdim, kernel_size=3, stride=2)
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor):
        # points: (B, P, 3)
        latent = self.encoder(x)  # (B, latent_dim)
        h = self.decoder(latent)  # (B, P, hidden)
        displacement = self.head(h)  # (B, P, outdim)
        return displacement