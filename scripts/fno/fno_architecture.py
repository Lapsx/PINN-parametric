import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Quantidade de baixas frequências de Fourier que vamos preservar na convolução
        self.modes1 = modes1 
        self.modes2 = modes2 
        
        self.scale = (1 / (in_channels * out_channels))
        # Tensores de pesos Complexos para multiplicar os modos no espaço de frequência
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        
        # 1. Transformada de Fourier 2D Real (Leva o campo espacial para frequências)
        x_ft = torch.fft.rfft2(x)
        
        # O tensor resultante da convolução espetral inicia zerado
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        
        # 2. Multiplicar apenas os modos de BAIXA FREQUÊNCIA (Filtragem Passa-Baixa Inteligente)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        
        # 3. Transformada de Fourier Inversa 2D (Traz de volta pro espaço real)
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1=16, modes2=16, width=32):
        super(FNO2d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        
        # Camada Lifting (P): Mapeia os Canais de Entrada para a dimensão "width"
        # O nosso Input terá 3 canais: (V(x,z), x, z) 
        # Passar x e z ajuda muito o FNO a entender os limites do mapa!
        self.p = nn.Linear(3, self.width)
        
        # As 4 Camadas principais de Fourier
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        
        # Camadas bypass lineares (ResNet-style, operam no domínio espacial diretamente)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        
        # Camada Projetora (Q): Transforma a abstração final na nossa função phi(x,z)
        self.q = nn.Sequential(
            nn.Linear(self.width, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # O PyTorch tradicional usa [Batch, Channel, X, Y], mas no começo passamos [B, X, Y, C]
        x = self.p(x)
        x = x.permute(0, 3, 1, 2) 
        
        # Bloco 1
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = F.gelu(x1 + x2)
        
        # Bloco 2
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = F.gelu(x1 + x2)
        
        # Bloco 3
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = F.gelu(x1 + x2)
        
        # Bloco 4
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = F.gelu(x1 + x2)
        
        # Saída
        x = x.permute(0, 2, 3, 1) # Volta o canal para o fim
        x = self.q(x)
        # Densidade φ² é não-negativa por definição física
        x = F.softplus(x)
        return x

if __name__ == "__main__":
    print("Testando a arquitetura FNO 2D...")
    
    # Criamos o modelo com 16 modos de frequência conservados e largura de 32 canais
    model = FNO2d(modes1=16, modes2=16, width=32)
    
    # Número de parâmetros
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Modelo FNO instanciado! Total de parâmetros: {n_params:,}")
    
    # Simulando um min-batch com tamanho arbitrário de grade
    # (Note como FNO suporta qualquer tamanho de grade N x M)
    B = 1         # Batch de 1 matriz
    Nx = 100      # 100 pontos em X
    Nz = 100      # 100 pontos em Z
    C = 3         # Canais (Potencial V, pos X, pos Z)
    
    dummy_input = torch.randn(B, Nx, Nz, C)
    
    print(f"\nFormato de Entrada (X): {list(dummy_input.shape)} -> [Batch, Nx, Nz, Canais(V,x,z)]")
    
    out = model(dummy_input)
    
    print(f"Formato de Saída   (Y): {list(out.shape)} -> [Batch, Nx, Nz, Canais(Phi)]")
    print("\nA operação Forward foi concluída com sucesso. A arquitetura está pronta!")
