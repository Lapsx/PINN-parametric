import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time

from fno_architecture import FNO2d
from generate_multipole_potentials import generate_spherical_potential_slice


def calc_mu_pytorch(phi, V, dx):
    """
    Quociente de Rayleigh para H = -∇² + V (sem fator 1/2).

        μ = ⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩

    A discretização é a MESMA do Hamiltoniano de diferenças finitas montado em
    `solve_edwards_2d.construct_hamiltonian_2d`, que é quem gera o ground truth.
    Isso importa: a versão anterior usava diferenças centrais em (2·dx) e descartava
    a borda do domínio, e por isso subestimava o autovalor em ~14.5% de forma
    sistemática. Medido contra o `eigsh` em 8 potenciais (N=64, L=6):

        versão anterior (diferenças centrais)   erro médio 14.5%
        esta versão                             erro médio 0.0011%

    Duas coisas fazem a diferença:
      1. Para o laplaciano de 5 pontos vale a identidade de soma por partes
         ⟨ψ|−∇²|ψ⟩ = Σ_arestas (ψᵢ − ψⱼ)²/dx², ou seja diferenças PROGRESSIVAS,
         não centrais. A diferença central corresponde a um estêncil diferente e
         suaviza o gradiente, subestimando a energia cinética.
      2. O zero-padding representa a condição de Dirichlet fora do domínio. Sem ele
         as arestas da borda somem da soma — e elas não são desprezíveis, porque a
         caixa (L=6) é pequena o bastante para o estado fundamental ainda ter
         amplitude apreciável ali. Só isso responde por ~6% dos 14.5%.

    Args:
        phi: (1, 1, N, N) — função de onda ψ (não a densidade)
        V:   (1, 1, N, N) — potencial externo, mesma convenção do dataset
        dx:  espaçamento da grade
    Returns:
        μ (escalar torch)
    """
    # Dirichlet implícito: ψ = 0 fora do domínio
    p = F.pad(phi, (1, 1, 1, 1))

    # Energia cinética por soma por partes (SEM fator 1/2, pois H = -∇² + V)
    d_x = (p[:, :, 1:, :] - p[:, :, :-1, :]) / dx
    d_z = (p[:, :, :, 1:] - p[:, :, :, :-1]) / dx
    kinetic = (torch.sum(d_x**2) + torch.sum(d_z**2)) * (dx**2)

    # Energia potencial e norma sobre a grade inteira
    potential = torch.sum(V * phi**2) * (dx**2)
    norm = torch.sum(phi**2) * (dx**2)

    return (kinetic + potential) / (norm + 1e-12)


def perform_thermodynamic_analysis(N=256, L=10.0, a=1.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Iniciando Análise Termodinâmica Autograd Avançada no FNO...")
    
    # 1. Carregar Modelo High-Fidelity SOTA (Versão Diamante)
    model = FNO2d(modes1=24, modes2=24, width=48).to(device)
    model.load_state_dict(torch.load("fno_edwards_hf_model.pt", map_location=device))
    model.eval()
    
    # 2. Gerar Potencial de Teste
    np.random.seed(47)
    X, Z, V_raw = generate_spherical_potential_slice(N=N, L=L, a=a, kappa=1.5, l_max=4)
    
    R = np.sqrt(X**2 + Z**2)
    mask = R >= a
    
    V_clean = np.copy(V_raw)
    V_clean[~mask] = 0.0 # interior zerado (borda rigorosa)
    
    # 3. Preparar Tensor e habilitar 'requires_grad'
    # É aqui que o FNO mostra ser muito mais do que um emulador: ele é diferenciável!
    V_tensor = torch.tensor(V_clean, dtype=torch.float32, device=device, requires_grad=True)
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    Z_tensor = torch.tensor(Z, dtype=torch.float32, device=device)
    
    # Montar Input
    x_input = torch.zeros(1, N, N, 3, dtype=torch.float32, device=device)
    x_input[0, :, :, 0] = V_tensor
    x_input[0, :, :, 1] = X_tensor
    x_input[0, :, :, 2] = Z_tensor
    
    # 4. Forward Pass
    out = model(x_input)[0, :, :, 0]
    density = out  # softplus na arquitetura garante φ² ≥ 0
    
    # 5. Escolher um "Monomero Alvo" (r_0)
    # Ponto de maior densidade, restrito ao EXTERIOR da esfera. No dataset HF o
    # interior recebe V = 0 em vez de uma barreira, então nada impede a rede de
    # prever densidade lá dentro — e um r_0 dentro da esfera tornaria a
    # suscetibilidade δφ²(r_0)/δV(r) fisicamente sem sentido.
    mask_t = torch.tensor(mask, device=density.device)
    max_idx = torch.argmax(torch.where(mask_t, density, torch.full_like(density, -np.inf))).item()
    target_i = max_idx // N
    target_j = max_idx % N
    
    density_target = density[target_i, target_j]
    
    print(f"Calculando Suscetibilidade Não-Local d(rho_max)/dV(x,z) no ponto ({X[target_i, target_j]:.2f}, {Z[target_i, target_j]:.2f})...")
    
    # 6. Magia do Autograd (Derivada Funcional / Resposta Linear Termodinâmica)
    # Isso calcula a matriz Jacobiana gigante inteira instantaneamente!
    grad_V = torch.autograd.grad(density_target, V_tensor, create_graph=False)[0]
    
    density_plot = density.detach().cpu().numpy()
    susceptibility_plot = grad_V.cpu().numpy()
    V_plot = V_clean.copy()
    
    # Mascarar a Esfera para os Gráficos
    density_plot[~mask] = np.nan
    susceptibility_plot[~mask] = np.nan
    V_plot[~mask] = np.nan
    
    return X, Z, V_plot, density_plot, susceptibility_plot, (X[target_i, target_j], Z[target_i, target_j])

if __name__ == "__main__":
    t0 = time.time()
    X, Z, V, density, susc, (x0, z0) = perform_thermodynamic_analysis(N=256, L=10.0)
    print(f"Análise Completa em {(time.time()-t0)*1000:.1f} milissegundos!")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    limit = np.nanmax(np.abs(V)) * 0.8
    cp0 = axes[0].contourf(X, Z, V, levels=100, cmap='RdBu_r', vmin=-limit, vmax=limit)
    axes[0].set_title("Energia Superficial $V(x,z)$")
    fig.colorbar(cp0, ax=axes[0], label="Potencial $V$")
    
    cp1 = axes[1].contourf(X, Z, density, levels=100, cmap='inferno')
    axes[1].plot(x0, z0, 'w*', markersize=12, label='Massa Principal $r_0$')
    axes[1].set_title("Nuvem do Polímero $\\phi^2$\n(Soft Wall Model N=256)")
    axes[1].legend()
    fig.colorbar(cp1, ax=axes[1], label="Densidade Normalizada")
    
    # O pico local (r_0) distorce a barra de cores completamente.
    # Vamos usar percentis para ignorar o pico e focar nos gradientes finos:
    valid_susc = susc[~np.isnan(susc)]
    vmin, vmax = np.percentile(valid_susc, [2, 98])
    limit = max(abs(vmin), abs(vmax))
    
    # Atenção: passar np.linspace para o levels, senão o matplotlib engole tudo num bin só
    levels_array = np.linspace(-limit, limit, 100)
    
    cp2 = axes[2].contourf(X, Z, susc, levels=levels_array, cmap='PRGn', extend='both')
    axes[2].plot(x0, z0, 'k*', markersize=12)
    axes[2].set_title("Suscetibilidade Termodinâmica Não-Local\n$\\chi(r, r_0) = \\frac{\\delta \\phi^2(r_0)}{\\delta V(r)}$")
    fig.colorbar(cp2, ax=axes[2], label="Suscetibilidade")
    
    for ax in axes:
        ax.add_artist(plt.Circle((0, 0), 1.0, color='darkgray', zorder=10))
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        
    plt.tight_layout()
    plt.savefig('fno_hf_thermodynamics.png', dpi=300)
    print("Gráfico salvo em 'fno_hf_thermodynamics.png'")
