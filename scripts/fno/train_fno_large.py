import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import os

from generate_multipole_potentials import generate_spherical_potential_slice
from solve_edwards_2d import solve_ground_state
from fno_architecture import FNO2d


class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True):
        self.d = d
        self.p = p
        self.size_average = size_average

    def __call__(self, x, y):
        num_examples = x.size()[0]
        x = x.view(num_examples, -1)
        y = y.view(num_examples, -1)
        diff_norms = torch.norm(x - y, p=self.p, dim=1)
        y_norms = torch.norm(y, p=self.p, dim=1)
        loss = diff_norms / (y_norms + 1e-8)
        if self.size_average:
            return torch.mean(loss)
        return loss


def edwards_pde_loss(phi_pred, V_batch, dx, wall_val=10.0):
    """
    Resíduo da equação de Edwards no ground state: H·ψ = μ·ψ
    onde H = -∇² + V e φ = ψ².

    Reescrevendo para φ = ψ²:
        ∇²√φ = (V - μ)·√φ

    Na prática, computamos o resíduo diretamente em ψ = √φ:
        resíduo = -∇²ψ + V·ψ - μ·ψ

    onde μ é estimado pelo quociente de Rayleigh: μ = ⟨ψ|H|ψ⟩/⟨ψ|ψ⟩.

    Usa diferenças finitas (5 pontos) — sem autograd, sem create_graph.
    """
    # ψ = √φ (softplus garante φ > 0)
    psi = torch.sqrt(phi_pred + 1e-12)

    # Laplaciano via diferenças finitas centrais (interior apenas)
    # ∇²ψ = (ψ[i+1,j] + ψ[i-1,j] + ψ[i,j+1] + ψ[i,j-1] - 4ψ[i,j]) / dx²
    lap = (psi[:, 2:, 1:-1] + psi[:, :-2, 1:-1] +
           psi[:, 1:-1, 2:] + psi[:, 1:-1, :-2] -
           4.0 * psi[:, 1:-1, 1:-1]) / (dx**2)

    V_int = V_batch[:, 1:-1, 1:-1]
    psi_int = psi[:, 1:-1, 1:-1]

    # Máscara: ignorar pontos de parede (V ≈ 10)
    valid = (V_int < wall_val - 0.1)

    # Hψ = -∇²ψ + V·ψ  (nos pontos interiores)
    H_psi = -lap + V_int * psi_int

    # μ via Rayleigh por amostra: μ_i = Σ(ψ·Hψ) / Σ(ψ²)
    psi_H_psi = (psi_int * H_psi * valid) * (dx**2)
    psi_sq = (psi_int**2 * valid) * (dx**2)
    mu = psi_H_psi.sum(dim=(1, 2)) / (psi_sq.sum(dim=(1, 2)) + 1e-12)  # (B,)

    # Resíduo: (Hψ - μψ) nos pontos válidos
    residual = H_psi - mu.view(-1, 1, 1) * psi_int
    residual = residual * valid

    # MSE relativo do resíduo
    res_norm = torch.mean(residual**2, dim=(1, 2))
    psi_norm = torch.mean(psi_int**2, dim=(1, 2)) + 1e-12
    return torch.mean(res_norm / psi_norm)


def mass_conservation_loss(phi_pred, V_batch, dx, wall_val=10.0):
    """
    Penaliza desvios de ∫φ² dx dz = 1 (normalização do ground state).
    φ_pred já é φ = ψ², e o solver normaliza ∫ψ² = 1 → ∫φ·dx² deve ser ~1
    quando φ = ψ² e ∫ψ²·dx² = 1.
    """
    valid = (V_batch < wall_val - 0.1).float()
    mass = torch.sum(phi_pred * valid, dim=(1, 2)) * (dx**2)
    return torch.mean((mass - 1.0)**2)


def create_and_save_dataset(filename, n_samples=1000, N=64, L=6.0):
    if os.path.exists(filename):
        print(f"Dataset '{filename}' já existe! Carregando direto do disco...")
        data = torch.load(filename, weights_only=False)
        return data['inputs'], data['targets']

    print(f"Gerando NOVO dataset: {n_samples} amostras (Resolução {N}x{N})...")
    print("Essa etapa vai demorar alguns minutos. Você pode ir tomar um café!")
    dx = L / (N - 1)

    inputs = torch.zeros(n_samples, N, N, 3)
    targets = torch.zeros(n_samples, N, N, 1)
    # O autovalor do eigsh era calculado e jogado fora. Guardá-lo custa 4 bytes por
    # amostra e dispensa recalcular um quociente de Rayleigh na validação.
    mus = torch.zeros(n_samples)

    t_start = time.time()
    for i in range(n_samples):
        np.random.seed(i + hash(filename) % 10000)

        grid_X, grid_Z, V = generate_spherical_potential_slice(N=N, L=L, a=1.0, kappa=1.5, l_max=4)

        V_clean = np.copy(V)
        V_clean[np.isnan(V)] = 10.0

        phi, mu = solve_ground_state(grid_X, grid_Z, V, dx)
        density = phi**2
        density[np.isnan(V)] = 0.0

        inputs[i, :, :, 0] = torch.tensor(V_clean, dtype=torch.float32)
        inputs[i, :, :, 1] = torch.tensor(grid_X, dtype=torch.float32)
        inputs[i, :, :, 2] = torch.tensor(grid_Z, dtype=torch.float32)
        targets[i, :, :, 0] = torch.tensor(density, dtype=torch.float32)
        mus[i] = float(mu)

        if (i + 1) % max(1, (n_samples // 10)) == 0:
            elapsed = time.time() - t_start
            eta = elapsed / (i + 1) * (n_samples - i - 1)
            print(f"Progresso: {i + 1}/{n_samples} ({elapsed:.1f}s, ETA: {eta:.0f}s)")

    # 'mus' é opcional para quem lê: os scripts antigos só pedem inputs/targets.
    torch.save({'inputs': inputs, 'targets': targets, 'mus': mus}, filename)
    print(f"Dataset cacheado com sucesso em '{filename}'")
    print(f"  μ ∈ [{mus.min():.4f}, {mus.max():.4f}]\n")
    return inputs, targets


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando hardware: {device}")

    N_res = 64
    L_box = 6.0
    dx = L_box / (N_res - 1)

    n_train = 5000
    n_test = 500

    train_file = f"dataset_fno_train_{n_train}_N{N_res}.pt"
    test_file = f"dataset_fno_test_{n_test}_N{N_res}.pt"

    x_train, y_train = create_and_save_dataset(train_file, n_train, N=N_res, L=L_box)
    x_test, y_test = create_and_save_dataset(test_file, n_test, N=N_res, L=L_box)

    x_train, y_train = x_train.to(device), y_train.to(device)
    x_test, y_test = x_test.to(device), y_test.to(device)

    model = FNO2d(modes1=16, modes2=16, width=48).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
    criterion = LpLoss()

    epochs = 500
    batch_size = 25

    # Pesos da physics loss com warm-up linear
    lambda_pde_max = 0.1
    lambda_mass_max = 0.05
    warmup_start = 50
    warmup_end = 200

    print("\n=======================================================")
    print(f" TREINAMENTO PHYSICS-INFORMED ({epochs} ÉPOCAS)")
    print(f" Dataset: {n_train} train + {n_test} test")
    print(f" Physics: PDE Edwards + Conservação de Massa")
    print("=======================================================")

    t_train = time.time()
    for ep in range(epochs):
        model.train()
        train_l2 = 0.0
        train_pde = 0.0
        train_mass = 0.0

        # Warm-up dos pesos de physics
        if ep < warmup_start:
            w_phys = 0.0
        elif ep < warmup_end:
            w_phys = (ep - warmup_start) / (warmup_end - warmup_start)
        else:
            w_phys = 1.0
        lambda_pde = lambda_pde_max * w_phys
        lambda_mass = lambda_mass_max * w_phys

        permutation = torch.randperm(n_train)

        for b in range(0, n_train, batch_size):
            indices = permutation[b:b + batch_size]
            batch_x = x_train[indices]
            batch_y = y_train[indices]

            optimizer.zero_grad()
            out = model(batch_x)

            # Loss L2 relativa (data-driven)
            loss_l2 = criterion(out, batch_y)

            # Physics losses (avaliadas no output sem autograd)
            phi_pred = out[:, :, :, 0]
            V_batch = batch_x[:, :, :, 0]

            loss_pde = edwards_pde_loss(phi_pred, V_batch, dx)
            loss_mass = mass_conservation_loss(phi_pred, V_batch, dx)

            loss = loss_l2 + lambda_pde * loss_pde + lambda_mass * loss_mass

            loss.backward()
            optimizer.step()

            bs = len(indices)
            train_l2 += loss_l2.item() * bs
            train_pde += loss_pde.item() * bs
            train_mass += loss_mass.item() * bs

        scheduler.step()
        train_l2 /= n_train
        train_pde /= n_train
        train_mass /= n_train

        # Validação
        model.eval()
        with torch.no_grad():
            test_l = 0.0
            for tb in range(0, x_test.shape[0], batch_size):
                out_test = model(x_test[tb:tb + batch_size])
                test_l += criterion(out_test, y_test[tb:tb + batch_size]).item() * out_test.shape[0]
            test_l /= x_test.shape[0]

        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"Ep {ep+1:03d} | L2: {train_l2:.4f} | PDE: {train_pde:.4f} | Mass: {train_mass:.6f} | Test: {test_l:.4f} | λ_pde: {lambda_pde:.3f}")

        if (ep + 1) % 100 == 0:
            torch.save(model.state_dict(), f"fno_edwards_large_ep{ep+1}.pt")

    print(f"\nTreinamento finalizado em {(time.time() - t_train) / 60:.1f} minutos!")

    torch.save(model.state_dict(), "fno_edwards_large_model.pt")
    print("Pesos finais salvos como 'fno_edwards_large_model.pt'")

    # Avaliação gráfica
    model.eval()
    with torch.no_grad():
        idx = 42
        pred = model(x_test[idx:idx + 1])[0, :, :, 0].cpu().numpy()
        true = y_test[idx, :, :, 0].cpu().numpy()
        pot = x_test[idx, :, :, 0].cpu().numpy()

    pred[pot >= 9.9] = np.nan
    true[pot >= 9.9] = np.nan
    pot[pot >= 9.9] = np.nan

    X_grid = x_test[idx, :, :, 1].cpu().numpy()
    Z_grid = x_test[idx, :, :, 2].cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    cp0 = axes[0].contourf(X_grid, Z_grid, pot, levels=60, cmap='RdBu_r')
    axes[0].set_title('Entrada: $V(x,z)$')
    fig.colorbar(cp0, ax=axes[0])

    cp1 = axes[1].contourf(X_grid, Z_grid, true, levels=60, cmap='inferno')
    axes[1].set_title('Gabarito Numérico (SciPy)')
    fig.colorbar(cp1, ax=axes[1])

    cp2 = axes[2].contourf(X_grid, Z_grid, pred, levels=60, cmap='inferno')
    axes[2].set_title('Previsão FNO (Physics-Informed)')
    fig.colorbar(cp2, ax=axes[2])

    for ax in axes:
        ax.add_artist(plt.Circle((0, 0), 1.0, color='darkgray', zorder=10))
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('fno_comparison_large.png', dpi=300)
    print("Gráfico salvo em 'fno_comparison_large.png'")
