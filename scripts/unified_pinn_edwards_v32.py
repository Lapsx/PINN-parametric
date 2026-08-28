"""
PINN Unificado — Equação de Edwards / Adsorção de Polieletrólito em Esfera  (v32)
================================================================================
Mudança em relação ao v31: a rede de φ opera na coordenada reescalada

    s = √μ · (u − κa)

em vez de u_rel diretamente, e a amplitude é fatorada analiticamente como μ^(1/4).

Motivação (medida no v31 em 22/ago/2026):
  O v31 acerta μ com erro relativo mediano de 3.8% (R² = 0.988 em log10) ao longo
  de 10 décadas, mas φ COLAPSA PARA ZERO quando μ ≳ 1e4 — 100% de erro L2.
  Causa: o envelope usava alpha = clamp(√(μ+0.1), max=3.0), ou seja comprimento de
  decaimento fixo em 1/3, enquanto o pico verdadeiro tem largura ~1/√μ (≈0.002 para
  μ=2e5). Na loss de âncora — MSE contra um perfil que é ~0 em 2998 de 3000 pontos —
  o mínimo mais barato é φ ≡ 0, e a rede aprendeu exatamente isso. A assinatura fica
  visível no training_history_v31.json: a loss `norm` nunca desce abaixo de ~20,
  isto é, ∫φ² = κ nunca é satisfeito para a fração de amostras com μ grande.

Por que s = √μ·u_rel e A = μ^(1/4):
  φ(u) = A·g(s) ⟹ ∫φ² du = A²/√μ · ∫g² ds. Com A = μ^(1/4) a normalização vira
  ∫g² ds = κ, independente de μ. E a PDE φ_uu − (μ+V)φ = 0 vira g'' − (1 + V/μ)g = 0,
  também O(1). A rede só precisa produzir uma FORMA O(1); toda a dependência em escala
  (5 décadas de largura, 2 décadas de amplitude) é analítica.

  Verificado numericamente no intervalo μ ∈ [0.93, 8.9e7]:
      s do pico       : 0.92 → 6.05   (7×,  contra 5 décadas em u_rel)
      s de 90% da massa: 2.16 → 11.6
      pico · μ^(−1/4)  : 0.262 → 0.112 (2.3×, contra 5 décadas em amplitude)

A rede de μ é idêntica à do v31 (já validada) e pode ser inicializada a partir dos
pesos do v31 com --init-from, o que encurta muito o treino.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import os
from scipy.linalg import eigh_tridiagonal
from scipy.special import erfc
import argparse
import time
import json

torch.manual_seed(42)
np.random.seed(42)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Parâmetros físicos (Cherstvy & Winkler 2011)
params = {
    'delta_min': 0.01, 'delta_max': 5000.0,
    'ka_min': 0.01, 'ka_max': 10.0,
    'u_max_rel': 12.0,
    'kappa': 0.1,
}

# Coordenada reescalada: s = √μ · u_rel, truncada em S_MAX.
# S_MAX = 25 cobre com folga o pior caso medido (90% da massa em s = 11.6).
S_MAX = 25.0
S_BC = 4.0    # taxa de subida da condição de contorno φ(u_rel=0) = 0
S_TAU = 4.0   # comprimento de decaimento do envelope em s
N_ANC_PTS = 400  # pontos por âncora, reamostrados uniformemente em s


def get_args():
    p = argparse.ArgumentParser(description='PINN Edwards v32 - coordenada reescalada por sqrt(mu)')
    p.add_argument('--epochs', type=int, default=60000)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--pretrain', type=int, default=50000)
    p.add_argument('--neurons', type=int, default=256)
    p.add_argument('--layers', type=int, default=6)
    p.add_argument('--lbfgs-rounds', type=int, default=50)
    p.add_argument('--init-from', type=str, default='unified_pinn_v31_model.pt',
                   help='carrega a mu_net deste checkpoint (mesma arquitetura do v31)')
    p.add_argument('--freeze-mu', type=int, default=10000,
                   help='congela a mu_net por N épocas quando inicializada do v31')
    return p.parse_args()


try:
    args = get_args()
except BaseException:
    class Args:
        epochs = 60000; lr = 3e-4; neurons = 256; layers = 6
        pretrain = 50000; lbfgs_rounds = 50
        init_from = 'unified_pinn_v31_model.pt'; freeze_mu = 10000
    args = Args()


# ---------------------------------------------------------------------------
# 2. Teoria WKB para o limiar de adsorção
# ---------------------------------------------------------------------------
def _sqrt_mu_floor(mu):
    """√μ com piso em S_MAX/u_max_rel.

    Para μ muito pequeno (perto da transição e na região não-adsorvida), s/√μ
    estouraria o domínio físico u_rel ≤ 12 e todos os pontos de colocação
    colapsariam sobre a borda. O piso faz a grade em s cobrir exatamente [0, 12]
    em u_rel nesse regime.
    """
    return torch.clamp(torch.sqrt(torch.clamp(mu, min=1e-12)),
                       min=S_MAX / params['u_max_rel'])


def get_wkb_delta_c(ka_val):
    C = 0.973
    ka_np = ka_val.detach().cpu().numpy() if isinstance(ka_val, torch.Tensor) \
        else np.asarray(ka_val, dtype=np.float64)
    num = 6 * ka_np * (1.0 + ka_np) * (C**2)
    den = 2 * np.pi * np.exp(ka_np) * (erfc(np.sqrt(ka_np / 2.0))**2)
    return num / (den + 1e-30)


# ---------------------------------------------------------------------------
# 3. Solver numérico de referência
# ---------------------------------------------------------------------------
def _solve_on_domain(delta_val, ka_val, L, n_grid):
    """Estado fundamental de H = -d²/du² + V(u) em [κa, κa+L], Dirichlet nas bordas.

    Usa eigh_tridiagonal (LAPACK) em vez de eigsh: para a matriz tridiagonal deste
    problema é exato, muito mais rápido e não falha para μ perto de zero, onde o
    eigsh do v31 chegava a levar minutos por ponto.
    """
    kappa = params['kappa']
    a = ka_val / kappa
    u = np.linspace(ka_val, ka_val + L, n_grid)
    du = u[1] - u[0]
    pre_factor = (delta_val / (kappa * a)) * (np.exp(ka_val) / (1.0 + ka_val))
    V = -pre_factor * (np.exp(-u) / (u + 1e-12))

    diag = 2.0 / du**2 + V[1:-1]
    off = np.full(len(diag) - 1, -1.0 / du**2)
    w, vecs = eigh_tridiagonal(diag, off, select='i', select_range=(0, 0))
    mu_val = -w[0]
    if mu_val <= 1e-6:
        return u, np.zeros_like(u), 1e-6
    phi = np.zeros(n_grid)
    phi[1:-1] = np.abs(vecs[:, 0])
    norm = np.trapz(phi**2, u)
    phi = phi / (np.sqrt(norm) + 1e-12) * np.sqrt(kappa)
    return u, phi, mu_val


def solve_edwards_numerical(delta_val, ka_val, n_grid=4000):
    """Solver com domínio ADAPTADO a μ — esta é a segunda metade da correção do v32.

    O v31 resolvia sempre em [κa, κa + 18] com 3000 pontos. Para μ = 2e5 o pico
    inteiro cabia em ~2 pontos dessa grade, então as âncoras eram, na prática,
    vetores de zeros e não havia gradiente capaz de ensinar a forma do pico.
    Aqui o domínio é encolhido para cobrir exatamente s ∈ [0, S_MAX], de modo que o
    pico é sempre resolvido pelos mesmos ~4000 pontos, qualquer que seja μ.
    """
    L_full = params['u_max_rel'] * 1.5
    u, phi, mu_val = _solve_on_domain(delta_val, ka_val, L_full, n_grid)
    if mu_val > 1e-6:
        L_adapt = S_MAX / np.sqrt(mu_val)
        if L_adapt < L_full:
            u, phi, mu_val = _solve_on_domain(delta_val, ka_val, L_adapt, n_grid)
    return u, phi, mu_val


# ---------------------------------------------------------------------------
# 4. Arquitetura
# ---------------------------------------------------------------------------
class RFFEncoding(nn.Module):
    """Random Fourier Features."""
    def __init__(self, in_features, out_features, sigma=1.0):
        super().__init__()
        self.B = nn.Parameter(torch.randn(in_features, out_features) * sigma,
                              requires_grad=False)

    def forward(self, x):
        v = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(v), torch.cos(v)], dim=-1)


class ModifiedMLP(nn.Module):
    def __init__(self, in_f, h_f, layers=6):
        super().__init__()
        self.enc = RFFEncoding(in_f, 128)
        self.U = nn.Linear(256, h_f)
        self.V = nn.Linear(256, h_f)
        self.hidden = nn.ModuleList([nn.Linear(h_f, h_f) for _ in range(layers)])
        self.out = nn.Linear(h_f, 1)
        self.act = nn.Tanh()

    def forward(self, x):
        e = self.enc(x)
        u_gate = self.act(self.U(e))
        v_gate = self.act(self.V(e))
        h = u_gate
        for l in self.hidden:
            z = torch.sigmoid(l(h))
            h = (1 - z) * u_gate + z * v_gate
        return self.out(h)


class UnifiedPINNv32(nn.Module):
    """φ = μ^(1/4) · bc(s) · decay(s) · softplus(net(s, δ, κa)),  s = √μ·(u−κa).

    A mu_net é idêntica à do v31 (prediz log10 μ) e pode ser carregada de lá.
    A phi_net recebe 4 entradas: [log1p(s)/log1p(S_MAX), s/S_MAX, δ_norm, κa_norm].
    """
    def __init__(self, neurons=256, layers=6):
        super().__init__()
        self.phi_net = ModifiedMLP(4, neurons, layers=layers)
        self.mu_net = nn.Sequential(
            nn.Linear(2, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, 128), nn.SiLU(),
            nn.Linear(128, 1)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

    def _norm_params(self, d, ka):
        dn = (torch.log10(torch.clamp(d, min=params['delta_min'])) - np.log10(params['delta_min'])) \
            / (np.log10(params['delta_max']) - np.log10(params['delta_min']))
        kan = (torch.log10(torch.clamp(ka, min=params['ka_min'])) - np.log10(params['ka_min'])) \
            / (np.log10(params['ka_max']) - np.log10(params['ka_min']))
        return dn, kan

    def mu_only(self, d, ka):
        """μ sem avaliar a phi_net — usado para montar a grade em s."""
        dn, kan = self._norm_params(d, ka)
        log_mu = self.mu_net(torch.cat([dn, kan], dim=1))
        return torch.pow(10.0, torch.clamp(log_mu, -6.0, 8.0))

    def forward(self, u_abs, d, ka):
        if d.shape[0] != u_abs.shape[0]:
            d = d.expand(u_abs.shape[0], -1)
        if ka.shape[0] != u_abs.shape[0]:
            ka = ka.expand(u_abs.shape[0], -1)

        dn, kan = self._norm_params(d, ka)
        log_mu = self.mu_net(torch.cat([dn, kan], dim=1))
        mu = torch.pow(10.0, torch.clamp(log_mu, -6.0, 8.0))

        u_rel = torch.clamp(u_abs - ka, min=0.0)

        # Coordenada natural. sqrt_mu é destacado: a phi_net aprende uma forma para
        # um dado μ, e não move μ para facilitar a própria vida.
        sqrt_mu = torch.sqrt(mu.detach())
        s = torch.clamp(sqrt_mu * u_rel, max=S_MAX)

        s_log = torch.log1p(s) / np.log1p(S_MAX)
        s_lin = s / S_MAX

        p_raw = self.phi_net(torch.cat([s_log, s_lin, dn, kan], dim=1))
        f = F.softplus(p_raw)

        bc = 1.0 - torch.exp(-S_BC * s)          # Dirichlet em u_rel = 0
        decay = torch.exp(-s / S_TAU)            # envelope O(1) em s
        g = bc * decay * f                       # forma normalizada: ∫g² ds = κ

        amp = torch.pow(mu.detach(), 0.25)       # amplitude analítica
        phi = amp * g
        return phi, mu


# ---------------------------------------------------------------------------
# 5. Treino
# ---------------------------------------------------------------------------
def train():
    model = UnifiedPINNv32(neurons=args.neurons, layers=args.layers).to(device)

    mu_frozen = False
    if args.init_from and os.path.exists(args.init_from):
        sd = torch.load(args.init_from, map_location=device)
        mu_sd = {k[len('mu_net.'):]: v for k, v in sd.items() if k.startswith('mu_net.')}
        if mu_sd:
            model.mu_net.load_state_dict(mu_sd)
            print(f"[+] mu_net inicializada de {args.init_from} (já validada: erro mediano 3.8%).")
            if args.freeze_mu > 0:
                for p in model.mu_net.parameters():
                    p.requires_grad_(False)
                mu_frozen = True
                print(f"[+] mu_net congelada pelas primeiras {args.freeze_mu} épocas.")

    history = {'epoch': [], 'total': [], 'pde': [], 'norm': [], 'anchor': [], 'unad': []}

    print("Pré-calculando âncoras (grade 10x10, domínio adaptado a mu)...")
    t_anc = time.time()
    anchors = []
    grid_ka = np.logspace(-2, 1, 10)
    grid_delta = np.logspace(-2, 3.7, 10)
    for ka in grid_ka:
        for d in grid_delta:
            u_n, phi_n, mu_n = solve_edwards_numerical(d, ka)
            if mu_n <= 1e-6:
                continue
            # Reamostra em s uniforme: N_ANC_PTS pontos bastam para definir a forma
            # (o perfil é O(1)-largo em s) e mantêm a memória do L-BFGS sob controle,
            # já que ele avalia TODAS as âncoras de uma vez com create_graph.
            sqrt_mu = np.sqrt(mu_n)
            s_t = np.linspace(0.0, S_MAX, N_ANC_PTS)
            u_t = ka + s_t / sqrt_mu
            u_t = np.clip(u_t, ka, ka + params['u_max_rel'])
            phi_t = np.interp(u_t, u_n, phi_n, left=0.0, right=0.0)
            g_t = phi_t / (mu_n ** 0.25)   # alvo O(1) para qualquer μ
            anchors.append({
                'ka': torch.tensor([[ka]]).float().to(device),
                'delta': torch.tensor([[d]]).float().to(device),
                'u': torch.tensor(u_t).float().view(-1, 1).to(device),
                'phi': torch.tensor(phi_t).float().view(-1, 1).to(device),
                'g': torch.tensor(g_t).float().view(-1, 1).to(device),
                'amp': float(mu_n ** 0.25),
                'mu': torch.tensor([[mu_n]]).float().to(device),
            })
    print(f"  {len(anchors)} âncoras adsorvidas em {time.time()-t_anc:.1f}s")

    def sample_collocation(n_par=10, n_pts=150):
        """Amostra (δ, κa) e converte uma grade em s para pontos em u.

        Como o perfil é O(1)-largo em s para qualquer μ, amostrar uniformemente em
        s coloca pontos onde a solução de fato vive — no v31, amostrar em u_rel
        deixava o pico inteiro sem um único ponto de colocação quando μ era grande.
        """
        d_val = 10**(np.random.uniform(-2.0, 3.7, n_par))
        ka_val = 10**(np.random.uniform(-2.0, 1.0, n_par))
        d_u = torch.tensor(d_val).float().to(device).view(-1, 1)
        ka_u = torch.tensor(ka_val).float().to(device).view(-1, 1)

        with torch.no_grad():
            mu_u = model.mu_only(d_u, ka_u)
        sqrt_mu = _sqrt_mu_floor(mu_u)

        # metade concentrada perto do pico (s < 8), metade em todo o domínio
        half = n_pts // 2
        s_near = 8.0 * torch.rand(n_par, half, device=device)**1.5
        s_far = S_MAX * torch.rand(n_par, n_pts - half, device=device)
        s = torch.cat([s_near, s_far], dim=1)

        u_rel = s / sqrt_mu
        u_rel = torch.clamp(u_rel, max=params['u_max_rel'])
        u_b = (ka_u + u_rel).reshape(-1, 1)
        d_b = d_u.repeat(1, n_pts).reshape(-1, 1)
        ka_b = ka_u.repeat(1, n_pts).reshape(-1, 1)
        return u_b, d_b, ka_b, n_pts

    def compute_loss(u, d, ka, n_pts, sampled_anchors=None):
        u.requires_grad_(True)
        phi, mu = model(u, d, ka)

        # ---- 1. Resíduo da PDE: φ_uu − (μ + V)φ = 0 -----------------------
        phi_u = torch.autograd.grad(phi, u, torch.ones_like(phi), create_graph=True)[0]
        phi_uu = torch.autograd.grad(phi_u, u, torch.ones_like(phi_u), create_graph=True)[0]

        a = ka / params['kappa']
        pre_factor = (d / (params['kappa'] * a)) * (torch.exp(ka) / (1.0 + ka))
        u_safe = torch.clamp(u, min=1e-9)
        V_u = -pre_factor * (torch.exp(-u_safe) / u_safe)

        res_raw = phi_uu - (mu + V_u) * phi
        # Escala natural do resíduo: φ = μ^(1/4)·g ⟹ φ_uu = μ^(5/4)·g''.
        # Dividir por μ^(5/4) deixa o resíduo O(1) para qualquer μ, no lugar do
        # fator empírico 1/(1+log1p|V|) do v31.
        scale = torch.pow(mu.detach(), 1.25) + 1e-12
        residual = res_raw / scale
        l_pde = torch.mean(residual**2)

        # ---- 2. Normalização ∫φ² du = κ  (equivale a ∫g² ds = κ) ----------
        d_unique = d.view(-1, n_pts)[:, 0:1]
        ka_unique = ka.view(-1, n_pts)[:, 0:1]
        n_batch = d_unique.shape[0]

        with torch.no_grad():
            mu_u = model.mu_only(d_unique, ka_unique)
        sqrt_mu = _sqrt_mu_floor(mu_u)

        n_int = 500
        s_grid = torch.linspace(0, S_MAX, n_int, device=device).view(1, n_int)
        u_rel_int = torch.clamp(s_grid / sqrt_mu, max=params['u_max_rel'])
        ui_batch = ka_unique + u_rel_int
        di_batch = d_unique.expand(n_batch, n_int)
        kai_batch = ka_unique.expand(n_batch, n_int)

        phi_flat, _ = model(ui_batch.reshape(-1, 1), di_batch.reshape(-1, 1),
                            kai_batch.reshape(-1, 1))
        phi_batch = phi_flat.view(n_batch, n_int)

        phi_sq = phi_batch**2
        du = ui_batch[:, 1:] - ui_batch[:, :-1]
        phi_mid = (phi_sq[:, 1:] + phi_sq[:, :-1]) / 2.0
        integrals = torch.sum(phi_mid * du, dim=1)

        ka_np = ka_unique.detach().cpu().numpy().flatten()
        d_np = d_unique.detach().cpu().numpy().flatten()
        dc_np = get_wkb_delta_c(ka_np)
        targets = torch.tensor(np.where(d_np > dc_np, params['kappa'], 0.0)).float().to(device)
        l_norm = torch.mean((integrals - targets)**2)

        # ---- 3. Região não-adsorvida --------------------------------------
        dc_col = torch.tensor(get_wkb_delta_c(ka.detach().cpu().numpy())).float().to(device)
        is_ads = (d > dc_col).float()
        l_unad_phi = torch.mean(((1.0 - is_ads) * phi)**2)
        l_unad_mu = torch.mean(((1.0 - is_ads) * (torch.log10(mu + 1e-8) + 6.0))**2)
        l_unad = 10000.0 * l_unad_phi + 100.0 * l_unad_mu

        # ---- 4. Âncoras, comparadas em g = φ/μ^(1/4) -----------------------
        # No v31 a comparação era em φ, cuja amplitude varia 2 décadas; amostras de
        # μ grande dominavam ou zeravam a loss. Em g todas pesam igual.
        l_anchor = 0.0
        target_anchors = sampled_anchors if sampled_anchors is not None else anchors
        for anc in target_anchors:
            p_a, m_a = model(anc['u'], anc['delta'], anc['ka'])
            g_a = p_a / anc['amp']
            l_anchor = l_anchor + 50000.0 * F.mse_loss(g_a, anc['g'])
            g_max = torch.clamp(anc['g'].max(), min=1e-6)
            l_anchor = l_anchor + 10000.0 * F.mse_loss(g_a / g_max, anc['g'] / g_max)
            l_anchor = l_anchor + 200.0 * F.mse_loss(torch.log10(m_a[:1] + 1e-7),
                                                     torch.log10(anc['mu'] + 1e-7))

        return l_pde, 20000.0 * l_norm, l_anchor / len(target_anchors), l_unad

    # ---- Estágio 1: Adam ---------------------------------------------------
    print(f"Estágio 1: Adam ({args.pretrain} épocas)...")
    opt_adam = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt_adam, T_0=10000, T_mult=2, eta_min=1e-5)

    t0 = time.time()
    for epoch in range(args.pretrain):
        if mu_frozen and epoch == args.freeze_mu:
            for p in model.mu_net.parameters():
                p.requires_grad_(True)
            opt_adam.add_param_group({'params': list(model.mu_net.parameters())})
            mu_frozen = False
            print(f"E {epoch:5} | mu_net descongelada para o ajuste fino conjunto.")

        opt_adam.zero_grad()
        u_b, d_b, ka_b, n_pts = sample_collocation()
        sampled = np.random.choice(anchors, min(8, len(anchors)), replace=False)
        lp, ln, la, lu = compute_loss(u_b, d_b, ka_b, n_pts, sampled_anchors=sampled)
        loss = lp + ln + la + lu

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt_adam.step()
        scheduler.step()

        if epoch % 1000 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (epoch + 1) * (args.pretrain - epoch - 1)
            print(f"E {epoch:5} | Loss: {loss.item():.4f} | PDE: {lp.item():.2e} | "
                  f"Norm: {ln.item():.2e} | Anc: {la.item():.2e} | Unad: {lu.item():.2e} | "
                  f"lr: {opt_adam.param_groups[0]['lr']:.2e} | ETA: {eta/60:.0f}min")
            history['epoch'].append(epoch)
            history['total'].append(loss.item())
            history['pde'].append(lp.item())
            history['norm'].append(ln.item())
            history['anchor'].append(la.item())
            history['unad'].append(lu.item())
            if epoch % 20000 == 0 and epoch > 0:
                torch.save(model.state_dict(), f"unified_pinn_v32_checkpoint_{epoch}.pt")

    # ---- Estágio 2: L-BFGS -------------------------------------------------
    print(f"Estágio 2: L-BFGS em {args.lbfgs_rounds} rondas...")
    for round_idx in range(args.lbfgs_rounds):
        u_b, d_b, ka_b, n_pts = sample_collocation()
        opt_lbfgs = torch.optim.LBFGS(model.parameters(), lr=0.005, max_iter=20,
                                      line_search_fn='strong_wolfe', history_size=20)

        def closure():
            opt_lbfgs.zero_grad()
            lp, ln, la, lu = compute_loss(u_b, d_b, ka_b, n_pts, sampled_anchors=anchors)
            l = lp + ln + la + lu
            l.backward()
            return l

        l_val = opt_lbfgs.step(closure)
        if round_idx % 5 == 0:
            print(f"L-BFGS {round_idx:3}/{args.lbfgs_rounds} | Loss: {l_val.item():.4f}")
            history['epoch'].append(args.pretrain + round_idx * 20)
            history['total'].append(l_val.item())
        if torch.isnan(l_val):
            print("NaN detectado, parando L-BFGS.")
            break

    torch.save(model.state_dict(), "unified_pinn_v32_model.pt")
    with open("training_history_v32.json", "w") as f:
        json.dump(history, f)
    return model, history


# ---------------------------------------------------------------------------
# 6. Gráficos e validação
# ---------------------------------------------------------------------------
def generate_all_plots(model, history):
    model.eval()

    if len(history.get('epoch', [])) > 0:
        plt.figure(figsize=(10, 6))
        plt.semilogy(history['epoch'], history['total'], label='Total', lw=1.5)
        if len(history.get('pde', [])) > 0:
            n = min(len(history['epoch']), len(history['pde']))
            for k in ['pde', 'norm', 'anchor']:
                plt.semilogy(history['epoch'][:n], history[k][:n], alpha=0.5, label=k.upper())
        plt.title('Training Convergence History (v32)')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(alpha=0.2)
        plt.savefig('training_loss_v32.png', dpi=300); plt.close()

    # 6.1 Perfis de densidade
    print("Gerando perfis de densidade...")
    for ka_val in [0.1, 1.0, 5.0]:
        plt.figure(figsize=(10, 6))
        a = ka_val / params['kappa']
        for d in [30, 100, 300, 1000, 3000]:
            u_n, p_n, mu_n = solve_edwards_numerical(d, ka_val)
            if mu_n <= 1e-6:
                continue
            dist = (u_n - ka_val) / params['kappa']
            u_t = torch.tensor(u_n).float().view(-1, 1).to(device)
            with torch.no_grad():
                p_p, mu_p = model(u_t, torch.tensor([[float(d)]]).float().to(device),
                                  torch.tensor([[ka_val]]).float().to(device))
            plt.plot(dist, p_n**2, ':', color='gray', alpha=0.5, lw=1.5)
            plt.plot(dist, p_p.cpu().numpy().flatten()**2,
                     label=f'$\\delta$={d} (μ={mu_p[0].item():.3g})')
        plt.title(f'Density Profiles | $\\kappa a$={ka_val} (a={a:.0f}Å)  —  v32')
        plt.xlabel('Distance from surface $r-a$ (Å)'); plt.ylabel('P(r)')
        plt.xscale('symlog', linthresh=1e-3)
        plt.legend(fontsize=9); plt.grid(alpha=0.15); plt.tight_layout()
        plt.savefig(f'hf_pinn_ka_{ka_val}_v32.png', dpi=300); plt.close()

    # 6.2 Mapa de fase + δ_c
    print("Gerando mapa de fase...")
    ka_range = np.logspace(-2, 1, 50)
    delta_range = np.logspace(-4, 5, 60)
    KA, DELTA = np.meshgrid(ka_range, delta_range)
    MU = np.zeros_like(KA)
    with torch.no_grad():
        for i in range(len(delta_range)):
            d_t = torch.tensor(DELTA[i, :]).float().view(-1, 1).to(device)
            k_t = torch.tensor(KA[i, :]).float().view(-1, 1).to(device)
            MU[i, :] = model.mu_only(d_t, k_t).cpu().numpy().flatten()

    plt.figure(figsize=(10, 8))
    norm = SymLogNorm(linthresh=1.0, linscale=0.5, vmin=0, vmax=np.max(MU))
    cp = plt.pcolormesh(KA, DELTA, MU, cmap='viridis', norm=norm, shading='gouraud')
    plt.colorbar(cp, label='$\\mu$')

    dc_pinn = []
    for k in ka_range:
        k_t = torch.tensor([[k]]).float().to(device)
        lo, hi = 1e-4, 1e6
        for _ in range(30):
            m = np.sqrt(lo * hi)
            with torch.no_grad():
                mv = model.mu_only(torch.tensor([[m]]).float().to(device), k_t)
            if mv.item() > 1e-4: hi = m
            else: lo = m
        dc_pinn.append(m)
    plt.plot(ka_range, dc_pinn, 'w--', lw=2.5, label='PINN $\\delta_c$')
    dc_wkb = [get_wkb_delta_c(k) for k in ka_range]
    plt.plot(ka_range, dc_wkb, 'r:', lw=2, label='Theory (WKB)')
    plt.xscale('log'); plt.yscale('log')
    plt.xlim(1e-2, 1e1); plt.ylim(1e-4, 1e5)
    plt.title('Phase Map: Adsorption Transition (v32)')
    plt.xlabel('$\\kappa a$'); plt.ylabel('$\\delta$'); plt.legend()
    plt.tight_layout(); plt.savefig('phase_map_v32.png', dpi=300); plt.close()

    plt.figure(figsize=(8, 6))
    plt.loglog(ka_range, dc_pinn, 'ro-', markersize=3, label='PINN $\\delta_c$')
    plt.loglog(ka_range, dc_wkb, 'k--', alpha=0.6, label='Theoretical Trend (WKB)')
    plt.xlim(1e-2, 1e1); plt.ylim(1e-4, 1e5)
    plt.title('Critical Adsorption $\\delta_c$ vs $\\kappa a$ (v32)')
    plt.xlabel('$\\kappa a$'); plt.ylabel('$\\delta_c$')
    plt.grid(True, which='both', alpha=0.15); plt.legend()
    plt.tight_layout(); plt.savefig('critical_adsorption_v32.png', dpi=300); plt.close()

    # 6.3 Validação quantitativa
    validate(model)


def validate(model, n_ka=13, n_d=13):
    """Mesmo protocolo usado para auditar o v31, para comparação direta."""
    print("\nValidando v32 contra o solver numérico...")
    model.eval()
    rec = []
    for ka in np.logspace(-2, 1, n_ka):
        for d in np.logspace(-2, 4, n_d):
            u_n, p_n, mu_n = solve_edwards_numerical(d, ka)
            with torch.no_grad():
                u_t = torch.tensor(u_n).float().view(-1, 1).to(device)
                p_p, mu_p = model(u_t, torch.tensor([[float(d)]]).float().to(device),
                                  torch.tensor([[ka]]).float().to(device))
            p_p = p_p.cpu().numpy().flatten()
            err_phi = np.nan
            if mu_n > 1e-4:
                err_phi = np.sqrt(np.sum((p_p - p_n)**2) / np.sum(p_n**2))
            # mu_p vem broadcast para o batch de u; todos os elementos são iguais
            rec.append((ka, d, mu_n, mu_p[0].item(), err_phi))

    A = np.array(rec)
    ads = A[:, 2] > 1e-4
    mn, mp = A[ads, 2], A[ads, 3]
    rel = np.abs(mp - mn) / mn
    ss = np.sum((np.log10(mp) - np.log10(mn))**2)
    st = np.sum((np.log10(mn) - np.log10(mn).mean())**2)
    print(f"  mu : erro relativo mediana {np.median(rel)*100:.2f}% | p90 {np.percentile(rel,90)*100:.2f}% "
          f"| R2(log10) {1-ss/st:.5f}")
    nad = ~ads
    if nad.sum():
        print(f"  nao-adsorvido: falsos positivos (mu>1e-3) = {(A[nad,3]>1e-3).sum()}/{nad.sum()}")

    ephi = A[ads, 4]
    print(f"  phi: erro L2 mediana {np.median(ephi)*100:.1f}% | p90 {np.percentile(ephi,90)*100:.1f}%")
    print("  phi por faixa de mu (era 100% no v31 acima de 1e4):")
    for lo, hi in [(1e-4, 1), (1, 1e2), (1e2, 1e4), (1e4, 1e6), (1e6, 1e12)]:
        m = (mn >= lo) & (mn < hi)
        if m.sum():
            print(f"    mu in [{lo:7.0e},{hi:7.0e}): n={m.sum():3d} | erro L2 mediana {np.median(ephi[m])*100:6.1f}%")
    return A


if __name__ == "__main__":
    if os.path.exists("unified_pinn_v32_model.pt") and args.epochs < 10:
        model = UnifiedPINNv32(neurons=args.neurons, layers=args.layers).to(device)
        model.load_state_dict(torch.load("unified_pinn_v32_model.pt", map_location=device))
        print("Modelo v32 carregado.")
        generate_all_plots(model, {'epoch': [], 'total': []})
    else:
        model, history = train()
        generate_all_plots(model, history)
