"""
Carregador comum dos modelos PINN de Edwards + utilidades de integração
=======================================================================
Centraliza as três coisas que mudam entre o v31 e o v32 (módulo, classe e arquivo
de pesos), para os scripts de análise não repetirem a troca, e oferece a grade
adaptada a μ que a integração de Γ passou a exigir.

Uso nos scripts de análise:

    from pinn_loader import load_pinn, solve_edwards_numerical, compute_gamma
    model, params = load_pinn(32)          # ou load_pinn(31)

As assinaturas do v31 e do v32 são idênticas — construtor `(neurons, layers)` e
`forward(u, d, ka) -> (phi, mu)` — então trocar a versão não muda nenhum ponto de
chamada.

Por que o default é 32: medido em 169 pontos, mesmo protocolo, contra o estado
fundamental exato do hamiltoniano discretizado (LAPACK `eigh_tridiagonal`):

    métrica                     v31       v32
    μ  erro relativo mediano    4.07%     0.61%
    μ  R² em log10              0.98802   0.99979
    φ  erro L2 mediano          18.0%     1.5%
    φ  para μ ∈ [1e4, 1e6]      100%      0.7%

Ver PINN_v32_vs_v31.md para a metodologia.
"""

import numpy as np
import torch

# O solver numérico do v32 usa domínio adaptado a μ e eigh_tridiagonal (exato para
# matriz tridiagonal). O do v31 usava domínio fixo [κa, κa+18] e eigsh iterativo, que
# para μ grande resolvia o pico com 2 ou 3 pontos. Sempre importar daqui.
from unified_pinn_edwards_v32 import solve_edwards_numerical, S_MAX, params  # noqa: F401


def load_pinn(version=32, neurons=256, layers=6, device=None, weights=None):
    """Instancia e carrega o PINN da versão pedida.

    Returns:
        (model em modo eval, dict de parâmetros físicos)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if version == 32:
        from unified_pinn_edwards_v32 import UnifiedPINNv32 as Net, params as p
        default_weights = "unified_pinn_v32_model.pt"
    elif version == 31:
        from unified_pinn_edwards_v31 import UnifiedPINNv31 as Net, params as p
        default_weights = "unified_pinn_v31_model.pt"
    else:
        raise ValueError(f"versão {version} desconhecida (use 31 ou 32)")

    model = Net(neurons=neurons, layers=layers).to(device)
    model.load_state_dict(torch.load(weights or default_weights, map_location=device))
    model.eval()
    print(f"[+] PINN v{version} carregado de {weights or default_weights}")
    return model, p


def mu_of(model, d_t, ka_t):
    """μ(δ, κa) sem avaliar a rede de φ, para qualquer versão."""
    with torch.no_grad():
        if hasattr(model, 'mu_only'):            # v32
            return model.mu_only(d_t, ka_t)
        return model(ka_t + 0.1, d_t, ka_t)[1]   # v31


def u_grid_adaptada(model, d_t, ka_t, n=2000):
    """Grade em u que resolve o pico para qualquer μ.

    O perfil tem largura ~1/√μ. Uma grade fixa como `linspace(0, 10, 100)`
    (espaçamento 0.1) passa por cima do pico quando μ ≳ 1e3 e a integral de φ² sai
    entre 33% e 99% baixa, de forma errática — depende de um ponto da grade cair ou
    não sobre o pico. Aqui a grade cobre s = √μ·u_rel ∈ [0, S_MAX], que é onde a
    solução vive em qualquer regime.

    Args:
        model: PINN v31 ou v32
        d_t, ka_t: (B, 1) tensores de δ e κa
        n: pontos por amostra

    Returns:
        u: (B, n) grade em u        du: (B, 1) espaçamento por amostra
    """
    mu = mu_of(model, d_t, ka_t)
    # Piso em S_MAX/u_max_rel: para μ minúsculo (região não-adsorvida) a grade em s
    # estouraria o domínio físico u_rel ≤ u_max_rel.
    sqrt_mu = torch.clamp(torch.sqrt(torch.clamp(mu, min=1e-12)),
                          min=S_MAX / params['u_max_rel'])
    s = torch.linspace(0.0, S_MAX, n, device=d_t.device).view(1, n)
    u_rel = s / sqrt_mu
    u = ka_t + u_rel
    du = (u[:, 1:2] - u[:, 0:1])
    return u, du


def compute_gamma(model, d_t, ka_t, n=2000, pontos_por_lote=200_000):
    """Γ = ∫φ² du, o parâmetro de ordem da adsorção.

    Na região adsorvida o valor correto é κ = 0.1, por construção da normalização
    imposta no treino. Serve como teste de sanidade: se Γ não der ~0.1 para δ bem
    acima de δ_c, ou o modelo ou a grade está errada.

    Args:
        d_t, ka_t: (B, 1) tensores
        n: pontos de integração por amostra
        pontos_por_lote: teto de pontos avaliados de uma vez (amostras × n).
            O lote é derivado daqui em vez de fixo porque n subiu de 100 para 2000
            ao trocar a grade fixa pela adaptada — um lote fixo de 500 amostras
            passaria a pedir 1 milhão de pontos de uma vez e estoura a VRAM.
    Returns:
        (B,) tensor com Γ
    """
    out = torch.zeros(d_t.shape[0], device=d_t.device)
    batch_size = max(1, pontos_por_lote // n)
    for i in range(0, d_t.shape[0], batch_size):
        d_b = d_t[i:i + batch_size]
        ka_b = ka_t[i:i + batch_size]
        B = d_b.shape[0]
        u, du = u_grid_adaptada(model, d_b, ka_b, n=n)
        with torch.no_grad():
            phi, _ = model(u.reshape(-1, 1),
                           d_b.repeat_interleave(n).view(-1, 1),
                           ka_b.repeat_interleave(n).view(-1, 1))
        out[i:i + B] = (phi.view(B, n) ** 2 * du).sum(dim=1)
    return out
