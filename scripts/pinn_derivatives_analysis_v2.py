import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pinn_loader import load_pinn, compute_gamma
from scipy.special import erfc
import time

VERSAO = 32   # troque para 31 para reproduzir a análise antiga

# As derivadas de μ funcionam igual nas duas versões: o .detach() do v32 está só no
# caminho que vai de μ para dentro de φ; o μ sai direto da mu_net e continua
# diferenciável em δ e κa. Conferido contra diferenças finitas: erro de 0.007% a 0.043%,
# incluindo a segunda derivada.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
try:
    model, params = load_pinn(VERSAO, neurons=256, layers=6, device=device)
except FileNotFoundError:
    print(f"Rode primeiro o unified_pinn_edwards_v{VERSAO}.py para gerar os pesos.")
    exit()

# =====================================================================
# 1. CÁLCULO DAS DERIVADAS VIA AUTOGRAD
# =====================================================================
print("Calculando derivadas de 1ª e 2ª ordem via autograd...")
n_deriv = 80
ka_d = np.logspace(-2, 1, n_deriv)
delta_d = np.logspace(-1, 4, n_deriv)
KA_D, DELTA_D = np.meshgrid(ka_d, delta_d)

ka_t = torch.tensor(KA_D.flatten()).float().view(-1, 1).to(device)
delta_t = torch.tensor(DELTA_D.flatten()).float().view(-1, 1).to(device)

ka_t.requires_grad_(True)
delta_t.requires_grad_(True)

u_fixed = torch.ones_like(ka_t) * 0.5
u_fixed = u_fixed.detach()

_, mu_pred = model(u_fixed, delta_t, ka_t)

# --- 1ª Derivada ---
gradientes = torch.autograd.grad(
    outputs=mu_pred, 
    inputs=[delta_t, ka_t], 
    grad_outputs=torch.ones_like(mu_pred),
    create_graph=True
)
dmu_ddelta_tensor = gradientes[0]
dmu_dka_tensor = gradientes[1]

# --- 2ª Derivada ---
d2mu_ddelta2_tensor = torch.autograd.grad(
    outputs=dmu_ddelta_tensor,
    inputs=delta_t,
    grad_outputs=torch.ones_like(dmu_ddelta_tensor),
    create_graph=False
)[0]

dmu_ddelta = dmu_ddelta_tensor.detach().cpu().numpy().reshape(n_deriv, n_deriv)
dmu_dka = dmu_dka_tensor.detach().cpu().numpy().reshape(n_deriv, n_deriv)
d2mu_ddelta2 = d2mu_ddelta2_tensor.detach().cpu().numpy().reshape(n_deriv, n_deriv)
mu_numpy = mu_pred.detach().cpu().numpy().reshape(n_deriv, n_deriv)
osmotic_pressure = -dmu_dka

# =====================================================================
# 2. CÁLCULO DA CAPACIDADE ADSORCIONAL (PARÂMETRO DE ORDEM: Gamma)
# =====================================================================
print("Calculando parâmetro de ordem (Gamma) via integração...")

# A grade fixa `linspace(0, 10, 100)` que estava aqui tem espaçamento 0.1, mas o pico
# do perfil tem largura ~1/√μ — 0.002 para μ = 2e5. Ela passava por cima do pico e a
# integral saía errada de forma errática (dependia de um ponto cair ou não sobre ele):
#
#   κa    δ       μ        grade fixa   grade em s   (correto = κ = 0.1)
#   0.10  30      1.2e3      0.06727      0.09989
#   0.10  300     1.9e4      0.00199      0.10006
#   0.10  3000    2.3e5      0.02181      0.09982
#   0.01  1000    7.8e6      0.00050      0.09985
#
# compute_gamma monta a grade em s = √μ·u_rel ∈ [0, S_MAX], que é onde a solução vive
# em qualquer regime. Vale para as duas versões do modelo.
gamma_map_flat = compute_gamma(model, delta_t.detach(), ka_t.detach(), n=2000)
gamma_map = gamma_map_flat.cpu().numpy().reshape(n_deriv, n_deriv)

# =====================================================================
# 3. GRÁFICOS (Grid 2x3)
# =====================================================================
print("Gerando gráficos...")
fig, axes = plt.subplots(2, 3, figsize=(24, 14))

# Linha teórica
ka_line = np.logspace(-2, 1, 200)
dc_line = []
for k in ka_line:
    C = 0.973
    num = 6 * k * (1 + k) * (C**2)
    den = 2 * np.pi * np.exp(k) * (erfc(np.sqrt(k/2))**2)
    dc_line.append(num / (den + 1e-30))

# --- Plot 1: Gamma (Excesso Superficial) ---
ax = axes[0, 0]
gamma_log = np.log10(gamma_map + 1e-10)
vmax_g = np.percentile(gamma_log, 98)
vmin_g = np.percentile(gamma_log, 2)
cp1 = ax.contourf(KA_D, DELTA_D, gamma_log, levels=50, cmap='inferno', vmin=vmin_g, vmax=vmax_g)
plt.colorbar(cp1, ax=ax, label='$\\log_{10}(\\Gamma)$')
ax.plot(ka_line, dc_line, 'w--', lw=2.0, label='$\\delta_c$ (WKB)')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_title('Excesso Superficial (Parâmetro de Ordem)\n$\\Gamma = \\int \\phi^2 du$', fontsize=13)
ax.set_xlabel('$\\kappa a$'); ax.set_ylabel('$\\delta$')
ax.legend(fontsize=10, loc='upper left')

# --- Plot 2: Suscetibilidade Termodinâmica ∂²μ/∂δ² ---
ax = axes[0, 1]
d2_log = np.sign(d2mu_ddelta2) * np.log10(np.abs(d2mu_ddelta2) + 1)
vmax3 = np.percentile(np.abs(d2_log), 98)
cp2 = ax.contourf(KA_D, DELTA_D, d2_log, levels=50, cmap='seismic', vmin=-vmax3, vmax=vmax3)
plt.colorbar(cp2, ax=ax, label='$\\mathrm{sgn} \\cdot \\log_{10}(|\\partial^2\\mu/\\partial\\delta^2| + 1)$')
ax.plot(ka_line, dc_line, 'k--', lw=2.0, label='$\\delta_c$ (WKB)')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_title('Suscetibilidade Termodinâmica\n$\\partial^2 \\mu / \\partial \\delta^2$', fontsize=13)
ax.set_xlabel('$\\kappa a$'); ax.set_ylabel('$\\delta$')
ax.legend(fontsize=10, loc='upper left')

# --- Plot 3: Pressão Osmótica -∂μ/∂(κa) ---
ax = axes[0, 2]
osm_log = np.sign(osmotic_pressure) * np.log10(np.abs(osmotic_pressure) + 1)
vmax2 = np.percentile(np.abs(osm_log), 98)
cp3 = ax.contourf(KA_D, DELTA_D, osm_log, levels=50, cmap='RdYlBu_r', vmin=-vmax2, vmax=vmax2)
plt.colorbar(cp3, ax=ax, label='$\\mathrm{sgn} \\cdot \\log_{10}(|\\Pi| + 1)$')
ax.plot(ka_line, dc_line, 'k--', lw=2.0, label='$\\delta_c$ (WKB)')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_title('Pressão Osmótica Efetiva\n$\\Pi = -\\partial \\mu / \\partial (\\kappa a)$', fontsize=13)
ax.set_xlabel('$\\kappa a$'); ax.set_ylabel('$\\delta$')
ax.legend(fontsize=10, loc='upper left')

# --- Plot 4: Gamma vs δ (Curvas do salto) ---
ax = axes[1, 0]
ka_curves = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(ka_curves)))
n_fine = 300
delta_fine = np.logspace(-1, 4, n_fine)

for idx, ka_val in enumerate(ka_curves):
    # Calculando Gamma
    d_line = torch.tensor(delta_fine).float().view(-1, 1).to(device)
    ka_line_t = torch.full((n_fine, 1), ka_val, dtype=torch.float32, device=device)
    
    # Mesma grade adaptada a μ usada no mapa 2D acima
    gamma_line = compute_gamma(model, d_line, ka_line_t, n=2000).cpu().numpy()


    ax.loglog(delta_fine, gamma_line + 1e-10, '-', color=colors[idx], lw=2.0, label=f'$\\kappa a$ = {ka_val}')
    
    C = 0.973
    num = 6 * ka_val * (1 + ka_val) * (C**2)
    den = 2 * np.pi * np.exp(ka_val) * (erfc(np.sqrt(ka_val/2))**2)
    dc_val = num / (den + 1e-30)
    if 0.1 <= dc_val <= 1e4:
        ax.axvline(dc_val, color=colors[idx], ls=':', alpha=0.5, lw=1.0)

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_ylim(1e-8, 10)
ax.set_xlabel('$\\delta$ (Parâmetro de Adsorção)', fontsize=12)
ax.set_ylabel('$\\Gamma$', fontsize=12)
ax.set_title('Transição de Fase: Excesso Superficial\n$\\Gamma$ vs $\\delta$', fontsize=13)
ax.legend(fontsize=8, ncol=2, loc='lower right')
ax.grid(True, alpha=0.15, which='both')

# --- Plot 5: ∂²μ/∂δ² vs δ (Curvas dos picos) ---
ax = axes[1, 1]
for idx, ka_val in enumerate(ka_curves):
    d_line = torch.tensor(delta_fine).float().view(-1, 1).to(device)
    ka_line_t = torch.full((n_fine, 1), ka_val, dtype=torch.float32, device=device)
    d_line.requires_grad_(True)
    ka_line_t.requires_grad_(True)
    
    u_fix = torch.ones(n_fine, 1, device=device) * 0.5
    u_fix = u_fix.detach()
    _, mu_line = model(u_fix, d_line, ka_line_t)
    
    dmu_dd_line = torch.autograd.grad(mu_line, d_line, torch.ones_like(mu_line), create_graph=True)[0]
    d2mu_dd2_line = torch.autograd.grad(dmu_dd_line, d_line, torch.ones_like(dmu_dd_line), create_graph=False)[0]
    
    d2_np = d2mu_dd2_line.detach().cpu().numpy().flatten()
    ax.semilogy(delta_fine, np.abs(d2_np) + 1e-10, '-', color=colors[idx], lw=2.0, label=f'$\\kappa a$ = {ka_val}')
    
    C = 0.973
    num = 6 * ka_val * (1 + ka_val) * (C**2)
    den = 2 * np.pi * np.exp(ka_val) * (erfc(np.sqrt(ka_val/2))**2)
    dc_val = num / (den + 1e-30)
    if 0.1 <= dc_val <= 1e4:
        ax.axvline(dc_val, color=colors[idx], ls=':', alpha=0.5, lw=1.0)
    
    peak_idx = np.argmax(np.abs(d2_np))
    ax.plot(delta_fine[peak_idx], np.abs(d2_np[peak_idx]) + 1e-10, 
            'v', color=colors[idx], markersize=6, markeredgecolor='k', markeredgewidth=0.5)

ax.set_xscale('log')
ax.set_xlabel('$\\delta$ (Parâmetro de Adsorção)', fontsize=12)
ax.set_ylabel('$|\\partial^2 \\mu / \\partial \\delta^2|$', fontsize=12)
ax.set_title('Transição de Fase: Picos de Suscetibilidade\n$|\\partial^2 \\mu / \\partial \\delta^2|$ vs $\\delta$', fontsize=13)
ax.legend(fontsize=8, ncol=2, loc='upper right')
ax.grid(True, alpha=0.15, which='both')
ax.annotate('$\\triangledown$ = pico PINN\nLinhas : = $\\delta_c$ WKB', 
            xy=(0.02, 0.95), xycoords='axes fraction', fontsize=9,
            va='top', ha='left', color='gray',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# --- Plot 6: Vazio ou 1ª Derivada ---
# Vamos plotar a 1ª Derivada ∂μ/∂δ no último painel, assim temos o conjunto completo
ax = axes[1, 2]
dmu_ddelta_log = np.sign(dmu_ddelta) * np.log10(np.abs(dmu_ddelta) + 1)
vmax1 = np.percentile(np.abs(dmu_ddelta_log), 98)
cp6 = ax.contourf(KA_D, DELTA_D, dmu_ddelta_log, levels=50, cmap='coolwarm', vmin=-vmax1, vmax=vmax1)
plt.colorbar(cp6, ax=ax, label='$\\mathrm{sgn} \\cdot \\log_{10}(|\\partial\\mu/\\partial\\delta| + 1)$')
ax.plot(ka_line, dc_line, 'k--', lw=2.0, label='$\\delta_c$ (WKB)')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_title('1ª Derivada: Sensibilidade\n$\\partial \\mu / \\partial \\delta$', fontsize=13)
ax.set_xlabel('$\\kappa a$'); ax.set_ylabel('$\\delta$')
ax.legend(fontsize=10, loc='upper left')

plt.tight_layout()
plt.savefig(f'analise_derivadas_completa_v{VERSAO}.png', dpi=300)
print(f"\nGráfico salvo: 'analise_derivadas_completa_v{VERSAO}.png'")

