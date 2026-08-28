import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm, kv

def modified_spherical_bessel_k(l, z):
    """
    Função de Bessel Esférica Modificada de 2ª Espécie: k_l(z)
    Relacionada à função de Bessel normal K por:
    k_l(z) = sqrt(pi / 2z) * K_{l+0.5}(z)
    """
    return np.sqrt(np.pi / (2.0 * z)) * kv(l + 0.5, z)

def generate_spherical_potential_slice(N=200, L=8.0, a=1.0, kappa=1.0, l_max=3):
    """
    Gera uma fatia 2D (plano x-z com y=0) de um campo de potencial 3D heterogêneo
    ao redor de uma ESFERA usando expansão de multipolos esféricos com Debye-Hückel.
    """
    x = np.linspace(-L/2, L/2, N)
    z = np.linspace(-L/2, L/2, N)
    X, Z = np.meshgrid(x, z)
    
    R = np.sqrt(X**2 + Z**2)
    
    # Ângulo polar (colatitude, medido do eixo Z, de 0 a pi)
    Polar = np.arccos(Z / (R + 1e-12))
    
    # Ângulo azimutal (longitude, no plano xy). Como y=0, é 0 (x positivo) ou pi (x negativo)
    Azimuthal = np.where(X >= 0, 0.0, np.pi)
    
    # Acumulador do potencial complexo
    V = np.zeros_like(R, dtype=np.complex128)
    
    mask = R >= a
    R_ext = R[mask]
    Polar_ext = Polar[mask]
    Azim_ext = Azimuthal[mask]
    
    V_ext = np.zeros_like(R_ext, dtype=np.complex128)
    
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            # Sorteamos um coeficiente complexo A_lm
            # O fator (l+1) garante que multipolos altíssimos (ruído fino) sejam menos intensos que monopolos/dipolos
            coeff = (np.random.randn() + 1j * np.random.randn()) / (l + 1)
            
            # Parte radial: k_l(kappa * r) normalizada na superfície (r=a)
            rad_val = modified_spherical_bessel_k(l, kappa * R_ext) / modified_spherical_bessel_k(l, kappa * a)
            
            # Parte angular: Harmônicos Esféricos Y_lm
            # A função sph_harm do SciPy recebe (ordem m, grau l, azimutal, polar)
            ang_val = sph_harm(m, l, Azim_ext, Polar_ext)
            
            V_ext += coeff * rad_val * ang_val
            
    V[mask] = V_ext
    
    # Como a física real exige um campo real, usamos a parte real da expansão 
    # (em uma implementação puramente rigorosa aplicaríamos A_{l,-m} = (-1)^m A_{l,m}^*, 
    # mas extrair a parte real funciona perfeitamente para gerar campos arbitrários).
    V_real = np.real(V)
    V_real[~mask] = np.nan  # Interior da esfera fica Nulo
    
    return X, Z, V_real

if __name__ == "__main__":
    print("Gerando amostras de potenciais heterogêneos esféricos (Fatia X-Z)...")
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    
    np.random.seed(42) # Semente fixa
    
    for i in range(4):
        # Esfera de raio a=1.0, simulada até l_max=4 (Monopolo a Hexadecapolo)
        X, Z, V = generate_spherical_potential_slice(N=200, L=8.0, a=1.0, kappa=1.5, l_max=4)
        
        ax = axes[i]
        
        limit = np.nanmax(np.abs(V)) * 0.8
        cp = ax.contourf(X, Z, V, levels=60, cmap='RdBu_r', vmin=-limit, vmax=limit)
        
        # Desenhar a esfera central (seção transversal)
        circle = plt.Circle((0, 0), 1.0, color='darkgray', zorder=10)
        ax.add_artist(circle)
        
        ax.set_aspect('equal')
        ax.set_title(f'Esfera Heterogênea {i+1}')
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        fig.colorbar(cp, ax=ax, fraction=0.046, pad=0.04, label='$V(x,z)$')

    plt.tight_layout()
    plt.savefig('potenciais_esfericos_amostras.png', dpi=300)
    print("Concluído! Abra a imagem 'potenciais_esfericos_amostras.png'.")
