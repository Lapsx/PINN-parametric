import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from generate_multipole_potentials import generate_spherical_potential_slice

def construct_hamiltonian_2d(V_matrix, dx):
    """
    Constrói o Hamiltoniano 2D em diferenças finitas: H = -∇² + V
    """
    N = V_matrix.shape[0]
    
    # Operador Laplaciano 1D: (-1, 2, -1) / dx²
    diag = np.full(N, 2.0 / dx**2)
    off_diag = np.full(N - 1, -1.0 / dx**2)
    D2_1d = sp.diags([off_diag, diag, off_diag], [-1, 0, 1])
    
    # Produto de Kronecker para criar o Laplaciano 2D: ∇² = I ⊗ D_xx + D_zz ⊗ I
    I_1d = sp.eye(N)
    Laplacian_2d = sp.kron(I_1d, D2_1d) + sp.kron(D2_1d, I_1d)
    
    # Tratamento do Potencial V(x, z)
    V_flat = V_matrix.copy().flatten()
    # O interior da esfera (onde V é NaN) recebe uma barreira de energia gigantesca
    # para forçar a probabilidade do polímero ir a zero ali dentro (Condição de Contorno de Dirichlet).
    V_flat[np.isnan(V_flat)] = 1e6 
    
    V_diag = sp.diags([V_flat], [0])
    
    # H = -∇² + V (Note que nossa matriz Laplaciana já atua como -∇² por causa do sinal)
    H = Laplacian_2d + V_diag
    return H

def solve_ground_state(X, Z, V, dx):
    H = construct_hamiltonian_2d(V, dx)
    
    # print("Diagonalizando o Hamiltoniano esparso 2D (isso pode levar alguns segundos)...")
    # Utilizamos eigsh (Eigenvalue Solver for Symmetric Hermitian matrices)
    # Buscamos o Menor Autovalor Algébrico ('SA')
    vals, vecs = eigsh(H, k=1, which='SA', tol=1e-5)
    
    mu_0 = vals[0]
    phi_0 = vecs[:, 0].reshape(V.shape)
    
    # O estado fundamental não tem "nós" (é todo positivo ou todo negativo).
    if np.sum(phi_0) < 0:
        phi_0 = -phi_0
        
    # Normalização da densidade de probabilidade (Integral de phi^2 = 1)
    norm = np.sum(phi_0**2) * (dx**2)
    phi_0 = phi_0 / np.sqrt(norm)
    
    return phi_0, mu_0

if __name__ == "__main__":
    print("Iniciando Etapa B: Solver Numérico...")
    
    # Reduzimos o N de 200 para 100 para a matriz 10.000 x 10.000 não estourar a RAM
    # e resolver rapidamente.
    N = 100 
    L = 6.0
    dx = L / (N - 1)
    
    np.random.seed(45) # Seed fixa para visualizar uma mancha bem interessante
    
    # Importando da Etapa A
    X, Z, V = generate_spherical_potential_slice(N=N, L=L, a=1.0, kappa=1.5, l_max=4)
    
    # Resolvemos a EDP
    phi, mu = solve_ground_state(X, Z, V, dx)
    density = phi**2
    
    # Ocultar a densidade de dentro da esfera para o gráfico ficar bonito
    density[np.isnan(V)] = np.nan
    
    # Visualização
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    # Plot 1: Potencial
    limit = np.nanmax(np.abs(V)) * 0.8
    cp1 = ax1.contourf(X, Z, V, levels=50, cmap='RdBu_r', vmin=-limit, vmax=limit)
    circle1 = plt.Circle((0, 0), 1.0, color='darkgray', zorder=10)
    ax1.add_artist(circle1)
    ax1.set_aspect('equal')
    ax1.set_title('Potencial de Superfície Heterogêneo\n$V(x,z)$')
    fig.colorbar(cp1, ax=ax1, label='Energia $V$')
    
    # Plot 2: Densidade do Polímero
    cp2 = ax2.contourf(X, Z, density, levels=50, cmap='inferno')
    circle2 = plt.Circle((0, 0), 1.0, color='darkgray', zorder=10)
    ax2.add_artist(circle2)
    ax2.set_aspect('equal')
    ax2.set_title(f'Densidade Acumulada do Polímero\n$\\phi^2(x,z)$ (Autovalor $\\mu$ = {mu:.2f})')
    fig.colorbar(cp2, ax=ax2, label='Probabilidade')
    
    plt.tight_layout()
    plt.savefig('edwards_2d_solution.png', dpi=300)
    print("Solução extraída! Imagem salva como 'edwards_2d_solution.png'")
