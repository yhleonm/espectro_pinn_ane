"""
Physics-Informed Neural Network (PINN) Solver — Arquitectura Residual.

La red neuronal NO predice el campo eléctrico E directamente.
En su lugar, predice un FACTOR DE CORRECCIÓN TOPOGRÁFICA que se multiplica
por el valor analítico FSPL (Free Space Path Loss).

    Señal_final = FSPL_analítico(d) × PINN_corrección(x, y, z)

Donde:
    - correction ≈ 1.0 → espacio libre sin obstáculos
    - correction ≈ 0.0 → montaña bloqueando completamente
    - correction > 1.0  → efecto guía de onda / canalización
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class FourierEmbedding(nn.Module):
    def __init__(self, input_dim=2, embed_dim=128, scale=10.0):
        super().__init__()
        B = torch.randn(input_dim, embed_dim // 2) * scale
        self.register_buffer('B', B)
    
    def forward(self, x):
        proj = x @ self.B  # (N, embed_dim//2)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

class SmoothMLPLayer(nn.Module):
    """
    Capa estándar de Red Neuronal (Multilayer Perceptron) con activación SiLU (Swish).
    A diferencia de SIREN (Seno), SiLU garantiza una salida suave y sin oscilaciones (ruido Moiré).
    """
    def __init__(self, in_features: int, out_features: int, is_first: bool = False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.SiLU()
        
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.linear(x))


class HelmholtzPINN(nn.Module):
    """
    Arquitectura Residual de la PINN para corrección topográfica.
    
    Entrada: (x, y, z) normalizados
    Salida: Factor de corrección topográfica (escalar > 0 vía Softplus)
    """
    def __init__(self, in_features: int = 3, hidden_features: int = 128, 
                 hidden_layers: int = 4, out_features: int = 1):
        super().__init__()
        
        print("    [PINN Model] Starting Residual PINN init...")
        self.fourier = FourierEmbedding(input_dim=in_features, embed_dim=hidden_features, scale=10.0)
        
        self.net = nn.Sequential()
        
        # Capa de entrada (recibe los features de Fourier, que tienen tamaño hidden_features)
        self.net.append(SmoothMLPLayer(hidden_features, hidden_features, is_first=True))
        
        # Capas ocultas
        for _ in range(hidden_layers):
            self.net.append(SmoothMLPLayer(hidden_features, hidden_features, is_first=False))
            
        print("    [PINN Model] Hidden layers appended")
        # Capa de salida lineal
        final_linear = nn.Linear(hidden_features, out_features)
        nn.init.xavier_uniform_(final_linear.weight)
        # Bias inicial positivo para que Softplus(output) empiece cerca de 1.0
        nn.init.constant_(final_linear.bias, 0.5)
        self.net.append(final_linear)
        print("    [PINN Model] Residual Init complete")

    def forward(self, coords_2d: torch.Tensor, scale_km: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Entrada: coords (N, 2) normalizadas [-1, 1]
        Salida: Fase T (N, 1) en escala FÍSICA (kilómetros).
        T(x) = dist_km * (1.0 + tau(x))
        """
        coords_2d = coords_2d.clone().requires_grad_(True) 
        fourier_features = self.fourier(coords_2d)
        raw_output = self.net(fourier_features)
        
        # tau: Exceso de lentitud relativo.
        tau = F.softplus(raw_output)
        
        # Distancia analítica en KILÓMETROS
        dist_norm = torch.norm(coords_2d, dim=1, keepdim=True) + 1e-6
        dist_km = dist_norm * scale_km
        
        # T total en km. Si tau=0 (vacío), T = dist_km.
        T = dist_km * (1.0 + tau)
        
        return T, coords_2d


def eikonal_pde_loss_phase(T: torch.Tensor, coords_2d: torch.Tensor, 
                           srtm_slowness: torch.Tensor, scale_xy: float = 1.0) -> torch.Tensor:
    """
    Eikonal sobre tiempo de viaje T (en km).
    Física: |∇T|² = s² (donde s ≈ 1.0 km/km)
    """
    grad = torch.autograd.grad(
        T, coords_2d,
        grad_outputs=torch.ones_like(T),
        create_graph=True, retain_graph=True
    )[0]

    # Convertir gradiente de espacio normalizado a espacio físico (km^-1)
    dT_dx = grad[:, 0:1] / scale_xy
    dT_dy = grad[:, 1:2] / scale_xy

    grad_mag_sq = dT_dx**2 + dT_dy**2

    # srtm_slowness ahora debe ser ≈ 1.0
    eikonal_residual = (grad_mag_sq - srtm_slowness**2)**2

    return torch.mean(eikonal_residual)

def phase_to_amplitude(T: torch.Tensor, coords_2d: torch.Tensor, scale_km: float = 1.0) -> torch.Tensor:
    """
    Convierte la fase T (km) en amplitud de corrección C.
    C = exp(-(T_phys - dist_phys)) = exp(-exceso_distancia_km)
    """
    dist_km = torch.norm(coords_2d, dim=1, keepdim=True) * scale_km
    excess_km = T - dist_km
    # Un exceso de 10km de "retraso" equivale a exp(-10) ≈ -86 dB de atenuación.
    return torch.exp(-torch.clamp(excess_km, min=0.0, max=20.0))
