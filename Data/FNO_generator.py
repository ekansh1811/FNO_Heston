import numpy as np
import pandas as pd
from scipy.stats import norm
import os

# Grid resolution
GRID_SIZE = 16
N_SAMPLES = 100  # You can increase for Colab
SAVE_DIR = "./fno_data"

os.makedirs(SAVE_DIR, exist_ok=True)

# ------------------------------
# Black-Scholes formula
# ------------------------------
def bs_call_price(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T) + 1e-9)
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# ------------------------------
# Heston Monte Carlo
# ------------------------------
def heston_call_mc(S0, K, T, r, v0, kappa, theta, sigma_v, rho, N=50, M=100):
    dt = T / N
    S = np.full(M, S0)
    v = np.full(M, v0)

    for _ in range(N):
        z1 = np.random.normal(size=M)
        z2 = rho * z1 + np.sqrt(1 - rho ** 2) * np.random.normal(size=M)

        v = np.maximum(v + kappa * (theta - v) * dt + sigma_v * np.sqrt(np.maximum(v, 0) * dt) * z2, 0)
        S *= np.exp((r - 0.5 * v) * dt + np.sqrt(v * dt) * z1)

    payoff = np.maximum(S - K, 0)
    return np.exp(-r * T) * np.mean(payoff)

# ------------------------------
# Grid for K and T
# ------------------------------
K_vals = np.linspace(80, 120, GRID_SIZE)
T_vals = np.linspace(0.1, 2.0, GRID_SIZE)
K_grid, T_grid = np.meshgrid(K_vals, T_vals)

def generate_input_grid(S, r, v0, kappa, theta, sigma_v, rho):
    grid = np.stack([
        np.full_like(K_grid, S),
        np.full_like(K_grid, r),
        np.full_like(K_grid, v0),
        np.full_like(K_grid, kappa),
        np.full_like(K_grid, theta),
        np.full_like(K_grid, sigma_v),
        np.full_like(K_grid, rho),
        K_grid,
        T_grid
    ], axis=-1)
    return grid

# ------------------------------
# Generate BS Data
# ------------------------------
bs_inputs = []
bs_outputs = []

for _ in range(N_SAMPLES):
    S = np.random.uniform(90, 110)
    r = np.random.uniform(0.01, 0.05)
    sigma = np.random.uniform(0.1, 0.4)

    input_grid = generate_input_grid(S, r, sigma**2, 0, 0, 0, 0)  # Simplified Heston param = 0
    call_prices = np.vectorize(bs_call_price)(S, K_grid, T_grid, r, sigma)
    call_prices = call_prices[..., np.newaxis]

    bs_inputs.append(input_grid)
    bs_outputs.append(call_prices)

np.save(f"{SAVE_DIR}/bs_FNO_in.npy", np.array(bs_inputs))
np.save(f"{SAVE_DIR}/bs_FNO_out.npy", np.array(bs_outputs))


# ------------------------------
# Generate Heston Data
# ------------------------------
heston_inputs = []
heston_outputs = []

for _ in range(N_SAMPLES):
    S = np.random.uniform(90, 110)
    r = np.random.uniform(0.01, 0.05)
    v0 = np.random.uniform(0.01, 0.2)
    kappa = np.random.uniform(1.0, 3.0)
    theta = np.random.uniform(0.01, 0.2)
    sigma_v = np.random.uniform(0.1, 0.4)
    rho = np.random.uniform(-0.8, -0.3)

    input_grid = generate_input_grid(S, r, v0, kappa, theta, sigma_v, rho)
    call_prices = np.zeros_like(K_grid)

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            K = K_grid[i, j]
            T = T_grid[i, j]
            call_prices[i, j] = heston_call_mc(S, K, T, r, v0, kappa, theta, sigma_v, rho)

    heston_inputs.append(input_grid)
    heston_outputs.append(call_prices[..., np.newaxis])

np.save(f"{SAVE_DIR}/heston_FNO_in.npy", np.array(heston_inputs))
np.save(f"{SAVE_DIR}/heston_FNO_out.npy", np.array(heston_outputs))