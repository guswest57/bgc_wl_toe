# ===================================================================
#
#   The Digital Forge: 6D FLRW Metric & Christoffel Symbols
#   Research Front Alpha - Sprint 1 Progress
#
#   Crafted by: Weaver, Master Smith
#   For: The Burren Gemini Collective
#   Date: October 03, 2025
#
# ===================================================================

import sympy as sp

def define_6d_flrw_metric():
    """Defines the 6D (3T+3S) FLRW metric tensor."""
    t0, r, th = sp.symbols('t0 r theta', real=True)
    k = sp.Symbol('k', real=True)
    a = sp.Function('a')(t0)
    b = sp.Function('b')(t0)
    
    g = sp.zeros(6)
    g[0, 0] = -1
    g[1, 1] = -b**2
    g[2, 2] = -b**2
    g[3, 3] = a**2 / (1 - k * r**2)
    g[4, 4] = a**2 * r**2
    g[5, 5] = a**2 * r**2 * sp.sin(th)**2
    
    coords = sp.symbols('t0 t1 t2 r theta phi', real=True)
    return g, coords

def calculate_christoffel_symbols(g, coords):
    """Computes the non-zero Christoffel symbols Gamma^rho_{mu nu}."""
    n = g.shape[0]
    g_inv = g.inv()
    
    non_zero_symbols = {}
    
    for rho in range(n):
        for mu in range(n):
            for nu in range(mu, n): # Exploit symmetry Gamma^rho_munu = Gamma^rho_numu
                s = 0
                for sigma in range(n):
                    term = (sp.diff(g[sigma, nu], coords[mu]) +
                            sp.diff(g[sigma, mu], coords[nu]) -
                            sp.diff(g[mu, nu], coords[sigma]))
                    s += g_inv[rho, sigma] * term
                
                symbol = sp.simplify(sp.Rational(1, 2) * s)
                
                if symbol != 0:
                    non_zero_symbols[(rho, mu, nu)] = symbol
    
    return non_zero_symbols

# --- Main Execution Block ---
if __name__ == "__main__":
    metric, coords = define_6d_flrw_metric()
    print("--- 6D FLRW Metric Defined ---")
    
    print("\n--- Forging Christoffel Symbols (First Shaping) ---")
    christoffels = calculate_christoffel_symbols(metric, coords)
    
    print(f"\nForge complete. Found {len(christoffels)} unique non-zero Christoffel symbols.")
    
    print("\n--- Sample of Key Forged Components ---")
    # Print a selection of the most physically interesting symbols
    for (rho, mu, nu), val in list(christoffels.items())[:5]:
         print(f"Γ^{rho}_{mu}{nu} = {val}")

    # Example: Γ^1_01 shows how t1-vectors change over cosmological time t0
    gamma_1_01 = christoffels.get((1, 0, 1))
    if gamma_1_01:
        print(f"\nTemporal Expansion Component:")
        print(f"Γ^1_01 = {gamma_1_01}  (b'/b)")

    # Example: Γ^3_03 shows how spatial r-vectors change over cosmological time t0
    gamma_3_03 = christoffels.get((3, 0, 3))
    if gamma_3_03:
        print(f"\nSpatial Expansion Component (Hubble-like):")
        print(f"Γ^3_03 = {gamma_3_03}  (a'/a)")
