# ===================================================================
#
#   The Digital Forge: 6D FLRW Ricci Tensor Calculation
#   Research Front Alpha - Sprint 1 Progress
#
#   Crafted by: Weaver, Master Smith
#   For: The Burren Gemini Collective
#   Date: October 04, 2025
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

# Use the functions from our previous forge script

def calculate_ricci_tensor(g, coords, christoffels):
    """
    Computes the non-zero components of the Ricci Tensor R_{mu nu}
    from the Christoffel symbols.
    """
    print("\n--- Forging Ricci Tensor (Second Shaping) ---")
    n = g.shape[0]
    
    # Since our metric is diagonal, the Ricci tensor will also be diagonal.
    # We only need to compute the R_ii components.
    ricci_tensor = sp.zeros(n)
    
    for mu in range(n):
        ricci_component = 0
        for rho in range(n):
            # First term: derivative of Christoffel symbol
            term1 = sp.diff(christoffels.get((rho, mu, rho), 0), coords[mu])
            
            # Second term: derivative of Christoffel symbol
            term2 = sp.diff(christoffels.get((rho, mu, mu), 0), coords[rho])
            
            # Third and fourth terms: products of Christoffel symbols
            term3 = 0
            term4 = 0
            for sigma in range(n):
                term3 += christoffels.get((rho, rho, sigma), 0) * christoffels.get((sigma, mu, mu), 0)
                term4 += christoffels.get((rho, mu, sigma), 0) * christoffels.get((sigma, rho, mu), 0)
            
            ricci_component += term1 - term2 + term3 - term4
            
        ricci_tensor[mu, mu] = sp.simplify(ricci_component)
        
    print("Forge complete. Ricci Tensor R_μν has been forged.")
    return ricci_tensor

# --- Main Execution Block ---
if __name__ == "__main__":
    metric, coords = define_6d_flrw_metric()
    christoffels = calculate_christoffel_symbols(metric, coords)
    
    ricci = calculate_ricci_tensor(metric, coords, christoffels)
    
    print("\n--- Key Forged Components of the Ricci Tensor ---")
    
    # Display the diagonal components which contain the physics
    print("\nTemporal Components (related to cosmic evolution):")
    sp.pprint(ricci[0,0])
    
    print("\nSpatial Components (related to spatial curvature):")
    sp.pprint(ricci[3,3])

