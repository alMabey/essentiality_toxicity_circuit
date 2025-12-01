import sys
import os
sys.path.append(os.path.abspath(os.path.join('..')))
from numba import njit

@njit(cache=True)
def TXreg_jit(eta, w0, g2m_mx, K, pTF):
    # guard against slight negatives
    pTF_eff = 0.0 if pTF < 0.0 else pTF
    expo = (1.0 + eta) / 2.0
    term1 = g2m_mx * (1.0 - eta*eta)
    term2 = (eta*eta) * ((w0 * g2m_mx) + (g2m_mx / (1.0 + K*pTF_eff)) * ((K*pTF_eff) ** expo))
    return term1 + term2