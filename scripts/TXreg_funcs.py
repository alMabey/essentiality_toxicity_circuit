import sys
import os
sys.path.append(os.path.abspath(os.path.join('..')))
from numba import njit

@njit(cache=True)
def TXreg_template(eta, w0, g2m_mx, K, pTF):
    # guard against slight negatives
    pTF_eff = 0.0 if pTF < 0.0 else pTF
    expo = (1.0 + eta) / 2.0
    term1 = g2m_mx*(1.0 - eta*eta)
    term2 = (eta*eta) * ((w0 * g2m_mx) + (g2m_mx / (1.0 + K*pTF_eff)) * ((K*pTF_eff) ** expo))
    return term1 + term2


@njit(cache=True)
def TXreg_mannan2025(eta, w0, g2m_mx, K, TF_P):
    # guard against slight negatives
    TF_P_eff = 0.0 if TF_P < 0.0 else TF_P
    expo = (1.0 + eta) / 2.0
    term1 = g2m_mx*(1.0 - eta*eta)
    term2 = (eta*eta) * ((w0 * g2m_mx) + g2m_mx*((K*TF_P_eff) ** expo)/(1.0 + K*TF_P_eff))
    return term1 + term2

@njit(cache=True)
def TXreg_ess(w0, g2m_mx, K, TF_P):
    # guard against slight negatives
    TF_P_eff = 0.0 if TF_P < 0.0 else TF_P
    return (w0 * g2m_mx) + g2m_mx*(TF_P_eff**2)/(K**2 + TF_P_eff**2)

@njit(cache=True)
def TXreg_tox(w0, g2m_mx, K, TF_P):
    # guard against slight negatives
    TF_P_eff = 0.0 if TF_P < 0.0 else TF_P
    return (w0 * g2m_mx) + g2m_mx*(1-(TF_P_eff**2)/(K**2 + TF_P_eff**2))
