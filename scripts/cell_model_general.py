import sys
import os
sys.path.append(os.path.abspath(os.path.join('..')))
from scripts.TXreg_funcs import *
import numpy as np
from numba import njit

@njit(cache=True, fastmath=True)
def BatchCultModel_DC(T, Y, hPR, xPR, SysTopol):

    # --- Define external variables -------------------------------------------
    N  = Y[0]; # total biomass
    xS = Y[1]; # substrate in media
    xP = Y[2]; # product in media

    # --- Define host variables -----------------------------------------------
    # Note m = mRNA, c = complex, p = protein
    iS  = Y[3]; ee  = Y[4];             # internal substrate (is) and energY (ee)
    mT  = Y[5]; cT  = Y[6]; pT = Y[7]; # transporter
    mE  = Y[8]; cE  = Y[9]; pE = Y[10]; # enzymes
    mH  = Y[11]; cH  = Y[12]; pH = Y[13]; # approx fixed amount of host proteins (q-fraction)
    mX  = Y[14]; cX  = Y[15]; pX = Y[16]; # export reaction
    mR  = Y[17]; cR  = Y[18]; xR = Y[19]; rr = Y[20]; pR = Y[21]; # ribosomes

    # --- Define pathway variables --------------------------------------------
    mEprod = Y[22]; cEprod = Y[23]; pEprod = Y[24]; iP  = Y[25]; # synthesis pathway enzyme
    mTF = Y[26]; cTF = Y[27]; pTF = Y[28]; TF_P = Y[29] 
    mEprotease = Y[30]; cEprotease = Y[31]; pEprotease = Y[32]; 
    mTprod = Y[33]; cTprod = Y[34]; pTprod = Y[35]            # binds to the product P and induces E, inhibits E_prot # the transcription factor/P dimer              # inducible biosensor for synthetic inducer 

    #% ===== PARAMETERS =======================================================

    # --- Define host sYstem paramters ---------------------------------------d
    sS    = hPR[0]
    vT    = hPR[1]; vE   = hPR[2]
    KmT   = hPR[3]; KmE  = hPR[4]
    wX    = hPR[5]; wH   = hPR[6]; wR = hPR[7]; wr = hPR[8]
    oX    = hPR[9]; oR   = hPR[10]
    nX    = hPR[11]; nR   = hPR[12]
    bX    = hPR[13]; uX   = hPR[14]
    brho  = hPR[15]; urho = hPR[16]
    deg_m = hPR[17]
    kH    = hPR[18]; hH   = hPR[19]
    maxG  = hPR[20]; kG   = hPR[21]; M0 = hPR[22]
    xphi  = hPR[23]; vX   = hPR[24]; KmX = hPR[25]

    # --- Define pathway parameters ------------------------------------------
    w0          = xPR[0];  # leakiness of synthetic regulated promoters
    wE          = xPR[1]   # maximum TX rate of E
    wEprod      = xPR[2];  # maximum TX rate of Eprod
    wTF         = xPR[3];  # maximum TX rate of TF biosensor
    wEprotease  = xPR[4];  # maximum TX rate of Eprotease
    wTprod      = xPR[5];  # maximum TX rate of the engineered product transporter Tprod
    k_Eprod     = xPR[6]; Km_Eprod     = xPR[7]; # kcat and KM for Eprod
    k_Eprotease = xPR[8]; Km_Eprotease = xPR[9]; # kcat and KM for Eprotease
    k_Tprod = xPR[10]; Km_Tprod = xPR[11]; # kcat and KM for Ep

    # --- Define pathway regulation parameters -------------------------------
    K_T            = xPR[12]
    K_E            = xPR[13]; # biosensor affinity for native E. If this is on then P is essential for cell growth
    K_Eprod        = xPR[14]; # biosensor affinity for the engineered Eprod
    K_Eprotease    = xPR[15]; # biosensor affinity for the engineered Eprotease
    K_Tprod        = xPR[16]; # biosensor affinity for the engineered Tprod
    K_TF           = xPR[17]; # biosensor affinity for TF, if there is recursive control

    # --- Product passive diffusion parameters ------------------------------------------
    kdiffP  = xPR[18]; # rate of diffusion of product into and out of a single cell
    VolCell = xPR[19]; # volume of cell in L
    VolCult = xPR[20]; # working volume of culture in L

    # --- Transcription Factor - Product Binding and Unbinding Rates ------------------------------------------
    ksf     = xPR[21]; # binding rate of TF to P
    ksr     = xPR[22]; # unbinding rate of TF_P

    # --- Define circuit topologies ------------------------------------------
    # circuit network topology
    ctT         = SysTopol[0]; # TF_P control on TX of T
    ctE         = SysTopol[1]; # TF_P control on TX of E
    ctEprod     = SysTopol[2]; # TF_P control on TX of Ep
    ctTF        = SysTopol[3]; # TF_P control on TF
    ctEprotease = SysTopol[4]; # TF_P control on Eprotease
    ctTprod     = SysTopol[5]; # TF_P control on TX of the engineered P transporter Tp

    #functions within topology (0 = Mannan 2025 TX regulation model)
    T_TXmodel         = SysTopol[6]
    E_TXmodel         = SysTopol[7]
    Eprod_TXmodel     = SysTopol[8]
    TF_TXmodel        = SysTopol[9]
    Eprotease_TXmodel = SysTopol[10]
    Tprod_TXmodel     = SysTopol[11]

    #% ===== CALCULATE RATES ==================================================

    # --- Define transcription rates ------------------------------------------
    # ee-dependent transcription rates with scaling
    g2mH     = ((wH*ee)/(oX + ee))*(1/(1+(pH/kH)**hH));
    g2mR     = ((wR*ee)/(oR + ee));
    g2rr     = ((wr*ee)/(oR + ee));
    g2mX     = xphi*(wX*ee)/(oX + ee);
    g2mE     = xphi*(wE*ee)/(oX + ee);
    g2mT_mx  = ((wX*ee)/(oX + ee));
    g2mE_mx  = ((wE*ee)/(oX + ee));
    g2mEprod_mx = ((wEprod*ee)/(oX + ee));
    g2mTF_mx = ((wTF*ee)/(oX + ee));
    g2mEprotease_mx = ((wEprotease*ee)/(oX + ee));
    g2mTprod_mx = ((wTprod*ee)/(oX + ee));

    # include TF regulation of TX:
    TXmodels = [TXreg_mannan2025]
    g2mT  = TXmodels[T_TXmodel](ctT,  w0, g2mT_mx,  K_T,  TF_P)
    g2mE  = TXmodels[E_TXmodel](ctE,  w0, g2mE_mx,  K_E,  TF_P)
    g2mEprod = TXmodels[Eprod_TXmodel](ctEprod, w0, g2mEprod_mx, K_Eprod, TF_P)
    g2mTF = TXmodels[TF_TXmodel](ctTF, w0, g2mTF_mx, K_TF, TF_P)
    g2mEprotease = TXmodels[Eprotease_TXmodel](ctEprotease, w0, g2mEprotease_mx, K_Eprotease, TF_P)
    g2mTprod = TXmodels[Tprod_TXmodel](ctTprod, w0, g2mTprod_mx, K_Tprod, TF_P)


    # --- Define translation rates --------------------------------------------
    # global translation rate (elongation rate):
    gammaX = (maxG*ee)/(kG + ee);

    # protein per translation complex per min)
    m2pT  = (gammaX/nX)*cT;
    m2pE  = (gammaX/nX)*cE;
    m2pH  = (gammaX/nX)*cH;
    m2xR  = (gammaX/nR)*cR;
    m2pX  = (gammaX/nX)*cX;
    m2pEprod = (gammaX/nX)*cEprod;
    m2pTF = (gammaX/nX)*cTF;
    m2pEprotease = (gammaX/nX)*cEprotease;
    m2pTprod = (gammaX/nX)*cTprod;

    # --- Growth rate ---------------------------------------------------------
    lam = (1/M0)*gammaX*(cH + cR + cT + cE + cX + cEprod + cTF + cEprotease + cTprod);         # ===== UPDATE IF NEW GENES ADDED ====================
    # --- Define Metabolic Reaction Rates -------------------------------------
    r_U = (pT*vT*xS)/(KmT + xS); # Substrate uptake rate
    r_E = (pE*vE*iS)/(KmE + iS); # Host metabolism to biosynthetic precursor

    # --- Production synthesis ------------------------------------------------
    r_Psynth_is = (pEprod*k_Eprod*iS)/(Km_Eprod + iS)

    #If export through passive diffusion
    r_Exp = kdiffP*(iP - (VolCell/VolCult)*xP);

    #If export through transporters
    #r_Exp = ((pX*vX*iP)/(KmX + iP)) + ((pTprod*k_Tprod*iP)/(Km_Tprod + iP));

    # --- Protease-Ribosome Interaction ------------------------------------------------
    r_R_protease = k_Eprotease*pR*pEprotease/(Km_Eprotease + pR)

    #% ===== ENVIRONMENTAL ODEs ===============================================

    # --- total biomass -------------------------------------------------------
    dN = (lam*N);

    dxS = - (r_U*N);

    # --- total product in media ----------------------------------------------
    dxP = (r_Exp*N);

    #% ===== HOST ODEs ========================================================
    # --- host metabolism -----------------------------------------------------
    diS = r_U - r_E - r_Psynth_is - lam*iS;                                 # ===== UPDATE IF NEW is CONSUMING REACTION ADDED ====================
    dee = (sS*r_E) - nR*m2xR  - nX*m2pT  - nX*m2pE - nX*m2pX - nX*m2pH - nX*m2pEprod  - nX*m2pTF - nX*m2pEprotease - nX*m2pTprod - lam*ee

    # --- substrate transporter (T) -------------------------------------------
    dmT = g2mT - (lam + deg_m)*mT + m2pT - bX*pR*mT + uX*cT;
    dcT = - lam*cT + bX*pR*mT - uX*cT - m2pT;
    dpT = m2pT - lam*pT;

    # --- metabolic enzyme (E) ------------------------------------------------
    dmE = g2mE - (lam + deg_m)*mE + m2pE - bX*pR*mE + uX*cE;
    dcE = - lam*cE + bX*pR*mE - uX*cE - m2pE;
    dpE = m2pE - lam*pE;

    # --- house-keeping proteins (H) ------------------------------------------
    dmH = g2mH - (lam + deg_m)*mH + m2pH - bX*pR*mH + uX*cH;
    dcH = - lam*cH + bX*pR*mH - uX*cH - m2pH;
    dpH = m2pH - lam*pH;

    # --- native product exporter (X) -----------------------------------------
    dmX = g2mX - (lam + deg_m)*mX + m2pX - bX*pR*mX + uX*cX;
    dcX = - lam*cX + bX*pR*mX - uX*cX - m2pX;
    dpX = m2pX - lam*pX;

    # --- inactive ribosomes (xR) and rRNA (rr) -------------------------------
    dmR = g2mR - (lam + deg_m)*mR + m2xR - bX*pR*mR + uX*cR;
    dcR = - lam*cR + bX*pR*mR - uX*cR - m2xR;
    dxR = m2xR - lam*xR - brho*xR*rr + urho*pR;
    drr = g2rr - lam*rr - brho*xR*rr + urho*pR;

    # -- activated ribosome (in complex with ribosomal RNA), pR ---------------
    dpR = brho*xR*rr - urho*pR - lam*pR + m2pT  - bX*pR*mT  + uX*cT + m2pE  - bX*pR*mE  + uX*cE + m2pH  \
          - bX*pR*mH  + uX*cH + m2pX  - bX*pR*mX  + uX*cX + m2xR  - bX*pR*mR  + uX*cR                    \
          + m2pEprod - bX*pR*mEprod + uX*cEprod + m2pTF - bX*pR*mTF + uX*cTF + m2pEprotease - bX*pR*mEprotease + uX*cEprotease + m2pTprod - bX*pR*mTprod + uX*cTprod \
            - r_R_protease

    #% ===== pathway AND CIRCUIT ODEs =========================================
    # --- pathway enzyme (Ep) -------------------------------------------------
    dmEprod = g2mEprod - (lam + deg_m)*mEprod + m2pEprod - bX*pR*mEprod + uX*cEprod;
    dcEprod = - lam*cEprod + bX*pR*mEprod - uX*cEprod - m2pEprod;
    dpEprod = m2pEprod - lam*pEprod;

    # --- metabolism - formation of intracellular product ---------------------
    diP  = r_Psynth_is - r_Exp - lam*iP - ksf*pTF*iP + ksr*TF_P;
    dxP = r_Exp

    # --- TF biosensor --------------------------------------------------------
    dmTF = g2mTF - (lam + deg_m)*mTF + m2pTF - bX*pR*mTF + uX*cTF;
    dcTF = -lam*cTF + bX*pR*mTF - uX*cTF - m2pTF;
    dpTF = m2pTF - lam*pTF - ksf*pTF*iP + ksr*TF_P;
    dTF_P = ksf*pTF*iP - ksr*TF_P - lam*TF_P

    # ---  protease enzyme (Eprotease) ----------------------------------------
    dmEprotease = g2mEprotease - (lam + deg_m)*mEprotease + m2pEprotease - bX*pR*mEprotease + uX*cEprotease;
    dcEprotease = - lam*cEprotease + bX*pR*mEprotease - uX*cEprotease - m2pEprotease;
    dpEprotease = m2pEprotease - lam*pEprotease;

    # --- pathway exporter enzyme (Tp) ----------------------------------------
    dmTprod = g2mTprod - (lam + deg_m)*mTprod + m2pTprod - bX*pR*mTprod + uX*cTprod;
    dcTprod = - lam*cTprod + bX*pR*mTprod - uX*cTprod - m2pTprod;
    dpTprod = m2pTprod - lam*pTprod;

    #% ===== RETURN OUTPUTS ===================================================

    # --- Update derivatives --------------------------------------------------
    ddN  = dN; # /N0;
    ddxS = dxS; # /xS0;
    ddxP = dxP; # /xS0;
    ddiS = diS; # /xS0;
    ddee = dee; # /xS0;
    ddiP = diP;

    # --- derivatives ---------------------------------------------------------
    dY = np.empty(36, dtype=np.float64)

    # assign each element in order
    dY[:] = [
        ddN, ddxS, ddxP,
        ddiS, ddee,
        dmT, dcT, dpT,
        dmE, dcE, dpE,
        dmH, dcH, dpH,
        dmX, dcX, dpX,
        dmR, dcR, dxR, drr, dpR,
        dmEprod, dcEprod, dpEprod,
        ddiP,
        dmTF, dcTF, dpTF, dTF_P,
        dmEprotease, dcEprotease, dpEprotease,
        dmTprod, dcTprod, dpTprod,
    ]
    # --- TX and TL rates -----------------------------------------------------
    TXrates = np.array([g2mT, g2mE, g2mH, g2mR, g2rr, g2mX, g2mEprod, g2mTF, g2mEprotease, g2mTprod]);
    TLrates = np.array([m2pT, m2pE, m2pH, m2xR, m2pX, m2pEprod, m2pTF, m2pEprotease, m2pTprod]);

    # --- protein masses and ribosomal mass fraction --------------------------
    protmass = np.array([nX*pT, nX*pE, nX*pX, nX*pH, nR*(xR + pR + cT + cE + cH + cR + cX + cEprod + cTF + cEprotease 
                                                         + cTprod), nX*(pTF + TF_P), nX*pEprod, nX*pTprod]);
    summass = np.sum(protmass);
    ribomassfrac = (nR/M0)*(xR + pR + cT + cE + cH + cR + cX + cEprod + cTF + cEprotease + cTprod);

    # ---- fluxes -------------------------------------------------------------
                     #s-> in  s->e    s -> P       P -> out
    fluxes = np.array([r_U,   r_E,    r_Psynth_is, r_Exp]);

    return dY#, lam, TXrates, TLrates, gammaX, protmass, summass, ribomassfrac, fluxes

@njit(cache=True, fastmath=True)
def BatchCultModel_SS(T, Y, hPR, xPR, SysTopol):

    # --- Calculate derivative ------------------------------------------------
    dY = BatchCultModel_DC(T, Y, hPR, xPR, SysTopol)

    # --- Make extracellular reactions zero -----------------------------------
    dY[0] = 0
    dY[1] = 0
    dY[2] = 0

    return dY
