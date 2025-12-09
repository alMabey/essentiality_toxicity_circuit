import sys
import os
sys.path.append(os.path.abspath(os.path.join('..')))
from scripts.TXreg_funcs import *
import numpy as np
from numba import njit

@njit(cache=True, fastmath=True)
def simple_batch_culture(T):
    return 0

@njit(cache=True, fastmath=True)
def fedbatch_culture(T):
    return 

@njit(cache=True, fastmath=True)
def BatchCultModel_DC(T, Y, hPR, xPR, SysTopol, kin):

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
    mR  = Y[14]; cR  = Y[15]; xR = Y[16]; rr = Y[17]; pR = Y[18]; # ribosomes

    # --- Define pathway variables --------------------------------------------
    mEp = Y[19]; cEp = Y[20]; pEp = Y[21]; iP  = Y[22]; # synthesis pathway enzyme
    mTF = Y[23]; cTF = Y[24]; pTF = Y[25]; TF_P = Y[26] #control transcription factor which binds to P
    mpTox = Y[27]; cpTox = Y[28]; ppTox = Y[29]; #the toxic protein controlled by TF-P
    mTp = Y[30]; cTp = Y[31]; pTp = Y[32]            #the heterologous transporter

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
    xphi  = hPR[23]
    dN    = hPR[24]



    # --- Define pathway parameters ------------------------------------------
    w0          = xPR[0];  # leakiness of synthetic regulated promoters
    wT          = xPR[1]
    wE          = xPR[2]   # maximum TX rate of E
    wEp         = xPR[3];  # maximum TX rate of Eprod
    wTF         = xPR[4];  # maximum TX rate of TF biosensor
    wpTox       = xPR[5];  # maximum TX rate of Eprotease
    wTp         = xPR[6];  # maximum TX rate of the engineered product transporter Tprod

    # --- Heterologous Enzyme Rates ------------------------------------------
    k_Ep = xPR[7]; Km_Ep = xPR[8]; # kcat and KM for Eprod
    k_Tp = xPR[9]; Km_Tp = xPR[10]; # kcat and KM for Ep

    #--- Toxic Protein Curve Parameter ------------------------------------------
    a_energy_pTox = xPR[11]
    a_elongation_pTox = xPR[12]

    # --- Define pathway regulation parameters -------------------------------
    K_E       = xPR[13]; # biosensor affinity for native E. If this is on then P is essential for cell growth
    K_pTox    = xPR[14]; # biosensor affinity for the engineered toxic protein pTox

    # --- Product passive diffusion parameters ------------------------------------------
    kdiffP  = xPR[15]; # rate of diffusion of product into and out of a single cell
    VolCell = xPR[16]; # volume of cell in L
    VolCult = xPR[17]; # working volume of culture in L

    # --- Transcription Factor - Product Binding and Unbinding Rates ------------------------------------------
    ksf     = xPR[18]; # binding rate of TF to P  
    ksr     = xPR[19]; # unbinding rate of TF_P

    # --- Define circuit topologies ------------------------------------------
    lin_trans      = SysTopol[0]; # indicator variable for linear transport
    Tp_trans       = SysTopol[1]; # ind. variable for Tp transport
    T_trans        = SysTopol[2]; # ind. varibale for T transport
    diff_trans     = SysTopol[3]; # ind. variable for diffusion transport

    #mutaually exclusive; only one of these can be 1, rest must be 0
    eprodtox       = SysTopol[4]  # ind. variable for pTox causing toxicity by shutting down energy production
    elongationtox  = SysTopol[5]  # ind. variable for pTox causing toxicity by acting like an antibiotic and shutting down translation

    #% ===== CALCULATE RATES ==================================================

    # --- Define transcription rates ------------------------------------------
    # ee-dependent transcription rates with scaling
    g2mH     = ((wH*ee)/(oX + ee))*(1/(1+(pH/kH)**hH));
    g2mR     = ((wR*ee)/(oR + ee));
    g2rr     = ((wr*ee)/(oR + ee));
    g2mT     = ((wT*ee)/(oX + ee));
    g2mEp    = ((wEp*ee)/(oX + ee));
    g2mTp    = ((wTp*ee)/(oX + ee));
    g2mTF    = ((wTF*ee)/(oX + ee));

    #maximum transcription rates for components under TF-P control
    g2mpTox_mx = ((wpTox*ee)/(oX + ee));
    g2mE_mx  = ((wE*ee)/(oX + ee));

    # include TF-P regulation
    g2mE  = g2mE_mx
    g2mpTox = TXreg_tox(w0, g2mpTox_mx, K_pTox, TF_P)

    # --- Define translation rates --------------------------------------------
    # global translation rate (elongation rate):
    gammaX = (1-elongationtox)*(maxG*ee)/(kG + ee) + elongationtox*(1/(1+(ppTox/a_elongation_pTox)))*(maxG*ee)/(kG + ee);

    # protein per translation complex per min)
    m2pT  = (gammaX/nX)*cT;
    m2pE  = (gammaX/nX)*cE;
    m2pH  = (gammaX/nX)*cH;
    m2xR  = (gammaX/nR)*cR;
    m2pEp = (gammaX/nX)*cEp;
    m2pTF = (gammaX/nX)*cTF;
    m2ppTox = (gammaX/nX)*cpTox;
    m2pTp = (gammaX/nX)*cTp;

    # --- Growth rate ---------------------------------------------------------
    lam = (1/M0)*gammaX*(cH + cR + cT + cEp + cTp + cTF + cpTox + cE);         # ===== UPDATE IF NEW GENES ADDED ====================
    # --- Define Metabolic Reaction Rates -------------------------------------
    r_U = (pT*vT*xS)/(KmT + xS); # Substrate uptake rate
    r_E = eprodtox*(1/(1+(ppTox/a_energy_pTox)))*(pE*vE*iS)/(KmE + iS);  + (1-eprodtox)*(pE*vE*iS)/(KmE + iS)# Host metabolism to biosynthetic precursor

    # --- Production synthesis ------------------------------------------------
    r_Psynth_is = (pEp*k_Ep*iS)/(Km_Ep + iS)

    #If export through passive diffusion
    r_Exp = lin_trans*kdiffP*iP + Tp_trans*k_Tp*pTp*iP/(Km_Tp + iP) + T_trans*vT*pT*iP/(KmT + iP) + diff_trans*kdiffP*(iP - (VolCell/VolCult)*xP);

    #% ===== ENVIRONMENTAL ODEs ===============================================

    # --- total biomass -------------------------------------------------------
    dNdt = (lam*N) - dN*N;

    dxS = kin - (r_U*N)

    # --- total product in media ----------------------------------------------
    dxP = (r_Exp*N);

    #% ===== HOST ODEs ========================================================
    # --- host metabolism -----------------------------------------------------
    diS = r_U - r_E - r_Psynth_is - lam*iS;                                 # ===== UPDATE IF NEW is CONSUMING REACTION ADDED ====================
    dee = (sS*r_E) - nR*m2xR  - nX*m2pT  - nX*m2pE - nX*m2pH - nX*m2pEp  - nX*m2pTF - nX*m2ppTox - nX*m2pTp - lam*ee

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

    # --- inactive ribosomes (xR) and rRNA (rr) -------------------------------
    dmR = g2mR - (lam + deg_m)*mR + m2xR - bX*pR*mR + uX*cR;
    dcR = - lam*cR + bX*pR*mR - uX*cR - m2xR;
    dxR = m2xR - lam*xR - brho*xR*rr + urho*pR
    drr = g2rr - lam*rr - brho*xR*rr + urho*pR

    # -- activated ribosome (in complex with ribosomal RNA), pR ---------------
    dpR = brho*xR*rr - urho*pR - lam*pR + m2pT - bX*pR*mT  + uX*cT + m2pE  - bX*pR*mE  + uX*cE + m2pH  \
          - bX*pR*mH  + uX*cH + m2xR  - bX*pR*mR  + uX*cR                    \
          + m2pEp - bX*pR*mEp + uX*cEp + m2pTF - bX*pR*mTF + uX*cTF + m2ppTox - bX*pR*mpTox + uX*cpTox + m2pTp - bX*pR*mTp + uX*cTp \


    #% ===== pathway AND CIRCUIT ODEs =========================================
    # --- pathway enzyme (Ep) -------------------------------------------------
    dmEp = g2mEp - (lam + deg_m)*mEp + m2pEp - bX*pR*mEp + uX*cEp;
    dcEp = - lam*cEp + bX*pR*mEp - uX*cEp - m2pEp;
    dpEp = m2pEp - lam*pEp;

    # --- metabolism - formation of intracellular product ---------------------
    diP  = r_Psynth_is - r_Exp - lam*iP - ksf*pTF*iP + ksr*TF_P;

    # --- TF biosensor --------------------------------------------------------
    dmTF = g2mTF - (lam + deg_m)*mTF + m2pTF - bX*pR*mTF + uX*cTF;
    dcTF = -lam*cTF + bX*pR*mTF - uX*cTF - m2pTF;
    dpTF = m2pTF - lam*pTF - ksf*pTF*iP + ksr*TF_P;
    dTF_P = ksf*pTF*iP - ksr*TF_P - lam*TF_P

    # ---  Toxic Protein pTox ----------------------------------------
    dmpTox = g2mpTox - (lam + deg_m)*mpTox + m2ppTox - bX*pR*mpTox + uX*cpTox;
    dcpTox = - lam*cpTox + bX*pR*mpTox - uX*cpTox - m2ppTox;
    dppTox = m2ppTox - lam*ppTox;

    # --- pathway exporter enzyme (Tp) ----------------------------------------
    dmTp = g2mTp - (lam + deg_m)*mTp + m2pTp - bX*pR*mTp + uX*cTp;
    dcTp = - lam*cTp + bX*pR*mTp - uX*cTp - m2pTp;
    dpTp = m2pTp - lam*pTp;

    #% ===== RETURN OUTPUTS ===================================================

    # --- derivatives ---------------------------------------------------------
    dY = np.empty(33, dtype=np.float64)
    # assign each element in order
    dY[:] = [
        dNdt, dxS, dxP,
        diS, dee,
        dmT, dcT, dpT,
        dmE, dcE, dpE,
        dmH, dcH, dpH,
        dmR, dcR, dxR, drr, dpR,
        dmEp, dcEp, dpEp,
        diP,
        dmTF, dcTF, dpTF, dTF_P,
        dmpTox, dcpTox, dppTox,
        dmTp, dcTp, dpTp,
    ]
    # --- TX and TL rates -----------------------------------------------------
    TXrates = np.array([g2mT, g2mE, g2mH, g2mR, g2rr, g2mEp, g2mTF, g2mpTox, g2mTp]);
    TLrates = np.array([m2pT, m2pE, m2pH, m2xR, m2pEp, m2pTF, m2ppTox, m2pTp]);

    # --- protein masses and ribosomal mass fraction --------------------------
    protmass = np.array([nX*pT, nX*pE, nX*pH, nR*(xR + pR + cT + cE + cH + cR + cEp + cTF + cpTox 
                                                         + cTp), nX*(pTF + TF_P), nX*ppTox, nX*pEp, nX*pTp]);
    summass = np.sum(protmass);
    ribomassfrac = (nR/M0)*(xR + pR + cT + cE + cH + cR + cEp + cTF + cpTox + cTp);

    # ---- fluxes -------------------------------------------------------------
                     #s-> in  s->e    s -> P       P -> out
    fluxes = np.array([r_U,   r_E,    r_Psynth_is, r_Exp]);

    return dY#, lam, TXrates, TLrates, gammaX, protmass, summass, ribomassfrac, fluxes

@njit(cache=True, fastmath=True)
def BatchCultModel_DC_full(T, Y, hPR, xPR, SysTopol, kin):

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
    mR  = Y[14]; cR  = Y[15]; xR = Y[16]; rr = Y[17]; pR = Y[18]; # ribosomes

    # --- Define pathway variables --------------------------------------------
    mEp = Y[19]; cEp = Y[20]; pEp = Y[21]; iP  = Y[22]; # synthesis pathway enzyme
    mTF = Y[23]; cTF = Y[24]; pTF = Y[25]; TF_P = Y[26] #control transcription factor which binds to P
    mpTox = Y[27]; cpTox = Y[28]; ppTox = Y[29]; #the toxic protein controlled by TF-P
    mTp = Y[30]; cTp = Y[31]; pTp = Y[32]            #the heterologous transporter

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
    xphi  = hPR[23]
    dN    = hPR[24]



    # --- Define pathway parameters ------------------------------------------
    w0          = xPR[0];  # leakiness of synthetic regulated promoters
    wT          = xPR[1]
    wE          = xPR[2]   # maximum TX rate of E
    wEp         = xPR[3];  # maximum TX rate of Eprod
    wTF         = xPR[4];  # maximum TX rate of TF biosensor
    wpTox       = xPR[5];  # maximum TX rate of Eprotease
    wTp         = xPR[6];  # maximum TX rate of the engineered product transporter Tprod

    # --- Heterologous Enzyme Rates ------------------------------------------
    k_Ep = xPR[7]; Km_Ep = xPR[8]; # kcat and KM for Eprod
    k_Tp = xPR[9]; Km_Tp = xPR[10]; # kcat and KM for Ep

    #--- Toxic Protein Curve Parameter ------------------------------------------
    a_energy_pTox = xPR[11]
    a_elongation_pTox = xPR[12]

    # --- Define pathway regulation parameters -------------------------------
    K_E       = xPR[13]; # biosensor affinity for native E. If this is on then P is essential for cell growth
    K_pTox    = xPR[14]; # biosensor affinity for the engineered toxic protein pTox

    # --- Product passive diffusion parameters ------------------------------------------
    kdiffP  = xPR[15]; # rate of diffusion of product into and out of a single cell
    VolCell = xPR[16]; # volume of cell in L
    VolCult = xPR[17]; # working volume of culture in L

    # --- Transcription Factor - Product Binding and Unbinding Rates ------------------------------------------
    ksf     = xPR[18]; # binding rate of TF to P  
    ksr     = xPR[19]; # unbinding rate of TF_P

    # --- Define circuit topologies ------------------------------------------
    lin_trans      = SysTopol[0]; # indicator variable for linear transport
    Tp_trans       = SysTopol[1]; # ind. variable for Tp transport
    T_trans        = SysTopol[2]; # ind. varibale for T transport
    diff_trans     = SysTopol[3]; # ind. variable for diffusion transport

    #mutaually exclusive; only one of these can be 1, rest must be 0
    eprodtox       = SysTopol[4]  # ind. variable for pTox causing toxicity by shutting down energy production
    elongationtox  = SysTopol[5]  # ind. variable for pTox causing toxicity by acting like an antibiotic and shutting down translation

    #% ===== CALCULATE RATES ==================================================

    # --- Define transcription rates ------------------------------------------
    # ee-dependent transcription rates with scaling
    g2mH     = ((wH*ee)/(oX + ee))*(1/(1+(pH/kH)**hH));
    g2mR     = ((wR*ee)/(oR + ee));
    g2rr     = ((wr*ee)/(oR + ee));
    g2mT     = ((wT*ee)/(oX + ee));
    g2mEp    = ((wEp*ee)/(oX + ee));
    g2mTp    = ((wTp*ee)/(oX + ee));
    g2mTF    = ((wTF*ee)/(oX + ee));

    #maximum transcription rates for components under TF-P control
    g2mpTox_mx = ((wpTox*ee)/(oX + ee));
    g2mE_mx  = ((wE*ee)/(oX + ee));

    # include TF-P regulation
    g2mE  = TXreg_ess(w0, g2mE_mx,  K_E,  TF_P)
    g2mpTox = TXreg_mannan2025(-1, w0, g2mpTox_mx, K_pTox, TF_P)

    # --- Define translation rates --------------------------------------------
    # global translation rate (elongation rate):
    gammaX = (1-elongationtox)*(maxG*ee)/(kG + ee) + elongationtox*(1/(1+(ppTox/a_elongation_pTox)))*(maxG*ee)/(kG + ee);

    # protein per translation complex per min)
    m2pT  = (gammaX/nX)*cT;
    m2pE  = (gammaX/nX)*cE;
    m2pH  = (gammaX/nX)*cH;
    m2xR  = (gammaX/nR)*cR;
    m2pEp = (gammaX/nX)*cEp;
    m2pTF = (gammaX/nX)*cTF;
    m2ppTox = (gammaX/nX)*cpTox;
    m2pTp = (gammaX/nX)*cTp;

    # --- Growth rate ---------------------------------------------------------
    lam = (1/M0)*gammaX*(cH + cR + cT + cEp + cTp + cTF + cpTox + cE);         # ===== UPDATE IF NEW GENES ADDED ====================
    # --- Define Metabolic Reaction Rates -------------------------------------
    r_U = (pT*vT*xS)/(KmT + xS); # Substrate uptake rate
    r_E = eprodtox*(1/(1+(ppTox/a_energy_pTox)))*(pE*vE*iS)/(KmE + iS);  + (1-eprodtox)*(pE*vE*iS)/(KmE + iS)# Host metabolism to biosynthetic precursor

    # --- Production synthesis ------------------------------------------------
    r_Psynth_is = (pEp*k_Ep*iS)/(Km_Ep + iS)

    #If export through passive diffusion
    r_Exp = lin_trans*kdiffP*iP + Tp_trans*k_Tp*pTp*iP/(Km_Tp + iP) + T_trans*vT*pT*iP/(KmT + iP) + diff_trans*kdiffP*(iP - (VolCell/VolCult)*xP);

    #% ===== ENVIRONMENTAL ODEs ===============================================

    # --- total biomass -------------------------------------------------------
    dNdt = (lam*N) - dN*N;

    dxS = kin - (r_U*N)

    # --- total product in media ----------------------------------------------
    dxP = (r_Exp*N);

    #% ===== HOST ODEs ========================================================
    # --- host metabolism -----------------------------------------------------
    diS = r_U - r_E - r_Psynth_is - lam*iS;                                 # ===== UPDATE IF NEW is CONSUMING REACTION ADDED ====================
    dee = (sS*r_E) - nR*m2xR  - nX*m2pT  - nX*m2pE - nX*m2pH - nX*m2pEp  - nX*m2pTF - nX*m2ppTox - nX*m2pTp - lam*ee

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

    # --- inactive ribosomes (xR) and rRNA (rr) -------------------------------
    dmR = g2mR - (lam + deg_m)*mR + m2xR - bX*pR*mR + uX*cR;
    dcR = - lam*cR + bX*pR*mR - uX*cR - m2xR;
    dxR = m2xR - lam*xR - brho*xR*rr + urho*pR
    drr = g2rr - lam*rr - brho*xR*rr + urho*pR

    # -- activated ribosome (in complex with ribosomal RNA), pR ---------------
    dpR = brho*xR*rr - urho*pR - lam*pR + m2pT - bX*pR*mT  + uX*cT + m2pE  - bX*pR*mE  + uX*cE + m2pH  \
          - bX*pR*mH  + uX*cH + m2xR  - bX*pR*mR  + uX*cR                    \
          + m2pEp - bX*pR*mEp + uX*cEp + m2pTF - bX*pR*mTF + uX*cTF + m2ppTox - bX*pR*mpTox + uX*cpTox + m2pTp - bX*pR*mTp + uX*cTp \


    #% ===== pathway AND CIRCUIT ODEs =========================================
    # --- pathway enzyme (Ep) -------------------------------------------------
    dmEp = g2mEp - (lam + deg_m)*mEp + m2pEp - bX*pR*mEp + uX*cEp;
    dcEp = - lam*cEp + bX*pR*mEp - uX*cEp - m2pEp;
    dpEp = m2pEp - lam*pEp;

    # --- metabolism - formation of intracellular product ---------------------
    diP  = r_Psynth_is - r_Exp - lam*iP - ksf*pTF*iP + ksr*TF_P;

    # --- TF biosensor --------------------------------------------------------
    dmTF = g2mTF - (lam + deg_m)*mTF + m2pTF - bX*pR*mTF + uX*cTF;
    dcTF = -lam*cTF + bX*pR*mTF - uX*cTF - m2pTF;
    dpTF = m2pTF - lam*pTF - ksf*pTF*iP + ksr*TF_P;
    dTF_P = ksf*pTF*iP - ksr*TF_P - lam*TF_P

    # ---  Toxic Protein pTox ----------------------------------------
    dmpTox = g2mpTox - (lam + deg_m)*mpTox + m2ppTox - bX*pR*mpTox + uX*cpTox;
    dcpTox = - lam*cpTox + bX*pR*mpTox - uX*cpTox - m2ppTox;
    dppTox = m2ppTox - lam*ppTox;

    # --- pathway exporter enzyme (Tp) ----------------------------------------
    dmTp = g2mTp - (lam + deg_m)*mTp + m2pTp - bX*pR*mTp + uX*cTp;
    dcTp = - lam*cTp + bX*pR*mTp - uX*cTp - m2pTp;
    dpTp = m2pTp - lam*pTp;

    #% ===== RETURN OUTPUTS ===================================================

    # --- derivatives ---------------------------------------------------------
    dY = np.empty(33, dtype=np.float64)
    # assign each element in order
    dY[:] = [
        dNdt, dxS, dxP,
        diS, dee,
        dmT, dcT, dpT,
        dmE, dcE, dpE,
        dmH, dcH, dpH,
        dmR, dcR, dxR, drr, dpR,
        dmEp, dcEp, dpEp,
        diP,
        dmTF, dcTF, dpTF, dTF_P,
        dmpTox, dcpTox, dppTox,
        dmTp, dcTp, dpTp,
    ]
    # --- TX and TL rates -----------------------------------------------------
    TXrates = np.array([g2mT, g2mE, g2mH, g2mR, g2rr, g2mEp, g2mTF, g2mpTox, g2mTp]);
    TLrates = np.array([m2pT, m2pE, m2pH, m2xR, m2pEp, m2pTF, m2ppTox, m2pTp]);

    # --- protein masses and ribosomal mass fraction --------------------------
    protmass = np.array([nX*pT, nX*pE, nX*pH, nR*(xR + pR + cT + cE + cH + cR + cEp + cTF + cpTox 
                                                         + cTp), nX*(pTF + TF_P), nX*ppTox, nX*pEp, nX*pTp]);
    summass = np.sum(protmass);
    ribomassfrac = (nR/M0)*(xR + pR + cT + cE + cH + cR + cEp + cTF + cpTox + cTp);

    # ---- fluxes -------------------------------------------------------------
                     #s-> in  s->e    s -> P       P -> out
    fluxes = np.array([r_U,   r_E,    r_Psynth_is, r_Exp]);

    return dY, lam, TXrates, TLrates, gammaX, protmass, summass, ribomassfrac, fluxes



@njit(cache=True, fastmath=True)
def BatchCultModel_SS(T, Y, hPR, xPR, SysTopol):

    # --- Calculate derivative ------------------------------------------------
    dY = BatchCultModel_DC(T, Y, hPR, xPR, SysTopol, 0)

    # --- Make extracellular reactions zero -----------------------------------
    dY[0] = 0
    dY[1] = 0

    return dY
