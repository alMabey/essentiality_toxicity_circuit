import sys
import os
sys.path.append(os.path.abspath(os.path.join('..')))
def model_params(sS0, vX0, KmX0, VolCult0, leaky_control=False):


    # --- Parameters of host components --------------------------------------
    # Nutrients
    sS      = sS0;          # 2 nutrient efficiency (i.e. stoichiometry of substrate to precursors conversion)
    vT      = 728;          # 3 kcat of transport rxn
    vE      = 5800;         # 4 kcat of metabolic rxn (with enzyme)
    KmT     = 1e3;          # 5 Km of transport rxn
    KmE     = 1e3;          # 6 Km of metabolic rxn

    # Transcription Parameters
    wX      = 4.14;         # 7 max transcription rate of metabolic proteins (transporter and enzymes)
    wH      = 948.93;       # 8 max transcription rate of house-keeping proteins
    wR      = 930;          # 9 max transcription rate of ribosomal protein
    wr      = 3170;         # 10 max transcription rate of ribosomal RNA
    wE      = 4.14

    oX      = 4.38;         # 11 effective Km of transcription of mRNA of metabolic proteins (T,H,E)
    oR      = 426.87;       # 12 effective Km of transcription of ribosomal mRNA and RNA

    # Translation Parameters
    nX      = 300;          # 13 average number of codons on mRNA of metabolic proteins (T,H,E), for translation
    nR      = 7459;         # 14 average number of codons on mRNA of ribosomal mRNA, for translation
    bX      = 1;            # 15 forward rate of active ribosomes to form complex with mRNA
    uX      = 1;            # 16 rate of unbinding ribosome from mRNA
    brho    = 1;            # 17 rate of binding ribosomal RNA to ribosome protein part
    urho    = 1;            # 18 rate of unbinding of ribosomal RNA to ribosome protein part
    deg_m   = 0.1;          # 19 average mRNA degradation rate
    kH      = 152219 * 0.8; # 20 inverse threshold of conc of house-keeping proteins to inhibit their own expression (phenomenological - to maintain fixed levels for diff conditions)
    hH      = 4 * 2;        # 21 Hill coefficient of house-keeping proteins to inhibit their own transcriptional expression
    maxG    = 1260;         # 22 maximal translation rate
    kG      = 7;            # 23 threshold conc of precursor for half-max translation rate
    M0      = 1e8;          # 24 total cell mass

    # Parameters of promiscuous host exporter:
    xphi    = 0.05;  # 25 scaling factor for max transcription rate of host co-opted transporter X, TX = xphi*wX
    vX      = vX0;    # 26 kcat of ip export rxn
    KmX     = KmX0;   # 27 Km of ip export rxn

    # --- Parameters of exogenous components ---------------------------------

    # Parameters of TX expression and kinetics of heterologous enzymes:
    if leaky_control:
        w0 = 1e-4;            # 1 leakiness of the artifical promoters as a fraction of their maximum transcription rate (%)
    else:
        w0 = 0

    wEp     = 20;              # 2 max transcription rate of Ep
    k_Ep    = vE; Km_Ep = KmE; # 5,6 kcat and Km for kinetics of Ep

    wTF        = 20;

    wpTox = 20;

    wTp     = 20;              # 3 max transcription rate of Tp
    k_Tp    = vT; Km_Tp = KmT; # 1e5*KmT; % 7,8 kcat and Km for kinetics of Tp

    # Parameters for toxicity
    a_energy_pTox = 1E4
    a_elongation_pTox = 1E4


    # Parameters of the control system 
    K_E     = 1/(1000/300);    
    K_pTox = 1/(1000/300); 

    # Parameters for inducer uptake and induction:
    kdiffP  = 3600/60;         # 19 1/min - rate of diffusion of inducer into and out of a single cell
    VolCell = 1e-15;           # 20 volume of cell in L - based on 1uM wide by 2uM sized E. coli cell (Neidhardt (1990))
    VolCult = VolCult0;        # 21 working volume of culture in L - based on a 3L benchtop vessel
    ksf     = 0.1;       # 22 1/(molecules^2.min) - forward rate of TF sequestration by inducer I - based on Mannan & Bates (2021)
    ksr     = 1000;       # 23 1/min - reverse rate of TF sequestration by inducer I, based on Mannan & Bates (2021)

    # --- Return vectors -----------------------------------------------------

    # Define vector of ...
    # ... params of endogenous components:
    h_params   = [sS, vT, vE, KmT, KmE, wX, wH, wR, wr, oX, oR, nX, nR, bX, uX, brho, urho, deg_m, kH, hH, maxG, kG, M0, xphi, vX, KmX]
    # ... params of exogenous components:
    x_params   = [w0, wE, wEp, wTF, wpTox, wTp, k_Ep, Km_Ep, k_Tp, Km_Tp, a_energy_pTox, a_elongation_pTox, K_E, K_pTox, kdiffP, VolCell, VolCult, ksf, ksr]

    return [h_params,x_params] 