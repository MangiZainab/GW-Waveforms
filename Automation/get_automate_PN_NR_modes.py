#python get_automate_PN_NR_modes.py --waveformnumber 0123

import numpy as np
import matplotlib.pyplot as plt

from Automate_utils import *
from get_sxs import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--waveformnumber", type=str, help="waveform number")
opts = parser.parse_args() 

T_for_RigRot, T_Hybrid_Interval_Start, T_Hybrid_Interval_End = -0.7, -0.7, -0.3

DISTANCE=300*megaparsec
MASS = 70 * msun

################## Get PN modes from sxs  ################ 

waveformnumber = opts.waveformnumber
#get_modes_from_sxs(waveformnumber, MASS=70, DISTANCE=300*megaparsec)




#################### Get NR mode #############
# we will implement automation by using argparser for parameters but below  




metadata = sxs.load("SXS:BBH:"+waveformnumber+"/Lev/metadata.json")
qratio = metadata["reference_mass_ratio"]
Mtotal = 70
spin1 = metadata["reference_dimensionless_spin1"]
spin2 = metadata["reference_dimensionless_spin2"]
forb = metadata["reference_orbital_frequency"]
#print(forb)
fmag = np.sqrt(forb[0]**2+forb[1]**2+forb[2]**2) #np.linalg.norm(forb)
frefHz = get_fref(fmag,70)
fmin = frefHz

#hlm_modes_given_param(qratio, Mtotal, spin1[0], spin1[1], spin1[2], spin2[0], spin2[1], spin2[2], frefHz, fmin, approximant='SpinTaylorT4', lmax=4, dist=300, model_name='PN')



T_NR, hlm_NR, T_MD, hlm_MD = get_hlm_NR_MD_from_data('SXS','PN')
Angles_NR, Angles_MD = get_Angels_NR_MD(hlm_NR, hlm_MD)

t0, phi0, psi0 = get_best_OptimizedResults(T_NR, T_MD , hlm_NR, hlm_MD, Angles_NR,Angles_MD,T_for_RigRot,10)

get_hyb_modes(T_NR, hlm_NR, T_MD, hlm_MD, Angles_NR, Angles_MD, t0, phi0, psi0, T_for_RigRot)
