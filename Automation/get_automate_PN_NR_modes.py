#python get_automate_PN_NR_modes.py --waveformnumber 0123

import numpy as np
import matplotlib.pyplot as plt

from Automate_utils import *
from get_sxs import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--waveformnumber", type=str, help="waveform number")
opts = parser.parse_args() 



DISTANCE=300*megaparsec
MASS = 70 * msun

################## Get PN modes from sxs  ################ 

waveformnumber = opts.waveformnumber
get_modes_from_sxs(waveformnumber, MASS=70, DISTANCE=300*megaparsec)




#################### Get NR mode #############
# we will implement automation by using argparser for parameters but below  




metadata = sxs.load("SXS:BBH:"+waveformnumber+"/Lev/metadata.json")
qratio = metadata["reference_mass_ratio"]
Mtotal = 70
spin1 = metadata["reference_dimensionless_spin1"]
spin2 = metadata["reference_dimensionless_spin2"]
forb = metadata["reference_orbital_frequency"]

fmag = np.linalg.norm(forb)
frefHz = get_fref(fmag,70)
fmin = frefHz
hlm_modes_given_param(qratio, Mtotal, spin1[0], spin1[1], spin1[2], spin2[0], spin2[1], spin2[2], frefHz, fmin, approximant='SpinTaylorT4', lmax=4, dist=300, model_name='PN')
