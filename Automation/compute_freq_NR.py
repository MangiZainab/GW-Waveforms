

from sympy.physics.units import G, kg, s, m, au, year
from numpy import pi
import numpy as np
from math import *
import matplotlib.pyplot as plt
import sys
from waveform_utils import Rescale_data_in_Total_Mass
#from waveform_utils import RescaleTimes
import sxs
import lal 



def plot_freq(T, Z, M1, M2,t1, t2):
    """
    return plot of freq for a Mtot
    include the vertical line describing
    hybrid interval
    M1 = standard mass of waveform
    M2 = needd total mass
    t1 = time of start of the hyb interval
    t2 = time of end of the hybrid interval
    """
    #rescale waveform and time using waveform_utils
    rescale_T, rescale_Z = Rescale_data_in_Total_Mass(M1, M2, T, Z)

#    newt1 = RescaleTime(t1, M1, M2)
#    newt2 = RescaleTime(t2, M1, M2)
    Newfreq = freq(rescale_T, rescale_Z)
    plt.figure(figsize=(10, 8))
    plt.plot(rescale_T,Newfreq, label="M={0}".format(M2))
#    plt.axvline(x= newt1, color='r')
#    plt.axvline(x= newt2, color='r')
    plt.xlabel("T[sec]")
    plt.ylabel("Freq[Hz]")
    plt.legend(loc= 2, fontsize=20)
    plt.show()

def freq(T, Z):
    """
    from unwraped phase get the frequecy
    This is omega/2pi
    """
    dt = T[1] -T[0]
    phase = np.angle(Z)
    unwrapphase = np.unwrap(phase)
    omega = 1e0/dt*(np.gradient(unwrapphase))
    return np.abs(omega)/(2*np.pi)


def mag(x,y,z):
    return  np.sqrt(x**2 + y**2 +z**2)

def get_fref(f, Mtot):
    """
    given f22 mode initial freq in natural units: not orbital freq
    and total mass get freq in Hz
    """
    MsunInSec = lal.MSUN_SI*lal.G_SI/lal.C_SI**3
    return f / (MsunInSec * Mtot)/np.pi

waveformnumber = "0123"
metadata = sxs.load("SXS:BBH:"+waveformnumber+"/Lev/metadata.json")
forb = metadata["reference_orbital_frequency"]



#compute fref through orbital frequency from metadata file
fmag = mag(forb[0], forb[1],forb[2])
fref = get_fref(fmag,70)
print(fref)

#compute freq of NR waveform for 22mode
d = np.loadtxt("SXS22").T
t,Z = d[0],d[1]+ d[2]*1j

f = freq(t,Z)
print(f[0])
