from __future__ import print_function
import h5py
import numpy as np


from sympy.physics.units import G, kg, s, m, au, year
import scipy.constants 
from numpy import pi
import numpy as np
import lalsimulation as lalsim
import RIFT.lalsimutils as lalsimutils
import lal
import matplotlib.pyplot as plt 
from scipy.optimize import minimize
from waveform_utils import *


parsec = 648000/pi * au
megaparsec = 10**6 * parsec
msun = 4*pi**2*au**3/G/year**2

c=scipy.constants.speed_of_light*m/s
T_for_RigRot, T_Hybrid_Interval_Start, T_Hybrid_Interval_End = -1.6, -1.6, -0.8
#print (G)

def GeometricTime_To_MKS_Time(t_over_m, mass):
  """
  t_over_m = numpy array with t/M in geometric units
  mass = total mass. Should have units recognized by sympy

  Return value is in units of seconds but the units are removed.
  """

  factor = np.float64(((G*mass/c**3/s)).simplify())
  print ("factor= ",factor)
  
  return t_over_m * factor

def GeometricStrain_TO_Observer_Strain(r_h_over_m, mass, distance):
  """
  r_h_over_m = numpy array with strain in geometric units
  mass = total mass. Should have units recognized by sympy
  distance = distance to source. Should have units recognized by sympy

  Return value is strain at observer location
  """
  factor = np.float64((G*mass/c**2 / distance).n().simplify())
  #print (factor)
  return r_h_over_m  * factor

def Planktaper(t, t1, t2, t3, t4):
  """
  Given time series of waveform compute the function that can deal with gibbs 
  phenominon for ringdown and initial junk
  """
  assert (t1 < t2 and t2 < t3 and t3 < t4)
  if (t <= t1):
    return 0
  if (t1 < t< t2):
    y = (t2 - t1)/(t-t1) + (t2-t1)/(t-t2)
    if y > 0:
      return np.exp(-y)/(1+ np.exp(-y))
    else:
      return 1.0/(1+ np.exp(y))
  if(t2 <= t <= t3):
    return 1
  if(t3 < t < t4):
    z = (t3-t4)/(t-t3) +  (t3-t4)/(t-t4)  #removes negative sign in first term
    return 1e0/(1+ np.exp(z))
  if (t >= t4):
    return 0
  return t





def mag(x,y,z):
    return  np.sqrt(x**2 + y**2 +z**2)

def get_fref(f, Mtot):
    """
    given f22 mode initial freq in natural units: not orbital freq
    and total mass get freq in Hz
    """
    MsunInSec = lal.MSUN_SI*lal.G_SI/lal.C_SI**3
    return f/(MsunInSec*Mtot)/(np.pi)



def taper(waveform):
    for l in range(2,5):
        for m in range(-l,l+1):

            d1 = np.loadtxt(waveform+"{0}{1}".format(l,m)).T
            t1,Re1, im1 = d1[0],d1[1], d1[2]
            z = Re1 + im1*1j
            newz = Waveform_afterWindowing(t1, z, -11.5,0.0)
            t2, Re2, im2  = t1 ,newz.real, newz.imag
            np.savetxt('cleanSXS{0}{1}'.format(l,m),np.c_[t2,Re2,im2])
    
    return 0


def hlm_modes_given_param(q, Mtot, s1x, s1y, s1z, s2x, s2y, s2z, fref, fmin, approximant=lalsim.GetApproximantFromString('SpinTaylorT4'), lmax=4, dist=300, model_name='PN'):
    """
    This function use intrinsic params for waveforms and the waveform approximant and save modes as well as plot them
    q = mass ratio
    Mtot = total mass 
       from q and Mtotal we will get m1, m2 for BHs 
    s1, s2 are spin components try aligned spin first
    fref is the reference frequecny at which black holes are along x-axis and 
    angular momentum is along z axis
    fmin is where we want to start waveforms
       note that fref and fmin are crucial.
    approximant are what model we want for waveforms
       check lalsimulation.  for all models [nonspin, align spin, precessing, now eccentric]
    lmax  = highest  spherical harmonics
    dist = distance of BBH system from detector
    modename is for housekeeping

    returns hlm modes given approximant and approriate model name
    also plot 22, 21, 33 and 44 modes
    """
    #get masses from q and Mtotal
    m1 = q/(1.+q)*Mtot
    m2 = 1./(1.+q)*Mtot
    m1_SI = m1*lal.MSUN_SI #convert into SI units (kg)
    m2_SI = m2*lal.MSUN_SI
    # for how many data points in waveforms we need
    T_window=32.
    #this is default for detector
    fmaxSNR=1700
    # for lalsimutils we need a param input way use ChooseWaveformParams utility
    P = lalsimutils.ChooseWaveformParams()
    P.deltaF = 1./T_window
    P.deltaT = 1./16384.
    P.incl = 0.735
    P.phiref=0.0
    P.m1 = m1_SI #46.6
    P.m2 = m2_SI #23.3
    P.s1x = s1x  
    P.s1y = s1y 
    P.s1z = s1z 
    P.s2x = s2x 
    P.s2y = s2y 
    P.s2z = s2z 
    P.fmin = fmin
    P.fref = fref
    P.dist = dist*lal.PC_SI*1.e6
    P.lambda1=0.0
    P.lambda2=0.0
    P.ampO=-1
    P.phaseO=7
    P.approx=lalsim.GetApproximantFromString(approximant)
    print("P.approx = ", P.approx)
    P.theta=0.0 
    P.phi=0.0
    P.psi=0.0
    P.tref=0.0
    P.radec=False
    P.detector='H1'
    extra_params = {}

    hlmT_lal = lalsimutils.hlmoft(P, Lmax=lmax)
    Tmodelvals =  float(hlmT_lal[(2,2)].epoch)+np.arange(len(hlmT_lal[(2,2)].data.data))*hlmT_lal[(2,2)].deltaT
    argzeros = np.argwhere(hlmT_lal[(2,2)].data.data)
    hlm_MD ={}
    T_MD = {}
    T_MD = Tmodelvals[argzeros][:, 0]
    l = []
    m =[]
    amp =[]
    

    for mode in hlmT_lal.keys():
        hlm_MD[mode] = hlmT_lal[mode].data.data[argzeros][:, 0]
        #saving the data in T Re Im columns
        Re, Im = hlm_MD[mode].real, hlm_MD[mode].imag
        
        np.savetxt("{0}{1}{2}".format(model_name, mode[0], mode[1]), np.stack((T_MD, Re, Im), axis=-1)) 
        #plot that mode
        get_modeplot(model_name, mode[0], mode[1])
        l.append(mode[0])
        m.append(mode[1])
        amp.append(np.max(Re))

    l = np.array(l)
    m = np.array(m)
    amp= np.array(amp)
    mode_and_amp = np.array([l,m,amp])
    print(mode_and_amp.T)


    return 0







def sxs_mode(DISTANCE,MASS):

    file = h5py.File("rhOverM_Asymptotic_GeometricUnits_CoM.h5","r")
    hlm=file[u'Extrapolated_N4.dir']
#Tshift using 22
    time22, real22, imag22 =hlm[u'Y_l{0}_m{1}.dat'.format(2,2)][:].T
    time_sec = GeometricTime_To_MKS_Time(time22, MASS)
    newreal22 = GeometricStrain_TO_Observer_Strain(real22, MASS, DISTANCE)
    newimag22 = GeometricStrain_TO_Observer_Strain(imag22, MASS, DISTANCE)
    max_amp_indx = np.argmax(np.absolute( real22 +  imag22 *1j ))
    tshift = time_sec[max_amp_indx]
    dt = 1e0/2**14#time_sec[-1] - time_sec[-2]
    N = int((time_sec[-1] - time_sec[0])/dt)

    Tarray = np.empty(N)
    for i in range(N):
       Tarray[i] = time_sec[0]+ i*dt

    for l in range(2,5):
        for m in range(-l,l+1):
            time, real, imag =hlm[u'Y_l{0}_m{1}.dat'.format(l,m)][:].T
            phys_time = GeometricTime_To_MKS_Time(time, MASS) #- tshift
            NewT = np.interp(Tarray ,phys_time,phys_time) - tshift
            phys_real = GeometricStrain_TO_Observer_Strain(real, MASS,
            DISTANCE)
            NewRe = np.interp(Tarray ,phys_time,phys_real)
            phys_imag = GeometricStrain_TO_Observer_Strain(imag, MASS,
            DISTANCE)
            NewIm = np.interp(Tarray ,phys_time,phys_imag)
            np.savetxt('SXS{0}{1}'.format(l,m), np.stack((NewT,NewRe,NewIm), axis=-1))
    file.close()
    return 0

def get_modeplot(name, l, m):
    data = np.loadtxt(name+"{0}{1}".format(l, m)).T
    T, Re, Im = data[0], data[1], data[2]
    print(l,m,np.max(Re))
    #plotting only real part
    plt.figure(figsize=(8, 3))
    plt.plot(T, Re, label=name+"l{0}_m{1}".format(l,m))
    #plt.plot(T, Im, label="hl{0}_m{1}".format(l,m))
    plt.xlabel('time')
    plt.ylabel('mode')
    plt.legend()
    plt.show()
    return 0


