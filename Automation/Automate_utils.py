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


def get_hlm_NR_MD_from_data(NR= 'SXS',PN='PN'):
    """ 
    Given PN and NR waveform file name, this function reads 
    and stores time and hlm mode values from data files
    """


    hlm_NR    = {}
    T_NR      = {}
    lmax = 4
    for l in range(2, lmax+1):
        for m in range(-l, l+1):
            DataNR = np.loadtxt(NR+"{0}{1}".format(l,m))
            hlm_NR[(l,m)] = DataNR[:,1] + DataNR[:, 2]*1j
            if (l==2 and m==2):
                T_NR = DataNR[:,0]
#
    hlm_MD ={}
    T_MD = {}
    for l in range(2, lmax+1):
        for m in range(-l,l+1):
            DataMD = np.loadtxt(PN+"{0}{1}".format(l,m))
            hlm_MD[(l,m)] = DataMD[:,1] + DataMD[:, 2]*1j
            if (l==2 and m==2):
               T_MD = DataMD[:,0]
    return T_NR,hlm_NR,T_MD, hlm_MD


def get_Angels_NR_MD(hlm_NR, hlm_MD):
    """
    Given hlm_NR and hlm_MD, this function 
    calculates Coprecessing Angles
    
    """
    Angles_NR = {}
    Angles_MD = {}
    lmax = 4
    alphanr, betanr, gammanr = Coprecess_Angles( lmax, hlm_NR)
    Angles_NR['Alpha_NR'] = alphanr.copy()
    Angles_NR['Beta_NR'] = betanr.copy()
    Angles_NR['Gamma_NR'] = gammanr.copy()
#
    alpha_md, beta_md, gamma_md = Coprecess_Angles( lmax, hlm_MD)
    Angles_MD['Alpha_MD'] = alpha_md.copy()
    Angles_MD['Beta_MD'] = beta_md.copy()
    Angles_MD['Gamma_MD'] = gamma_md.copy()


    return Angles_NR ,Angles_MD


def get_data_for_hybrid_interval(l, m, t_rigrot, thyb_str, thyb_end, waveNR, waveModel, tnr, tmodel,Angles_NR,Angles_MD,t0_guess):
    """
    This is the important function that gives the data
    to be used for optimization for best choice of t, phi
    shifts to align waveforms in hybrid interval [thyb_str, thyb_end]
    t_rigrot = time for rigid rotation of two waveforms
      This can be chosen arbitrarily but must be the time in NR 
    waveform
    t0guess: Based on max freq match in hybrid interval.
    """
    startnr, endnr = np.searchsorted(tnr, (thyb_str, thyb_end)) #finds index of start and end of hybrid interval
    startnr = startnr -1
    tnr = tnr[startnr:endnr] # tnr at hybrid interval
    alpha_nr = np.interp(t_rigrot , tnr , Angles_NR['Alpha_NR'][startnr:endnr]) # alpha_nr at rigid rotation time.
    beta_nr  = np.interp(t_rigrot , tnr , Angles_NR['Beta_NR'][startnr:endnr])  # beta_nr ......
    gamma_nr = np.interp(t_rigrot , tnr , Angles_NR['Gamma_NR'][startnr:endnr]) #gamma_nr ......
 
    alpha_model  =  np.interp(t_rigrot - t0_guess, tmodel , Angles_MD['Alpha_MD']) 
    beta_model   =  np.interp(t_rigrot - t0_guess, tmodel , Angles_MD['Beta_MD'])
    gamma_model  =  np.interp(t_rigrot - t0_guess, tmodel , Angles_MD['Gamma_MD'])
    #For non-precessing case angles = 0
    #alpha_nr, beta_nr,gamma_nr , alpha_model, beta_model, gamma_model = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    NRhlm_hyb = Rigid_rotate_waveform(l, m, alpha_nr, beta_nr, gamma_nr, waveNR)[startnr:endnr]
    #print(len(NRhlm_hyb), len(tnr))
    MDhlm_hyb = T_interp_Rigid_rotate_waveform(l, m, alpha_model, beta_model, gamma_model, waveModel,  t0_guess, tnr, tmodel)

    return  NRhlm_hyb, MDhlm_hyb #,tnr  if we want to test it


##########################################################################
############ Function For optimization #####################
def OptimizeFunction(x,T_NR,hlm_NR,T_MD,hlm_MD,Angles_NR,Angles_MD):
    """
    Given the x = np.array( [t0guess, phi0suess])
    and get get_data_for_hybrid_interval(l, m, t_rigrot, thyb_str, thyb_end, waveNR, waveModel, tnr, tmodel,t0_guess)
    which are actual data  sets from dictionary used
    This function will find the time and phase shifts
    time shifts and phase shifts can be chosen in many number of ways 
    phi = -2pi to +2pi
    t0 as we are using seconds so must must be larger than 0.001? not sure
    """
    t0 = x[0]
    phi0 = x[1]
    psi0 = x[2]
    lmax = 4
    Sum = 0
    T_for_RigRot, T_Hybrid_Interval_Start, T_Hybrid_Interval_End = -0.7, -0.7, -0.3
    for l in range(2, lmax+1):
        for m in range(-l, l+1):
            partial = get_data_for_hybrid_interval(l, m, T_for_RigRot, T_Hybrid_Interval_Start, T_Hybrid_Interval_End, hlm_NR, hlm_MD, T_NR, T_MD,Angles_NR,Angles_MD,t0)
            factor = 1; #20 if m%2==1 else  for m=1 mode dominance
            #Removing contributions from m=0 modes
            if m == 0:
                factor = 0
            Sum += np.sum(factor* np.absolute(partial[0] - partial[1] * np.exp( (m*phi0  + 2*psi0) *1j)))
    return  Sum *1e17#e17 just for getting reasonable number waveform is in such units


#######################################################################################
#def Get_Optimized_Values(xstart,  method='Nelder-Mead' , tol=1e-30):
def Get_Optimized_Values(xstart,T_NR,hlm_NR,T_MD,hlm_MD,Angles_NR,Angles_MD):
    """
    Given initial guesses and this function use the above 
    defined function to get optimized values for t0, phi0
    One can experiment with different methods 
    check like numpy minimize function for other choices
    tolerance can aso be change
get_hlm_NR_MD_from_data(NR= 'cleanSXS',PN='PN')
    The best min val and optimized vals will be the output of this function
    """
    opt_result = minimize(OptimizeFunction, xstart,args = (T_NR,hlm_NR,T_MD,hlm_MD,Angles_NR,Angles_MD), method='Nelder-Mead' , tol=1e-30)
    return opt_result.fun,  opt_result.x[0], opt_result.x[1] , opt_result.x[2]


###########################################################################################
def get_best_OptimizedResults(T_NR, T_MD , hlm_NR, hlm_MD, Angles_NR,Angles_MD,tchoose,nchoose):
    """
    This function will try to get the best Optimized 
    results. We will iterate 
    over different initial guesses
    Return will be t0, phi0  best values
    """
    hlmNR22 = hlm_NR[(2,2)]
    hlmMD22 = hlm_MD[(2,2)]
    phi0guess = np.linspace(-2*np.pi,2*np.pi, nchoose) # np.array([0.5, 0.01, 3.12, np.pi, np.pi/4e0, np.pi/2e0, 2.987, 3.36])
#timeshift choices
    t0guess =  get_t0guess(hlmNR22, hlmMD22,T_NR,T_MD,tchoose)
    tmin, tmax = -t0guess, t0guess
    #t0guess, tmin, tmax = t0_guess_Vijay(tnr, tmd, hlmNR22, hlmMD22, tchoose)
    alen = len(phi0guess)
    t0guess = np.linspace(tmin, tmax,alen)
    psi0 = np.array([0, np.pi/2])
    psi0guess= np.tile(psi0, int(alen/2)) #one must have alen to be even



   ###Optimization function ###########
    Func = np.zeros(alen, dtype=np.float64)
    t0_val  = np.zeros(alen, dtype=np.float64)
    phi0_val = np.zeros(alen, dtype=np.float64)
    psi0_val = np.zeros(alen, dtype=np.float64)

    for i in range(alen):
        Func[i] , t0_val[i] , phi0_val[i] , psi0_val[i]= Get_Optimized_Values(np.array([ t0guess[i], phi0guess[i], psi0guess[i]]),T_NR,hlm_NR,T_MD,hlm_MD,Angles_NR,Angles_MD)
    print( "array of optimized func vals =" ,Func)
    idx = np.argmin(Func)
    print( Func[idx], t0_val[idx], phi0_val[idx], psi0_val[idx]) # put if condition on t0 
    return t0_val[idx], phi0_val[idx] , psi0_val[idx]



