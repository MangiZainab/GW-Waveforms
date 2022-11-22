import numpy as np
import matplotlib
from mpl_toolkits.mplot3d import axes3d
matplotlib.use('agg')

import matplotlib.pyplot as plt

import sys
import os
import scipy
from scipy.optimize import minimize
from math import *
from waveform_utils import *


T_for_RigRot, T_Hybrid_Interval_Start, T_Hybrid_Interval_End = -0.7, -0.7, -0.3
lmax = 4
#Data in form that fullcode needed
#Save Into Dictionary now
hlm_NR    = {}
T_NR      = {}
Angles_NR = {}
for l in range(2, lmax+1):
    for m in range(-l, l+1):
        DataNR = np.loadtxt("cleanSXS{0}{1}".format(l,m))
        hlm_NR[(l,m)] = DataNR[:,1] + DataNR[:, 2]*1j
        if (l==2 and m==2):
            T_NR = DataNR[:,0]
#
hlm_MD ={}
T_MD = {}
Angles_MD ={}
for l in range(2, lmax+1):
    for m in range(-l,l+1):
        DataMD = np.loadtxt("PN{0}{1}".format(l,m))
        hlm_MD[(l,m)] = DataMD[:,1] + DataMD[:, 2]*1j
        if (l==2 and m==2):
            T_MD = DataMD[:,0]


###############Step I Get Rotations of Coprecessing Frame #################################
####Tested Coprecess angles function
### Can we do like quaternion rotation here?

alphanr, betanr, gammanr = Coprecess_Angles( lmax, hlm_NR)
Angles_NR['Alpha_NR'] = alphanr.copy()
Angles_NR['Beta_NR'] = betanr.copy()
Angles_NR['Gamma_NR'] = gammanr.copy()
#
alpha_md, beta_md, gamma_md = Coprecess_Angles( lmax, hlm_MD)
Angles_MD['Alpha_MD'] = alpha_md.copy()
Angles_MD['Beta_MD'] = beta_md.copy()
Angles_MD['Gamma_MD'] = gamma_md.copy()


############### Step II   Get data in Hybrid interval   verified it ###################

def get_data_for_hybrid_interval(l, m, t_rigrot, thyb_str, thyb_end, waveNR, waveModel, tnr, tmodel,t0_guess):
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
def OptimizeFunction(x):
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
    Sum = 0
    for l in range(2, lmax+1):
        for m in range(-l, l+1):
            partial = get_data_for_hybrid_interval(l, m, T_for_RigRot, T_Hybrid_Interval_Start, T_Hybrid_Interval_End, hlm_NR, hlm_MD, T_NR, T_MD,t0)
            factor = 1; #20 if m%2==1 else  for m=1 mode dominance
            #Removing contributions from m=0 modes
            if m == 0:
                factor = 0
            Sum += np.sum(factor* np.absolute(partial[0] - partial[1] * np.exp( (m*phi0  + 2*psi0) *1j)))
    return  Sum *1e17#e17 just for getting reasonable number waveform is in such units
#######################################################################################
#def Get_Optimized_Values(xstart,  method='Nelder-Mead' , tol=1e-30):
def Get_Optimized_Values(xstart):
    """
    Given initial guesses and this function use the above 
    defined function to get optimized values for t0, phi0
    One can experiment with different methods 
    check like numpy minimize function for other choices
    tolerance can aso be change

    The best min val and optimized vals will be the output of this function
    """
    opt_result = minimize(OptimizeFunction, xstart, method='Nelder-Mead' , tol=1e-30)
    return opt_result.fun,  opt_result.x[0], opt_result.x[1] , opt_result.x[2]


###########################################################################################
def get_best_OptimizedResults(tnr, tmd, hlmNR22, hlmMD22, tchoose,nchoose):
    """
    This function will try to get the best Optimized 
    results. We will iterate 
    over different initial guesses
    Return will be t0, phi0  best values
    """
    phi0guess = np.linspace(-2*np.pi,2*np.pi, nchoose) # np.array([0.5, 0.01, 3.12, np.pi, np.pi/4e0, np.pi/2e0, 2.987, 3.36])
#timeshift choices
    t0guess =  get_t0guess(hlmNR22, hlmMD22,tnr ,tmd,tchoose)
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
        Func[i] , t0_val[i] , phi0_val[i] , psi0_val[i]= Get_Optimized_Values( np.array([ t0guess[i], phi0guess[i], psi0guess[i]]))
    #print( "array of optimized func vals =" ,Func)
    idx = np.argmin(Func)
    print( Func[idx], t0_val[idx], phi0_val[idx], psi0_val[idx])
    return Func[idx],t0_val[idx], phi0_val[idx] , psi0_val[idx]


func1, t01, phi01, psi1 = get_best_OptimizedResults(T_NR, T_MD, hlm_NR[(2,2)], hlm_MD[(2,2)], T_for_RigRot,10)
func2, t02, phi02, psi2 = get_best_OptimizedResults(T_NR, T_MD, hlm_NR[(2,2)], hlm_MD[(2,2)], T_for_RigRot,20)
sxs = 'SXS:BBH:1346'
data = np.array(zip(sxs,func1,func2), dtype=[('s1', 'S16'),('func1', float), ('func2', float)])
print(data)
