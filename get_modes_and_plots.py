#python get_modes_and_plots.py

import numpy as np
import lalsimulation as lalsim
import  RIFT.lalsimutils as lalsimutils
import lal
import matplotlib.pyplot as plt

def mag(x,y,z):
    return  np.sqrt(x**2 + y**2 +z**2)

def get_fref(f, Mtot):
    """
    given f22 mode initial freq in natural units: not orbital freq
    and total mass get freq in Hz
    """
    MsunInSec = lal.MSUN_SI*lal.G_SI/lal.C_SI**3
    return f / (MsunInSec * Mtot)/np.pi

fmag = mag(0.000134130951557, -0.000158548342344, 0.0145160312287)
frefHz = get_fref(fmag,70)

print(frefHz)



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
    #for l in range(2, 5):
    #    for m in range(-l, l+1):
    #        Re, Im = hlm_MD[(l,m)].real, hlm_MD[(l,m)].imag
    #        np.savetxt("{0}{1}{2}".format(model_name, l, m), np.stack((T_MD, Re, Im), axis=-1))

    #now data is saved so plot 22 21 33 44 mode using function of plot modes
    #get_modeplot(model_name, 2, 2) 
    #get_modeplot(model_name, 2, 1) 
    #get_modeplot(model_name, 3, 3) 
    #get_modeplot(model_name, 4, 4)
    l = np.array(l)
    m = np.array(m)
    amp= np.array(amp)
    mode_and_amp = np.array([l,m,amp])
    print(mode_and_amp.T)


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
   


#tests an example with plots:
#Try a Simple Aligned spin case mass ratio of 2
mass1    = 0.573607077412
mass2    = 0.426442587444

qratio = mass1/mass2
Mtotal = 70


spin1x, spin1y, spin1z = 6.8205135257944303,   0.0008077050752341,   0.0000000000000000
spin2x, spin2y, spin2z = -9.1794864742055697,   0.0008077050752341,   0.0000000000000000


print("minfrefNRSur = ",frefHz)
fmin = 0
frefHz = 20
# get PN spinTaylorT4 modes
#hlm_modes_given_param(qratio, Mtotal, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z, frefHz, fmin, approximant='SpinTaylorT4', lmax=4, dist=300, model_name='PN')

#SEOB modes
#hlm_modes_given_param(qratio, Mtotal, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z, frefHz, fmin, approximant='SEOBNRv4PHM', lmax=4, dist=300, model_name='SEOB')

#...................NewNR params..............................
#qratio = 0.666671367139 / 0.33332605284
#Mtotal = 70.0

#spin1x = -0.077182092184
#spin1y = 0.594997453288
#spin1z = 0.00288300874193

#spin2x = -0.136999903231
#spin2y = 0.584144814823
#spin2z = 0.00586454523012

#frefHz = 8.490409668550074
#frefHz = 5.0 
print("minfrefNRSur = ",frefHz)
#fmin =  7.4
print(spin2x)
hlm_modes_given_param(qratio, Mtotal, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z, frefHz, fmin, approximant='NRSur7dq4', lmax=4, dist=300, model_name='NRSur')
