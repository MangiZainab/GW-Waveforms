# Given the data of two wave form. This program will generate figures for each Mtot
#1: Plots of frequency as function of T for two waves but also it shows where the hybrid interval was  for each cae
#2: It also give plots for mismatch of strain 
#How to st the colormap limits? comment what mass is not interestable as for large total mass out LIGO bucket will only contains the NR data.

# Hybrid interval with Mtot =70.  define a function which change these ts for new Mtota

import numpy as np
import lal
import lalsimulation as lalsim
import matplotlib.pyplot as plt
import lalsimutils
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
from waveform_utils import *
plt.style.use("./presentation.mplstyle")
tstr, tend = -0.5, -0.07#-1.5, -0.4



def RescaleTime(T, Mused, Mneed):
    """
    Return the rescaled time for new M_total 
    """
    return T*float(Mneed/Mused)



# Data enter
lmax = 4
Exc ={}
Hyb = {}
Thyb ={}
# Assuming we already interpolated the two data sets onto same time grid
for l in range(2, lmax+1):
    for m in range(-l, l+1):
        DataExc = np.loadtxt("RigrotatedEX{0}{1}".format(l,m))
#        DataHyb = np.loadtxt("MD_and_NR_Hybrid{0}{1}".format(l,m))
#        DataHyb = np.loadtxt("ZeroNR40data{0}{1}".format(l,m))
        DataHyb = np.loadtxt("ZeroNR20data{0}{1}".format(l,m))

        ZExc = DataExc[:,1] + DataExc[:,2]*1j
        TExc = DataExc[:,0]
        IntrpDataEX = np.interp(DataHyb[:,0], TExc, ZExc)
        Exc[(l,m)] = IntrpDataEX.copy()

        Hyb[(l,m)] = DataHyb[:,1] + DataHyb[:,2]*1j
        if(l==2 and m==2):
            Thyb = DataHyb[:,0]
            
# compute the  Frequencies
def freq(T, Z):
    """
    from unwraped phase get the frequecy
    This is omega/2pi
    """
    dt = T[1] -T[0]
    phase = np.angle(Z)
    unwrapphase = np.unwrap(phase)
    omega = 1e0/dt*(np.gradient(unwrapphase))
    return np.abs(omega)/2./np.pi

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
    newt1 = RescaleTime(t1, M1, M2)
    newt2 = RescaleTime(t2, M1, M2)
    Newfreq = freq(rescale_T, rescale_Z)
    tindx = np.searchsorted(Newfreq, 20.)
    print(cycles(rescale_Z),  cycles(rescale_Z[tindx:]))
    #plt.figure(figsize=(10, 8))
    #plt.plot(rescale_T,Newfreq, label="M={0}".format(M2))
    #plt.axvline(x= newt1, color='r')
    #plt.axvline(x= newt2, color='r')
    #plt.xlabel("T[sec]")
    #plt.ylabel("Freq[Hz]")
    #plt.legend(loc= 2, fontsize=20)
    #plt.show()
    return int(abs(cycles(rescale_Z))),  int(abs(cycles(rescale_Z[tindx:])))

def Mismatch_from_hlm(Time, h1, h2):
  """
  h1 and h2 are complex time series
  Assuming the two waveforms
  have same time and dt
  If not interp and make them 
  of same length.
  For Now above part of code 
  can do this
  """
  deltaT = Time[1] - Time[0]
  npts = len(Time)
  #Creating h data structure
  hT1 = lal.CreateCOMPLEX16TimeSeries("Complex overlap",lal.LIGOTimeGPS(0.), 0., deltaT, lal.DimensionlessUnit, npts)
  hT1.data.data = h1
  hT2 = lal.CreateCOMPLEX16TimeSeries("Complex overlap",lal.LIGOTimeGPS(0.), 0., deltaT, lal.DimensionlessUnit, npts)
  hT2.data.data = h2


   # ZeroPadding
  pow2npts = nextPow2(npts)
  hT1 = lal.ResizeCOMPLEX16TimeSeries(hT1, 0, pow2npts)
  hT2 = lal.ResizeCOMPLEX16TimeSeries(hT2, 0, pow2npts)


  # Create fourier transforms
  hF1 = lalsimutils.DataFourier(hT1)
  hF2 = lalsimutils.DataFourier(hT2)


  # Create inner product
  psd=lalsim.SimNoisePSDaLIGOZeroDetHighPower
  #IP = lalsimutils.CreateCompatibleComplexIP(hF1,psd=psd,fLow=20,fMax=1700,analyticPSD_Q=True)
  IP = lalsimutils.CreateCompatibleComplexOverlap(hF1,psd=psd,fLow=20,fMax=2000,analyticPSD_Q=True,interpolate_max=True)
  rho_1 = IP.norm(hF1)
  rho_2 = IP.norm(hF2)
  inner_12 = IP.ip(hF1,hF2)/rho_1/rho_2
  return 1.0 - inner_12, rho_1





def get_plots_mismatch__freq_over_TotMass(Time, h1, h2, MtotUsed,l,m,t1,t2):
  """
 Given h1, h2 with same dt, and time
 get fft compute mismtach
 Rescales the to mass and get new missmatch
 Plot the final mismatch for each mass 
 Right now I am not using any windowing on data
Just rescaling by mass
  """
  Masschange = np.linspace(10,150,15)
  #Masschange = np.array([70., 40.]) #np.linspace(10,150,15)
  Mtot = []#np.zeros_like(Masschange)
  Mismatch =[]# np.zeros_like(Masschange)
  for k in Masschange:
    Rescsalet1, Rescaleh1 = Rescale_data_in_Total_Mass(MtotUsed,k , Time, h1)
    Rescsalet2, Rescaleh2 = Rescale_data_in_Total_Mass(MtotUsed,k , Time, h2)
    mismatch_i, rhoIP_i =  Mismatch_from_hlm(Rescsalet1, Rescaleh1 ,Rescaleh2 )
    print (k, mismatch_i)
    Mtot.append(k)
    Mismatch.append(mismatch_i)
  #data
    #outfile= open("MismatchData{0}{1}".format(l,m), "a")
    #outfile= open("MismatchData20{0}{1}".format(l,m), "a")
    #outfile= open("MismatchData40{0}{1}".format(l,m), "a")
    #outfile.write("{0} ".format(k))
    #outfile.write("{0:20.16e} ".format(mismatch_i))
    #outfile.write("{0:20.16e} ".format(rhoIP_i))
    #outfile.write("\n")
    #outfile.close()    

  print Mismatch
  #plt.figure(figsize=(10, 8))
  ax1.plot(Mtot, Mismatch, linestyle='--', marker='o', label= "$\ell = {0}, m ={1}$".format(l,m))
  #plt.title("$\ell = {0}, m ={1}$".format(l,m))
  #plt.grid()
  ax1.set_xlabel('$\mathrm{M}_{tot}$', fontsize = 30)
  ax1.set_ylabel("$\mathcal{M}$", fontsize =30)
  #plt.semilogy()
  #plt.savefig("/home/jxs1805/GIT/JamSadiq/HybridWaveforms/HybridWaveforms/Paper/figs/MismatchLogl{0}m{1}SXS1412.png".format(l,m))
  #plt.show()
  return l, m, Mtot, Mismatch 




fig = plt.figure(figsize=(14, 12))
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny()

#taper and zeropadding data and mismatch  Plots freq, mismatch for each Mtot with modeby mode
for l in range(2,5):
    for m in range(-l,l+1):
        #if  m== 0:
        if  m <= 0:
            print("m=0 or neg")
        else:

            Tjunk,Tringdown = -11.2, 0.0
            wave1 = Waveform_afterWindowing(Thyb, Exc[(l,m)], Tjunk,Tringdown)
            wave2 = Waveform_afterWindowing(Thyb, Hyb[(l,m)], Tjunk,Tringdown)

            l, m, Tptotalmass, Tptotmismatch= get_plots_mismatch__freq_over_TotMass(Thyb, wave1, wave2 ,70 , l, m, tstr, tend)
            print ("l=", l , "m=", m, "Mtot", Tptotalmass, "mismatch for l{0},m{1}".format(l,m), Tptotmismatch)

Tjunk,Tringdown = -11.2, 0.0
wave22 = Waveform_afterWindowing(Thyb, Exc[(2,2)], Tjunk,Tringdown)
Masschange = np.linspace(10,150,15)
fref20Hz = []
for k in Masschange:
    fref1, fref2 = plot_freq(Thyb, wave22, 70.,k ,tstr, tend)
    fref20Hz.append(fref2) 

def EM(H):
    Sum1 = 0e0
    Sum2 = 0e0
    M = []
    EMRic =[]
    for l in range(2, 5):
        for m in range(-l, l+1):
            if  (m ==0):
                print ("m = 0")
            else:
                data = np.loadtxt("MismatchData20{0}{1}".format(l,m))
                Mass = data[H][0]
                Sum1 += data[H][2]*data[H][1]
                Sum2 += data[H][2]
                EMRic = Sum1/Sum2
    return Mass, EMRic




EMmass = []
EMfunc = []
for k in range(15):
    EMmass.append(EM(k)[0])
    EMfunc.append(EM(k)[1])

X = np.asarray(EMmass)


ax1.plot(EMmass,EMfunc, "k", label="WM", linewidth=4)
ax1.legend(fontsize=28)
plt.semilogy()
#ax1.set_title('$q=$'' 5, Non-Spinning, PN-NRHybridvEOB', fontsize=22, color='k')
ax1.grid()
new_tick_locations = np.array([20, 40, 60, 80, 100, 120, 140])
ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks(new_tick_locations)
ax2.set_xticklabels(np.asarray(fref20Hz[1:-1:2]))
#ax2.set_xlabel(r"Cycles before merger", fontsize =26)
#ax2.set_xlabel('$q=$'' 5, Non-Spinning, PN-NRHybridvEOB', fontsize=30, color='k')
ax1.set_title("(SXS:BBH:1410) NRvsTruncatedNR (20 cycles)", fontsize=28, color='k')
ax2.text(1,0.8, "$N_{cycles}$", fontsize = 25)
ax2.tick_params(axis="x",direction="in", pad=-20)
plt.semilogy()
plt.show()

# Mismatch with strain
#Plots freq,  strain 


