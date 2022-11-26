#Jam Sadiq
#May 10, 2020
#Useful functions for precessing hybrids

import numpy as np
import matplotlib.pyplot as plt
from math import *
from numpy import linalg as LA
import lal
import lalsimulation as lalsim
import RIFT.lalsimutils as lalsimutils


def SphericalPolarAngles(v):
  """
  Given a 3D vector v, this function
  calculates its polar and azimuthal
  orientation. Returns (Theta, Phi)
  Theta = arccos (v[2]);
  Phi = atan2(v[1], v[0]);
  """
  norm = np.sqrt(np.dot(v,v))
  Theta = np.arccos(v[2] / norm) #theta ==>-beta (Schmidt et al)
  Phi = np.arctan2(v[1], v[0]);
  return Theta, Phi
#######################################################################
##### Lab tensor fo two Euler Rotations of Coprecessing frame##########
#######################################################################
#The average orientation tensor for preferred direction
def clm(l,m):
  """
  Prefactors in definitions of average orientation tensor
  given in Appendex of arXive 1205.2287v1
  l,m are related to modes 
  """
  if (m > l or m < -l ):
    return  0
  return sqrt(l*(l+1) - m*(m+1))

def I0(lmax, waveform): 
  """
  lmax = max modes l (2 or more)
  waveform is a dictionary with keys (l,m) of modes
  One of the terms in average orientation tensor 
  given in Appendex of arXive 1205.2287v1
  This will be a real quantity
  """
  Sum = 0.
  for l in range(2, lmax + 1):
    for m in range(-l, l+1):
      Sum += (l*(l+1) -m**2)* (abs(waveform[(l,m)])**2)
  return (1./2.)*Sum

def I1(lmax, waveform):
  """
  One of the terms in average orientation tensor 
  given in Appendex of arXive 1205.2287v1
  This will be a complex quantity
  """
  Sum =0.
  for l in range(2,lmax + 1):
    for m in range(-l, l):  #to avoid psi(l, m+1 , m+1 > l case)
      Sum += clm(l,m)*(m+1./2)*waveform[(l,m)]* waveform[(l,m+1)].conjugate()
  return Sum

def I2(lmax, waveform):
  """
  One of the terms in average orientation tensor 
  given in Appendex of arXive 1205.2287v1
  This will be a complex quantity
  """
  Sum = 0.
  for l in range(2,lmax + 1):
    for m in range(-l, l-1): #to avoid psi(l, m+1 , m+2 > l case)
      Sum += clm(l,m)*clm(l,m+1)*waveform[(l,m)]* waveform[(l,m+2)].conjugate()
  return (1./2.)*Sum

def Izz(lmax, waveform):
  """
  One of the terms in average orientation tensor 
  given in Appendex of arXive 1205.2287v1
  This will be a real quantity
  """
  Sum = 0.
  for l in range(2,lmax + 1):
    for m in range(-l, l+1):
      Sum += m**2 * (abs(waveform[(l,m)])**2)
  return Sum

def Lab(lmax, waveform):
  """
  A 3 by 3 Matrix 
  Based on average orientation tensor 
  in appendix A of arXiv:1304.3176
  Purpose of Lab is to construct a 3 by 3 matrix whose dominant  
  eigen vector provides two Euler angles that can be used to  
  rotate the waveform into a frame where radiation is emitted 
  along z direction such that waveform behave in this frame 
  essentially as a non-precessing wavefvorm. 
  The matrix is symmetric where it has all components 
  like lxx, lxy, lxz etc are scalars
  """

  denom = 0.
  for l in range(2,lmax + 1):
    for m in range(-l, l+1):
      denom += (abs(waveform[(l,m)])**2)

  lxx = 1.0/denom *(I0(lmax, waveform) + I2(lmax, waveform).real)
  lyy = 1.0/denom *(I0(lmax, waveform) - I2(lmax, waveform).real)
  lzz = 1.0/denom * Izz(lmax, waveform)
  lxy = 1.0/denom * I2(lmax, waveform).imag
  lxz = 1.0/denom * I1(lmax, waveform).real
  lyz = 1.0/denom * I1(lmax, waveform).imag

  M = [[0]*3 for i in range(3)]
  M[0][0] = lxx
  M[0][1] = lxy
  M[0][2] = lxz
  M[1][0] = lxy
  M[1][1] = lyy
  M[1][2] = lyz
  M[2][0] = lxz
  M[2][1] = lyz
  M[2][2] = lzz

  return M

def Eigvectors(lmax, waveform):
  """
  lmax=  max val of l mode (2 or large)
  waveform is in dictionary format 
  with keys (l,m).

  This function will output the component of
  dominant eigen vectors using Lab matrix 
  defined above. This vector contains the
  information about the orientations of 
  radiation direction.
  Lab is matrix whose eigen vectors
  contains the radiation directions
  For aligned spin cases dominant eigen vector
  is aligned with Z direction
  v1 is called that eigen vector.
  v1 is sorted to be the dominant vector 
  plot it to check if there is precession
  of orbital plane 
  """

  if 'length' in waveform:
    #alen = waveform('length')
    alen = waveform['length']
  else:
    alen = len(waveform[(2,2)])

  v1x = v2x = v3x = np.zeros(alen, dtype=np.float64)
  v1y = v2y = v3y = np.zeros(alen, dtype=np.float64)
  v1z = v2z = v3z = np.zeros(alen, dtype=np.float64)
  
  
  lab = Lab(lmax, waveform)

  axx = lab[0][0]
  axy = lab[0][1]
  axz = lab[0][2]
  ayy = lab[1][1]
  ayz = lab[1][2]
  azz = lab[2][2]

  mat =  [[0]*3 for i in range(3)]
  x = [[0]*3 for i in range(3)]
  d = [0]*3
  v2 = [0]*3
  v3 = [0]*3
  vold = None

  for i in range(alen):
    mat[0][0] = axx[i]
    mat[0][1] = axy[i]
    mat[1][0] = axy[i]
    mat[2][0] = axz[i]
    mat[0][2] = axz[i]
    mat[1][1] = ayy[i]
    mat[1][2] = ayz[i]
    mat[2][1] = ayz[i]
    mat[2][2] = azz[i]

    Amat = np.nan_to_num(np.array(mat))
    eigenValues, eigenVectors = LA.eig(np.array(Amat))  #LA.eig(np.array(mat))
  
    #Since LA.eig dinot sort eigenvalues we 
    #first sort the eigen values and define 
    #indx (0,1,2) which tells which is larger eigenvalue

    idx = eigenValues.argsort()[::-1]
    d = eigenValues[idx]
    x = eigenVectors[:,idx]
    # Note eigenVector[:,0] is the first eigenvector, etc.

    v1 = np.array((x[0][0], x[1][0], x[2][0]))
    v2 = np.array((x[0][1], x[1][1], x[2][1]))
    v3 = np.array((x[0][2], x[1][2], x[2][2]))


    # Note: Want the z-component of the waveform direction initially to
    # be positive. Otherwise. This is an arbitrary choice. For later
    # times, choose sign of the eigenvector to maximize overlap with
    # previous time's eigenvector.
    if ((i==0 and v1[[2]] < 0) or (i> 0 and np.dot(vold, v1)< 0)):
      v1 = -v1

    vold = v1.copy()  #in numpy it points to address so vold is v[i]t v[i-1]
   
    v1x[i], v2x[i], v3x[i]  = v1[0], v2[0], v3[0] 
    v1y[i], v2y[i], v3y[i]  = v1[1], v2[1], v3[1] 
    v1z[i], v2z[i], v3z[i]  = v1[2], v2[2], v3[2] 
                        #remove comment in return for all eigvecs
  return v1x, v1y, v1z  #, v2x, v2y, v2z, v3x, v3y, v3z

###########################################################################
###########################################################################
#Plot of Dom EigVector
def plotDomEigvector(lmax, waveform, time):
    """
    Given waveform modes in dictionary 
    with keys (l,m), time and lmax 
    this function will compute the 
    eigenvectors using Eigenvectors function
    defined above. That is using Lab matrix of 
    Co-precessing Frame. 
    The function will plot the vector before the merger
    and return the values of dominant eigen-vector component .
    """
    v1x, v1y, v1z =  Eigvectors(lmax, waveform)
    merge_indx = np.searchsorted(time, 0.0)
    print("index at t=0 is ", merge_indx)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(v1x[0:merge_indx], v1y[0:merge_indx], v1z[0:merge_indx],c ='r', label ="inertial")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_title("Dominant EigVector")
    ax.legend()
    plt.show()
    return v1x, v1y, v1z
###########################################################################
###### Coprecessing Frame Euler Rotations #################################
###########################################################################
def Coprecess_Angles( lmax, waveform): 
  """
  Using the principle direction of Lab matrix 
  we will compute the two Euler angles 
  for rotating waveform into a co-presssing frame.
  The third Euler angle is obtained using the two given Euler
  angles using Boyle et al integral  in paper
  Arxive   ...
  """

# remove unused variables

  if 'length' in waveform:
    #alen = waveform('length')
    alen = waveform['length']
  else:
    alen = len(waveform[(2,2)])

  Alp = np.zeros(alen, dtype=np.float64)
  Bta = np.zeros(alen, dtype=np.float64)
  Gma = np.zeros(alen, dtype=np.float64)

  lab = Lab(lmax, waveform)
  axx = lab[0][0]
  axy = lab[0][1]
  axz = lab[0][2]
  ayy = lab[1][1]
  ayz = lab[1][2]
  azz = lab[2][2]
  
  mat =  [[0]*3 for i in range(3)]
  x = [[0]*3 for i in range(3)]
  d = [0]*3
  v2 = [0]*3
  v3 = [0]*3
  vold = None
 
  for i in range(alen):
    mat[0][0] = axx[i]
    mat[0][1] = axy[i]
    mat[1][0] = axy[i]
    mat[2][0] = axz[i]
    mat[0][2] = axz[i]
    mat[1][1] = ayy[i]
    mat[1][2] = ayz[i]
    mat[2][1] = ayz[i]
    mat[2][2] = azz[i]
    
    Amat = np.nan_to_num(np.array(mat))#[:, :, 0] #issue here in dict data
    eigenValues, eigenVectors =  LA.eig(np.array(Amat)) #LA.eig(np.array(mat))

# Explain sorting

    idx = eigenValues.argsort()[::-1]   
    d = eigenValues[idx]
    x = eigenVectors[:,idx]  


    # Note eigenVector[:,0] is the first eigenvector, etc.

    v1 = np.array((x[0][0], x[1][0], x[2][0]))
    v2 = np.array((x[0][1], x[1][1], x[2][1]))
    v3 = np.array((x[0][2], x[1][2], x[2][2]))


    # Note: Want the z-component of the waveform direction initiall to
    # be positive. Otherwise. This is an arbitrary choice. For later
    # times, choose sign of the eigenvector to maximize overlap with
    # previous time's eigenvector.
    if ((i==0 and v1[[2]] < 0) or (i> 0 and np.dot(vold, v1)< 0)):
      v1 = -v1


    vold = v1.copy()  #in numpy it points to address so vold is v[i]t v[i-1]


    # Euler angles using dominant eigen vector of coprecessing frame
    beta, gamma = SphericalPolarAngles(v1)
    if (i == 0):
      alpha = - gamma * cos(beta)
      alphabar = 0
    else:
      # Boyle et al paper Eq.
      #alpha  = alphaold - (beta -betaold)*(cos(0.5*(betaold + beta)))
      alphabar = alphabarold - 0.5*(gammaold+gamma)*((beta - betaold))*sin(0.5*(betaold+beta))
      alpha  =  -gamma *cos(beta) + alphabar
    alphabarold = alphabar
    betaold = beta
    gammaold = gamma

    Alp[i] = alpha
    Bta[i] = beta
    Gma[i] = gamma

  return  Alp, Bta, Gma 



def wigner(l, m, mp, alpha, beta, gamma):
  """
  For rotating waveforms Wigner rotations are used. factorial from numpy.
  Based on Patricia Schmidth paper. Verified via M-Boyle numpy module
  """
  fac = np.math.factorial
  term = sqrt(fac(l+m) * fac(l-m)* fac(l+mp) * fac(l-mp))
  MinK = max(0, m-mp)
  MaxK = min(l+m , l-mp)
  Sum = 0.

  for k in range(MinK, MaxK+1):
    Sum = Sum + (sin(beta/2.0)**(2*k +mp-m))*(cos(beta/2.0)**(2*l-2*k-mp+m))*((-1)**(k+m-mp))/(fac(k)*fac(l+m-k)*fac(l-mp-k)*fac(mp-m+k))
  Sum = Sum * term * (cos(mp * gamma + m * alpha) + (sin(mp * gamma + m * alpha) * 1j))
  return Sum


def Fully_rotated_waveform(l, m, alp, bta, gma, waveformModes ,i):
  """
  Given waveform modes in dictionary and coprecessing
  angles we can use Wigner rotatio and  rotate  waveform modes 
  at each iteraion in wigner definition we need
  rotation at each time
  so its a scalar function.
  """
  Sum = 0
  for mp in range(-l, l+1):
    Sum += wigner(l, m, mp, alp, bta, gma)*waveformModes[(l,mp)][i]
  return Sum.real , Sum.imag



def Rigid_rotate_waveform(l, m, alpha, beta, gamma, wave):
  """
 Given  wave as data with dictionary wave[(l, m)]
 it rotate waveform with Euler angles alpha, beta and
 gamma and return the rotated waveform
 This is fixed t independent rotation
  """
  Sum = 0
  for mp in range(-l, l+1):
    Sum += wigner(l, m, mp, alpha, beta, gamma) * wave[(l,mp)]
  return Sum

def T_interp_wave(l, m, wave,  t0, tn, tp):
  """
  This interpolate the Model waveform based on time shiftes
  This is needed for optimizations with time shift done on waveform
  """
  return np.interp(tn-t0, tp, wave[(l,m)])


def T_interp_Rigid_rotate_waveform(l, m, alpha, beta, gamma, wave,  t0, tn, tp):
  """
  Given Model wavefome, the t0 shift guess this function first
  interpolate the wavefomr such that it is time shifted
  Then the way above function of rigid rotation follow
  This is a fixed time rigid rotation.
  """
  Sum = 0
  for mp in range(-l, l+1):
    Sum += wigner(l, m, mp, alpha, beta, gamma)* T_interp_wave(l, mp,wave, t0, tn, tp)
  return Sum
###############################################################################
############## Optimization for parameters to hybridize waveform ##############
### Time Translation ######################
def get_t0guess(waveNR, waveModel,tNR ,tmodel,thybrid):
  """
  Only valid if two waveforms using same time units. 
  use wave1 and wave2 22 modes compute the frequencies and
  get the timeshift such that the analytic freq match closely to 
  the numerical frequency at thybrid.
  frequency provide us guess for time shift need for aligning two
  waveforms.
  Use time of two waveforms such that at those time analytical frequency 
  match numerical frequency. Use the time difference to get time shifts.
  NOTE:
  If two waveforms start at zero than use Ian Hinder trick. That is use 
  the time index for time shifts rather time itself.
  
  """
  wave1 = waveNR.copy()
  wave2 = waveModel.copy()
  

  t1 = tNR.copy()
  del_t1 = t1[1]-t1[0] #timestep for derivative
  
  t2 = tmodel.copy()
  del_t2 = t2[1]-t2[0]
  
  amp1 = np.absolute(wave1)
  phase1 = np.angle(wave1)
  smoothphase1 = np.unwrap(phase1) #positive phase and freq
  freq1 = np.abs(1e0/del_t1*np.gradient(smoothphase1))
  #freq1 = np.abs(1e0/del_t1*np.diff(smoothphase1))
  target_indx = np.searchsorted(t1, thybrid) #issue 
  #omega_target
  wtarget = freq1[target_indx] #frequency at ~thybrid 
  #t_vary = 2*np.pi/wtarget
  amp2 = np.absolute(wave2)
  phase2 = np.angle(wave2)
  smoothphase2 =np.unwrap(phase2)
  freq2 =np.abs(1e0/del_t2*np.gradient(smoothphase2))
  #freq2 =np.abs(1e0/del_t2*np.diff(smoothphase2))
  indx1 = np.argmin(np.absolute(freq1 - wtarget))
  indx2 = np.argmin(np.absolute(freq2 - wtarget))
  t0guess = t1[indx1] - t2[indx2]  #if only need shift in Model, +- issue 
  #t0guess =  np.abs(indx2 -indx1) #Ian hinder data case both hlm starts at same t_vals
  return  t0guess


#############   For Hybrid Data  ######################################

def TransitionFunction(t, tstr, tend):
    """
    Function for transition in hybrid interval:
    can use Vijay one as well
    """
    if t < tstr:
        return 1
    if t > tend:
        return 0
    return 0.5* (1 + np.cos(np.pi*(t-tstr)/(tend -tstr)))


def Analytic_NR_Hybrid_Data(tnr, tmodel, waveNR, waveModel, thyrid_str, thybrid_end):
    """
    Given rotated and shifted data for two waveforms 
    with their corresponding time
    and given the starting and ending hybrid time.
    This function will return time, re and im part of
    Hybrid  waveform
    """
    #Step 1: Make analytic data dt = dt of NR

    deltaTNR = tnr[1] - tnr[0]
    N =int((tnr[-1] - tmodel[0])/deltaTNR)

    # we may need even number data points  for FFT for mismatch using Lal
    #if (N%2 != 0):  
    #    N = N+1    
    Tarray = np.empty(N)
    for i in range(0,N):
        Tarray[i] = tmodel[0] + i*deltaTNR
    #Make a long NR grid zeros where data is nor present
    print("issue =", len(Tarray), len(tnr), len(waveNR))
    NRarray = np.interp(Tarray, tnr, waveNR)
    ReNR = NRarray.real
    ImNR = NRarray.imag
    # make model array with same dt as NR but end at tnr[-1] or close to it

    MDarray = np.interp(Tarray, tmodel,waveModel)
    ReMD = MDarray.real
    ImMD = MDarray.imag
    #Hybrid waveform
    Rehyb = np.zeros_like(Tarray)
    Imhyb = np.zeros_like(Tarray)
    for i in range(len(Tarray)):
        Rehyb[i] = TransitionFunction(Tarray[i] , thyrid_str, thybrid_end)*ReMD[i] + (1.0 - TransitionFunction(Tarray[i] , thyrid_str, thybrid_end))* ReNR[i] 
        Imhyb[i] = TransitionFunction(Tarray[i] , thyrid_str, thybrid_end)*ImMD[i] + (1.0 - TransitionFunction(Tarray[i] , thyrid_str, thybrid_end))* ImNR[i]

    return  Tarray, Rehyb, Imhyb



############################################################################
#########################From Surrogate Code using gwtools libraries########
############################################################################
def unwrap_phase(Z):
  """
  Given complex hlm a+ib
  Function will return 
  unwrapped phase
  """
  phase =  np.angle(Z)
  return np.unwrap(phase)

def Amplitude(Z):
  """
  Given complex waveform a+ib
  the function will return 
  amplitude = sqrt(a^2 + b^2) 
  """
  return np.absolute(Z)

def angular_frequency(Z, T):
  """
  Given a+ib and T for delta_t 
  the function will compute
  smooth phase, get the omega
  by central differencing 
  of frequency.
  NOTE: This is angular frequency
  freq = omega/2pi if one need
  """
  phase = np.angle(Z)
  smoothphase = np.unwrap(phase)#-1123.73 
  # angular frequency using centered differencing not a good method though
  delt = T[1] -T[0]
  omega = np.empty()
  omega = 1e0/delt*np.diff(smoothphaseSXS)
  return omega

#Using GWtools
def get_peak(x, y):
  """Get argument and values of x and y at maximum value of |y|"""
  arg = np.argmax(y)
  return [arg, x[arg], y[arg]]


def phase(h):
  """Get phase of waveform, h = A*exp(i*phi)"""
  if np.shape(h):
    # Compute the phase only for non-zero values of h, otherwise set phase to zero.
    nonzero_h = h[np.abs(h) > 1e-300]
    phase = np.zeros(len(h), dtype='double')
    phase[:len(nonzero_h)] = np.unwrap(np.real(-1j*np.log(nonzero_h/np.abs(nonzero_h))))
  else:
    nonzero_h = h
    phase = np.real(-1j*np.log(nonzero_h/np.abs(nonzero_h)))
  return phase


def cycles(h):
  """Count number of cycles (to merger, if present) in waveform"""
  phi = phase(h)
  ipk, phi_pk, A_pk = get_peak(phi, np.abs(h))
  return (phi_pk - phi[0])/(2.*np.pi)

#############Old Code for Strain Calculations################
# Strain   for fixed theta phi
def fac(n):
   result = 1
   for i in range(2, n+1):
      result *= i
   return result

# coefficient function
def Cslm(s, l, m):
    return sqrt( l*l * (4.0*l*l - 1.0) / ( (l*l - m*m) * (l*l - s*s) ) )# recursion function


def s_lambda_lm(s, l, m, x):
    Pm = pow(-0.5, m)
    if (m !=  s): Pm = Pm * pow(1.0+x, (m-s)*1.0/2)
    if (m != -s): Pm = Pm * pow(1.0-x, (m+s)*1.0/2)
    Pm = Pm * sqrt( fac(2*m + 1) * 1.0 / ( 4.0*pi * fac(m+s) * fac(m-s) ) )
    if (l == m):
        return Pm
    Pm1 = (x + s*1.0/(m+1) ) * Cslm(s, m+1, m) * Pm
    if (l == m+1):
        return Pm1
    else:
        for n in range (m+2, l+1):
            Pn = (x + s*m * 1.0 / ( n * (n-1.0) ) ) * Cslm(s, n, m) * Pm1 - Cslm(s, n, m) * 1.0 / Cslm(s, n-1, m) * Pm
            Pm = Pm1
            Pm1 = Pn
        return Pn


def sYlm(ss, ll, mm, theta, phi):
    Pm = 1.0
    l = ll
    m = mm
    s = ss

    if (l < 0):
        return 0
    if (abs(m) > l or l < abs(s)):
        return 0

    if (abs(mm) < abs(ss)):
        s=mm
        m=ss
        if ((m+s) % 2):
            Pm  = -Pm

    if (m < 0):
        s=-s
        m=-m
        if ((m+s) % 2):
            Pm  = -Pm
    result = Pm * s_lambda_lm(s, l, m, cos(theta))
    return complex(result * cos(mm*phi), result * sin(mm*phi))


#Strain
def strain(lmax, modes):
  Sum = 0.0
  for l in range(2, lmax+1):
    for m in range(-l, l+1):
      Sum += modes[(l,m)]*sYlm(-2, l,m, np.pi/2, -np.pi)
  return Sum


# Strain  as function of theta phi
def Angle_strain(lmax, modes, theta, phi):
  Sum = 0.0
  for l in range(2, lmax+1):
    for m in range(-l, l+1):
      Sum += modes[(l,m)]*sYlm(-2, l,m, theta, phi)
  return Sum




######################################################################
################## For Mismatch ######################################

def nextPow2(length):
    """
    Find next power of 2 <= length of data
    """
    return int(2**np.ceil(np.log2(length)))

def Mode_Mismatch_after_alignment(Time, WaveHybrid, WaveExact, psd, f_Low, f_Max):
    """
    Given two waveforms mode (any mode), and time
    time must be same for two waveforms with same dt
    This function will compute mismatch based one
    COMPLEXIP method see below or lalsimutils.
    The waveforms overlap is computed in frequency domain.
    psd  is used from  lalsimulation
    lalsimulation.SimNoisePSDaLIGOZeroDetHighPower
    fLow =20 LIGO sensitive freq
    fMax = 2000 
    COMPLEXIP is computed using
    Complex-valued inner product. self.ip(h1,h2) computes

          fNyq
    2 int       h1(f) h2*(f) / Sn(f) df
          -fNyq

    And similarly for self.norm(h1)

    N.B. DOES NOT assume h1, h2 are Hermitian - they should contain negative
         and positive freqs. packed as
    [ -N/2 * df, ..., -df, 0, df, ..., (N/2-1) * df ]
    DOES NOT maximize over time or phase. 
    """
    deltaT = Time[1] - Time[0]
    npts = len(Time)

    # Create h data structures
    hT1 = lal.CreateCOMPLEX16TimeSeries("Complex overlap", lal.LIGOTimeGPS(0.), 0., deltaT, lal.DimensionlessUnit, npts)
    hT1.data.data = WaveHybrid

    hT2 = lal.CreateCOMPLEX16TimeSeries("Complex overlap",lal.LIGOTimeGPS(0.), 0., deltaT, lal.DimensionlessUnit, npts)
    hT2.data.data =   WaveExact

    # ZeroPadding
    pow2npts = nextPow2(npts)
    hT1 = lal.ResizeCOMPLEX16TimeSeries(hT1, 0, pow2npts)
    hT2 = lal.ResizeCOMPLEX16TimeSeries(hT2, 0, pow2npts)

    # Create fourier transforms
    hF1 = lalsimutils.DataFourier(hT1)
    hF2 = lalsimutils.DataFourier(hT2)
    #lalfreq1 = lalsimutils.evaluate_fvals(hF1)
    #lalfreq2 = lalsimutils.evaluate_fvals(hF2)
    IP = lalsimutils.CreateCompatibleComplexIP(hF1,psd=psd, fLow=f_Low, fMax=f_Max,analyticPSD_Q=True,interpolate_max=False)

    print ("Inner product of two waveforms =",IP.ip(hF1,hF2))
    rho_1 = IP.norm(hF1)
    rho_2 = IP.norm(hF2)
    inner_12 = IP.ip(hF1,hF2)/rho_1/rho_2
    Overlap, Mismatch = inner_12, 1.0 - inner_12
    print ("ipwavehyb = ",   rho_1 , "ipwaveexc = " , rho_2 ," overlap = ", inner_12, "Mismatch = ", 1.0 - inner_12)

    return  Mismatch


def Mode_Mismatch_Maximize_over_T_Phi(Time, WaveHybrid, WaveExact, psd, f_Low, f_Max):
    """
    Given two waveforms mode (any mode), and time
    time must be same for two waveforms with same dt
    This function will compute mismatch based one
    COMPLEXOverlap method see below or lalsimutils 
    `psd`  use lalsimulation  to get it
    we are choosing  lalsim.SimNoisePSDaLIGOZeroDetHighPower
    fLow =20
    fMax = 1800
    These are LIGO freq domain
    Inner product maximized over time and polarization angle.
    This inner product does not assume Hermitianity and is therefore
    valid for waveforms that are complex in the TD, e.g. h+(t) + 1j hx(t).
    self.IP(h1,h2) computes:

                  fNyq
    max 2 Abs int      h1*(f,tc) h2(f) / Sn(f) df
     tc          -fNyq

    h1, h2 must be COMPLEX16FrequencySeries defined in [-fNyq, fNyq-deltaF]
    At least one of which should be non-Hermitian for the maximization
    over phase to work properly.    

    """
    deltaT = Time[1] - Time[0]
    npts = len(Time)

    # Create h data structures
    hT1 = lal.CreateCOMPLEX16TimeSeries("Complex overlap", lal.LIGOTimeGPS(0.), 0., deltaT, lal.DimensionlessUnit, npts)
    hT1.data.data = WaveHybrid

    hT2 = lal.CreateCOMPLEX16TimeSeries("Complex overlap",lal.LIGOTimeGPS(0.), 0., deltaT, lal.DimensionlessUnit, npts)
    hT2.data.data =   WaveExact

    # ZeroPadding
    pow2npts = nextPow2(npts)
    hT1 = lal.ResizeCOMPLEX16TimeSeries(hT1, 0, pow2npts)
    hT2 = lal.ResizeCOMPLEX16TimeSeries(hT2, 0, pow2npts)
       ##Try Further zero padding

    # Create fourier transforms
    hF1 = lalsimutils.DataFourier(hT1)
    hF2 = lalsimutils.DataFourier(hT2)
    #lalfreq1 = lalsimutils.evaluate_fvals(hF1)
    #lalfreq2 = lalsimutils.evaluate_fvals(hF2)
    IP = lalsimutils.CreateCompatibleComplexOverlap(hF1,psd=psd, fLow=f_Low, fMax=f_Max,analyticPSD_Q=True,interpolate_max=False)

    print ("Inner product of two waveforms =",IP.ip(hF1,hF2))
    rho_1 = IP.norm(hF1)
    rho_2 = IP.norm(hF2)
    inner_12 = IP.ip(hF1,hF2)/rho_1/rho_2
    Overlap, Mismatch = inner_12, 1.0 - inner_12
    print ("ipwavehyb = ",   rho_1 , "ipwaveexc = " , rho_2 ," overlap = ", inner_12, "Mismatch = ", 1.0 - inner_12)

    return  Mismatch


#Mismatch based on 22 mode phase maximization  # Incomplete
def Mismatch_Maximize_over_T_Phi_of22Mode(Time, WaveHybrid, WaveExact, psd, f_Low, f_Max):
    """
    Given two waveforms mode (any mode), and time
    time must be same for two waveforms with same dt
    This function will compute mismatch based one
    COMPLEXOverlap method see below or lalsimutils 
    `psd`  use lalsimulation  to get it
    we are choosing  lalsim.SimNoisePSDaLIGOZeroDetHighPower
    fLow =20
    fMax = 1800
    These are LIGO freq domain

    """
    deltaT = Time[1] - Time[0]
    npts = len(Time)

    # Create h data structures
    hT1 = lal.CreateCOMPLEX16TimeSeries("Complex overlap", lal.LIGOTimeGPS(0.), 0., deltaT, lal.DimensionlessUnit, npts)
    hT1.data.data = WaveHybrid

    hT2 = lal.CreateCOMPLEX16TimeSeries("Complex overlap",lal.LIGOTimeGPS(0.), 0., deltaT, lal.DimensionlessUnit, npts)
    hT2.data.data =   WaveExact

    # ZeroPadding
    pow2npts = nextPow2(npts)
    hT1 = lal.ResizeCOMPLEX16TimeSeries(hT1, 0, pow2npts)
    hT2 = lal.ResizeCOMPLEX16TimeSeries(hT2, 0, pow2npts)

    # Create fourier transforms
    hF1 = lalsimutils.DataFourier(hT1)
    hF2 = lalsimutils.DataFourier(hT2)
    #lalfreq1 = lalsimutils.evaluate_fvals(hF1)
    #lalfreq2 = lalsimutils.evaluate_fvals(hF2)
    IP = lalsimutils.CreateCompatibleComplexOverlap(hF1,psd,psd=psd, fLow=f_Low, fMax=f_Max,analyticPSD_Q=True,interpolate_max=False)

    print ("Inner product of two waveforms =",IP.ip(hF1,hF2))
    rho_1 = IP.norm(hF1)
    rho_2 = IP.norm(hF2)
    inner_12 = IP.ip(hF1,hF2)/rho_1/rho_2
    Overlap, Mismatch = inner_12, 1.0 - inner_12
    print ("ipwavehyb = ",   rho_1 , "ipwaveexc = " , rho_2 ," overlap = ", inner_12, "Mismatch = ", 1.0 - inner_12)

    return  Mismatch

#############   Tapering the Data and Zero Padding ###########################
def Planktaper(t, t1, t2, t3, t4):
    """
    Given time series of waveform compute the function that can deal with gibbs 
    phenomenon for ringdown and initial junk
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
        if z > 0:
            return np.exp(-z)/(1+ np.exp(-z))
        else:
            return 1.0/(1+ np.exp(z))
    if (t >= t4):
        return 0
    return t



def get_Windowed_Zeropadded_Data(Time, WaveHybrid, Tstr,Tend):
    """
    Return T and Z= zeropadded windowed data
    Give the clean data after applying window
    and appropriately zeropadded 
    The time array  with same dt but length equal
    to hlm
    """
    newHlm = np.zeros_like(WaveHybrid)
    for i in range(len(Time)-1):
        newHlm[i] = Planktaper(Time[i], Time[0], Tstr, Tend, Time[-1])*WaveHybrid[i]
    deltaT = Time[1] - Time[0]
    npts = len(Time)

    # Create h data structures
    hT1 = lal.CreateCOMPLEX16TimeSeries("Complex overlap", lal.LIGOTimeGPS(0.), 0., deltaT, lal.DimensionlessUnit, npts)
    hT1.data.data = newHlm
    # ZeroPadding
    pow2npts = nextPow2(npts)
    NewT = np.zeros(pow2npts)
    hT1 = lal.ResizeCOMPLEX16TimeSeries(hT1, 0, pow2npts)
    for i in range(pow2npts):
        NewT[i] = Time[0] + i * deltaT 

    return NewT, hT1.data.data
    

def Waveform_afterWindowing(Time, Hlm, Tstr,Tend):
    """
    Given hlm mode and time, Tstart where we start the actual wave
    form Tend where we want to remove the later portion of waveform
    This fucntion will return tapered waveform.
  
    The choice of Tstr can be few cycles after the fmin used in Model
    The choice of Tend can be cycle after the merger 
    We in our case take t== 0 as merger 
    """
    newHlm = np.zeros_like(Hlm)
    for i in range(len(Time)-1):
        newHlm[i] = Planktaper(Time[i], Time[0], Tstr, Tend, Time[-1])*Hlm[i]
    return newHlm

#############  rescale of total mass ####################
def Rescale_data_in_Total_Mass(Mold, Mnew, Time, Hlm):
  """
  Take data with given total mass, change Hlm 
  by rescaling total mass as well as time.
  New data is of same sample rate as original
  that is dt is same  
  """
  deltaT = Time[1] - Time[0]
  Tnew = Time *Mnew/Mold
  Hlmnew = Hlm *Mnew/Mold
  N = int((Tnew[-3] - Tnew[2])/deltaT)
  if (N %2 != 0):
    N = N+1
  T_interp = np.zeros(N)
  Hlminterp = np.zeros(N)
  for i in range(N):
    T_interp[i] = Tnew[2] + i*deltaT
  Hlm_interp = np.interp(T_interp, Tnew, Hlmnew)
  return  T_interp, Hlm_interp

# If one have already windowed and zero paded the waveforms then use this function
def get_mismatch_over_TotMass(Time, h1, h2, MtotUsed,l,m, psd, f_Low, f_Max):
  """
  Given h1, h2 with same dt, and time
  get fft compute mismtach
  Rescales the to mass and get new missmatch
  Plot the final mismatch for each mass 
  Right now I am not using any windowing on data
  Just rescaling by mass
  """
  Masschange = np.linspace(20,140,20)
  Mtot = []
  Mismatch =[]
  for k in Masschange:
    Rescsalet1, Rescaleh1 = Rescale_data_in_Total_Mass(MtotUsed,k , Time, h1)
    Rescsalet2, Rescaleh2 = Rescale_data_in_Total_Mass(MtotUsed,k , Time, h2)
    mismatch_i =  Mode_Mismatch_Maximize_over_T_Phi(Rescsalet1, Rescaleh1 ,Rescaleh2, psd, f_Low, f_Max)
    print (k, mismatch_i)
    Mtot.append(k)
    Mismatch.append(mismatch_i)
  #plt.figure(figsize=(10, 8))
  #plt.plot(Mtot, Mismatch, linestyle='--', marker='o', color='b')
  #plt.title("$\ell = {0}, m ={1}$".format(l,m))
  #plt.grid()
  #plt.xlabel('$\mathrm{M}_{tot}$')
  #plt.ylabel("$\mathcal{M}$")

  #plt.show()
  return Mtot, Mismatch

##########################################################################################
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

  # Create Fourier transforms
  hF1 = lalsimutils.DataFourier(hT1)
  hF2 = lalsimutils.DataFourier(hT2)

  # Create inner product
  psd=lalsim.SimNoisePSDaLIGOZeroDetHighPower
  #IP = lalsimutils.CreateCompatibleComplexIP(hF1,psd=psd,fLow=20,fMax=1700,analyticPSD_Q=True)
  IP = lalsimutils.CreateCompatibleComplexOverlap(hF1,psd=psd,fLow=20,fMax=1700,analyticPSD_Q=True,interpolate_max=True)
  rho_1 = IP.norm(hF1)
  rho_2 = IP.norm(hF2)
  inner_12 = IP.ip(hF1,hF2)/rho_1/rho_2
  return 1.0 - inner_12

##########################################################################################
def get_plot_mismatch_over_TotMass(Time, h1, h2, MtotUsed,l,m):
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
    mismatch_i =  Mismatch_from_hlm(Rescsalet1, Rescaleh1 ,Rescaleh2 )
    print (k, mismatch_i)
    Mtot.append(k)
    Mismatch.append(mismatch_i)
  #Rescale

  print (Mismatch)
  plt.figure(figsize=(10, 8))
  plt.plot(Mtot, Mismatch, linestyle='--', marker='o', color='b')
  plt.title("$\ell = {0}, m ={1}$".format(l,m))
  plt.grid()
  plt.xlabel('$\mathrm{M}_{tot}$')
  plt.ylabel("$\mathcal{M}$")

  #plt.show()
  return Mtot, Mismatch

########## Weighted Mismatch #######################
def EMRichard(lmax):
    Sum = 0
    for l in range(2, lmax+1):
        for m in range(-l, l+1):
            Sum += norm(l,m)*Mismatch(l,m)/norm(l,m)
    return Sum




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
    return np.abs(omega)/(2*np.pi)

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
    plt.axvline(x= newt1, color='r')
    plt.axvline(x= newt2, color='r')
    plt.xlabel("T[sec]")
    plt.ylabel("Freq[Hz]")
    plt.legend(loc= 2, fontsize=20)
    plt.show()


def get_plots_mismatch__freq_over_TotMass(Time, h1, h2, MtotUsed,l,m,t1,t2):
  """
 Given h1, h2 with same dt, and time
 get fft compute mismtach
 Rescales the to mass and get new missmatch
 Plot the final mismatch for each mass 
 Right now I am not using any windowing on data
Just rescaling by mass
  """
  Masschange = np.linspace(10,140,14)
  #Masschange = np.array([70., 40.]) #np.linspace(10,150,15)
  Mtot = []#np.zeros_like(Masschange)
  Mismatch =[]# np.zeros_like(Masschange)
  for k in Masschange:
    Rescsalet1, Rescaleh1 = Rescale_data_in_Total_Mass(MtotUsed,k , Time, h1)
    Rescsalet2, Rescaleh2 = Rescale_data_in_Total_Mass(MtotUsed,k , Time, h2)
    mismatch_i =  Mismatch_from_hlm(Rescsalet1, Rescaleh1 ,Rescaleh2 )
    #plot_freq(Time, h1 , MtotUsed,k, t1,t2)
    print (k, mismatch_i)
    Mtot.append(k)
    Mismatch.append(mismatch_i)
  #Rescale

  print (Mismatch)
  #plt.figure(figsize=(10, 8))
  plt.plot(Mtot, Mismatch, linestyle='--', marker='o', label= "$\ell = {0}, m ={1}$".format(l,m))
  #plt.title("$\ell = {0}, m ={1}$".format(l,m))
  #plt.grid()
  plt.xlabel('$\mathrm{M}_{tot}$')
  plt.ylabel("$\mathcal{M}$")
  #plt.semilogy()
  #plt.savefig("/home/jxs1805/GIT/JamSadiq/HybridWaveforms/HybridWaveforms/Paper/figs/MismatchLogl{0}m{1}SXS1412.png".format(l,m))
  #plt.show()
  return l, m, Mtot, Mismatch






########################mismatch from Strain for different angles

def hoft(waveform, time, theta, phi):
    """
    given waveform modes all available
    the result will be h(t) the strain
    h = sum(l,m)  hlm Y^-2_lm(thta, phi)
    thta = 0.785 ~np.pi/4
    phi = 0.0
    lal.SpinWeightedSphericalHarmonic(REAL8 theta, REAL8 phi, int s, int l, int m)
    """
    deltaT  = time[1] - time[0]
    npts = len(time)
    wfmTS = lal.CreateCOMPLEX16TimeSeries("Psi4", lal.LIGOTimeGPS(0.), 0., deltaT, lal.DimensionlessUnit, npts)
    wfmTS.data.data[:] = 0
    for mode in waveform.keys():
        wfmTS.data.data +=  waveform[mode]*lal.SpinWeightedSphericalHarmonic(theta,phi,-2, int(mode[0]),int(mode[1])) #*np.exp(2*sgn*1j*self.P.psi)
    return wfmTS

def Rescalehoft(waveform, time, theta, phi, Mused, Mneed):
    """
    hoft will be in lalformat so data.data 
    will need for rescaling and then we need to save in
    same format too
    """
    deltaT = time[1] - time[0]
    Tnew = time * float(Mneed)/Mused
    ht = hoft(waveform, time, theta, phi)
    ht.data.data *= float(Mneed)/Mused
    N = int((Tnew[-1] - Tnew[0])/deltaT)
    if(N%2 != 0 ):
        N = N+1
    Tinterp = np.zeros(N)
    Hlminterp = np.zeros(N)
    for i in range(N):
        Tinterp[i] = Tnew[0] + i*deltaT
    Hlminterp = np.interp(Tinterp, Tnew,ht.data.data)
    return  Tinterp, Hlminterp

def Mismatch_hoft_Optimize_OverTphi(h1,h2, time, psd, f_Low, f_Max):
    """ 
     Mismatch optimized over t and phi.
     
    """
    deltaT = time[1] - time[0]
    npts = len(time)
    #get hoft and rescale it as needed we can also window the data as well
    # Create h data structures
    hT1 = lal.CreateCOMPLEX16TimeSeries("Complex overlap", lal.LIGOTimeGPS(0.), 0., deltaT, lal.DimensionlessUnit, npts)
    hT1.data.data[:] = np.array(h1)

    hT2 = lal.CreateCOMPLEX16TimeSeries("Complex overlap",lal.LIGOTimeGPS(0.), 0., deltaT, lal.DimensionlessUnit, npts)
    hT2.data.data[:] =  np.array(h2)

    # ZeroPadding
    pow2npts = nextPow2(npts)
    hT1 = lal.ResizeCOMPLEX16TimeSeries(hT1, 0, pow2npts)
    hT2 = lal.ResizeCOMPLEX16TimeSeries(hT2, 0, pow2npts)

   # Create fourier transforms
    hF1 = lalsimutils.DataFourier(hT1)
    hF2 = lalsimutils.DataFourier(hT2)
    psd=lalsim.SimNoisePSDaLIGOZeroDetHighPower
    #lalfreq1 = lalsimutils.evaluate_fvals(hF1)
    #lalfreq2 = lalsimutils.evaluate_fvals(hF2)
    IP = lalsimutils.CreateCompatibleComplexOverlap(hF1,psd=psd, fLow=f_Low, fMax=f_Max,analyticPSD_Q=True,interpolate_max=False)

    print ("Inner product of two waveforms =",IP.ip(hF1,hF2))
    rho_1 = IP.norm(hF1)
    rho_2 = IP.norm(hF2)
    inner_12 = IP.ip(hF1,hF2)/rho_1/rho_2
    Overlap, Mismatch = inner_12, 1.0 - inner_12
    print ("ipwavehyb = ",   rho_1 , "ipwaveexc = " , rho_2 ," overlap = ", inner_12, "Mismatch = ", 1.0 - inner_12)

    return  Mismatch


def Mismatchplot(hHyb,hExc,Time, theta, phi, M, Mstandard, psd, f_Low, f_Max):
  """
  Return the min and max of mismatch for each Mtot
  will also plot the mismatch as function
  of theta phi for each Mtot
  """
  xx = []
  yy = []
  zz = []
  cc = []
  for i in range(len(theta)):
    for j in range(len(phi)):
      h1 = hoft(hHyb, Time, theta[i], phi[j]).data.data
      h2 = hoft(hExc, Time, theta[i], phi[j]).data.data
      Rescaledh1 = h1 *float(M)/Mstandard
      Rescaledh2 = h2 *float(M)/Mstandard
      deltaT = Time[1] - Time[0]
      Tnew = Time * float(M)/Mstandard
      N = int((Tnew[-1] - Tnew[0])/deltaT)
      if(N%2 != 0 ):
        N = N+1
      Tinterp = np.zeros(N)
      Hlminterp = np.zeros(N)
      for k in range(N):
        Tinterp[k] = Tnew[0] + k*deltaT
      Hlminterp1 = np.interp(Tinterp, Tnew,Rescaledh1)
      Hlminterp2 = np.interp(Tinterp, Tnew,Rescaledh2)

      xx.append(np.sin(theta[i])*np.cos(phi[j]))
      yy.append(np.sin(theta[i])*np.sin(phi[j]))
      zz.append(np.cos(theta[i]))
      mismatch_i =  Mismatch_hoft_Optimize_OverTphi(Hlminterp1,Hlminterp2, Tinterp,psd, f_Low, f_Max)
      cc.append(mismatch_i)

  xx = np.array(xx)
  yy = np.array(yy)
  zz = np.array(zz)
  cc = np.array(cc)
  #print("min and max mismatches ")
  #print (cc.min(), cc.max())
#np.savetxt("mismatch",np.stack((xx, yy, zz, cc), axis=-1))

  r = 1
  pi = np.pi
  cos = np.cos
  sin = np.sin
  phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0*pi:100j]
  x = r*sin(phi)*cos(theta)
  y = r*sin(phi)*sin(theta)
  z = r*cos(phi)
  fig = plt.figure(figsize=(10,8))
  ax = fig.add_subplot(111, projection='3d')
  ax.plot_wireframe(x, y, z,  rstride=1, cstride=1, color='c', alpha=0.01, linewidth=0.1)
  
  p=ax.scatter(xx,yy,zz, c=cc,cmap="coolwarm" ,s=20)
  fig.colorbar(p, ax=ax)
  ax.set_xlim([-1,1])
  ax.set_ylim([-1,1])
  ax.set_zlim([-1,1])
  #ax.set_aspect("equal")
  plt.title("M ={0}".format(M))
  plt.tight_layout()
  return cc.min(), cc.max()

