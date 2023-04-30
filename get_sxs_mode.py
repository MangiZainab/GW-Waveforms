from __future__ import print_function
import h5py
import numpy as np


from sympy.physics.units import G, kg, s, m, au, year
import scipy.constants 
from numpy import pi
import numpy as np

parsec = 648000/pi * au
megaparsec = 10**6 * parsec
msun = 4*pi**2*au**3/G/year**2

c=scipy.constants.speed_of_light*m/s

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



def get_22modes_from_file(filename):
    DISTANCE=300*megaparsec
    MASS = 70 * msun
    file = h5py.File(filename,"r")
    print(list(file.keys()))
    #quit()
    hlm=file[u'Extrapolated_N4.dir']
    #Tshift using 22
    time22, real22, imag22 =hlm[u'Y_l{0}_m{1}.dat'.format(2,2)][:].T
    time_sec = GeometricTime_To_MKS_Time(time22, MASS)
    newreal22 = GeometricStrain_TO_Observer_Strain(real22, MASS, DISTANCE)
    newimag22 = GeometricStrain_TO_Observer_Strain(imag22, MASS, DISTANCE)
    max_amp_indx = np.argmax(np.absolute( real22 +  imag22 *1j ))
    #for i in range(len(time22)):
    #  print(time_sec[i]) 
    tshift = time_sec[max_amp_indx]
    dt = 1e0/2**14#time_sec[-1] - time_sec[-2]
    N = int((time_sec[-1] - time_sec[0])/dt)
    Tarray = np.empty(N)
    for i in range(N):
        Tarray[i] = time_sec[0]+ i*dt
    time, real, imag =hlm[u'Y_l{0}_m{1}.dat'.format(2,2)][:].T
    phys_time = GeometricTime_To_MKS_Time(time, MASS) #- tshift
    NewT = np.interp(Tarray ,phys_time,phys_time) - tshift
    phys_real = GeometricStrain_TO_Observer_Strain(real, MASS,DISTANCE)
    NewRe = np.interp(Tarray ,phys_time,phys_real)
    phys_imag = GeometricStrain_TO_Observer_Strain(imag, MASS,DISTANCE)
    NewIm = np.interp(Tarray ,phys_time,phys_imag)
    return NewT , NewRe+ NewIm*1j


def get_modes_from_file(filename):
    DISTANCE=300*megaparsec
    MASS = 70 * msun
    file = h5py.File(filename,"r")
    print(list(file.keys()))
    #quit()
    hlm=file[u'Extrapolated_N4.dir']
    #Tshift using 22
    time22, real22, imag22 =hlm[u'Y_l{0}_m{1}.dat'.format(2,2)][:].T
    time_sec = GeometricTime_To_MKS_Time(time22, MASS)
    newreal22 = GeometricStrain_TO_Observer_Strain(real22, MASS, DISTANCE)
    newimag22 = GeometricStrain_TO_Observer_Strain(imag22, MASS, DISTANCE)
    max_amp_indx = np.argmax(np.absolute( real22 +  imag22 *1j ))
    #for i in range(len(time22)):
    #  print(time_sec[i]) 
    tshift = time_sec[max_amp_indx]
    dt = 1e0/2**14#time_sec[-1] - time_sec[-2]
    N = int((time_sec[-1] - time_sec[0])/dt)
    Tarray = np.empty(N)
    for l in range(2,5):
        for m in range(-l,l+1):
            time, real, imag =hlm[u'Y_l{0}_m{1}.dat'.format(l,m)][:].T
            phys_time = GeometricTime_To_MKS_Time(time, MASS) #- tshift
            NewT = np.interp(Tarray ,phys_time,phys_time) - tshift
            phys_real = GeometricStrain_TO_Observer_Strain(real, MASS,DISTANCE)
            NewRe = np.interp(Tarray ,phys_time,phys_real)
            phys_imag = GeometricStrain_TO_Observer_Strain(imag, MASS,DISTANCE)
            NewIm = np.interp(Tarray ,phys_time,phys_imag)
            np.savetxt('SXS{0}{1}'.format(l,m), np.stack((NewT,NewRe,NewIm), axis=-1))
    file.close()

