import numpy as np
import matplotlib.pyplot as plt
import sys 
def plot_hlm(l,m):
    d1 = np.loadtxt(sys.argv[1]+"{0}{1}".format(l,m)).T
    d2 = np.loadtxt(sys.argv[2]+"{0}{1}".format(l,m)).T
    t1,Re1, im1 = d1[0],d1[1], d1[2]
    t2,Re2 , im2 = d2[0],d2[1], d2[2]
    plt.figure(figsize=(11,4))
    plt.plot(t1,Re1,'k',label ='{0}{1}'.format(l,m))
    plt.plot(t2,Re2, 'r--')
    plt.legend()
    plt.axvline(x= -2.2,color='orange')
    plt.axvline(x= -1.7,color='orange')
    plt.figure(figsize=(11,4))
    plt.plot(t1,im1,'k',label ='{0}{1}'.format(l,m))
    plt.plot(t2,im2, 'r--')
    plt.legend()
    plt.axvline(x= -2.2,color='orange')
    plt.axvline(x= -1.7,color='orange')
    
    

    plt.show()

plot_hlm(2,2)

plot_hlm(4,4)


plot_hlm(3,3)
plot_hlm(2,1)
    

