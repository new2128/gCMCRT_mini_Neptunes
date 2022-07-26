import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pylab as plt
import scipy.io as sio
from netCDF4 import Dataset
from astropy.constants import sigma_sb,L_sun,au
from sklearn import linear_model

# Read in iprf
iprf = 'MSteinrueck.iprf'
data = np.loadtxt(iprf)
idx = data[:,0]
P = data[:,1]
T = data[:,2]
ni = len(idx)

# ------------------------ read in abundance file and PT grid ----------------------------------------------------------

# ....... Define necessary arrays ..........
nspecies=37  #number of species in the table
np_abunds=18 # # of pressure points
nt_abunds=60 # # of temperature points
ncp=[15,16,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,
     18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18]
abunds=np.zeros((nt_abunds, np_abunds, nspecies)) #temperature,pressure,species
P_abunds=np.zeros((nt_abunds, np_abunds)) #in mbar
T_abunds=np.zeros((nt_abunds, np_abunds))

# ........ Read in list of species from abundances file .........
abundsfile=open('m+0.0_co1.0.data.11/full_abunds.txt','r')
species=abundsfile.readline()
species = species.split(' ')
for i in range(nt_abunds):
    for j in range(ncp[i]):
        line = abundsfile.readline()
        newline = line.strip()
        abunds[i,j,:] = newline.split('  ')
abundsfile.close()

#%% ............ Read in pressure temperature grid .............
ptfile=open('PT_list_all.txt','r')
data = np.loadtxt(ptfile,skiprows=1)
layers  = (data[:,0])
nL = len(layers)
print("number of layers: ", nL)
ptfile.close()

ptfile=open('PT_list_all.txt','r')
header=ptfile.readline()

for i in range(nt_abunds):
    for j in range(ncp[i]):
        line=ptfile.readline()
        newline = line.strip()
        layer,P_abunds[i,j],T_abunds[i,j]=newline.split()
ptfile.close()

# ......... Read abundances file again as a 1D array ..........
abundsfile=open('m+0.0_co1.0.data.11/full_abunds.txt','r')
data = np.loadtxt(abundsfile,skiprows=1)
VMR_2D = data[:,:]

# ..........Extract mu at each P and T .........................
ptfile = 'm+0.0_co1.0.data.11/cp_all'
data = np.loadtxt(ptfile,skiprows=1)
T_1D = data[:,0]
P_1D = data[:,1]
mu_1D = data[:,2]

# Remove duplicates from T and P
T_filtered = [] 
P_filtered = []
[T_filtered.append(x) for x in T_1D if x not in T_filtered] 
[P_filtered.append(x) for x in P_1D if x not in P_filtered] 

 # Create a dataframe from the temp, pressure, mu table
df = pd.DataFrame({'pressure':P_1D, 'temperature':T_1D, 'mu':mu_1D})
        
# Create a linear regression function for mu
X = df[['temperature','pressure']]
Y = df['mu']
regr = linear_model.LinearRegression()
regr.fit(X, Y)

# Now loop across the whole 1D profile and interpolate to each P-T point in the GCM
mu_1D = np.zeros(ni)
VMR_1D = np.zeros((ni,nspecies))
for n in range(ni):
    #regression for mu
    mu_1D[n] = regr.predict([[T[n],P[n]]])

def interpolate_abunds(Pp, Tp, ispecies, logabunds, P_abunds, T_abunds, Tinv, logP_abunds, ncp):
    # Interpolates abundances similar to the way abundances are interpolated in the GCM
    # Inputs:
    # Float Pp:     Pressure, for which interpolation is desired, in mbar
    # Float Tp:     Temperature, for which interpolation is desired, in mbar
    # ispecies:     species index for which interpolation is desired
    #               some popular species: 1=H2, 8=He, 9=H2O, 10=CH4, 11=CO, 12=NH3, 13=N2, 25=CO2, 26=HCN
    # ndarray logabunds:
    #               2D array containing the natural logarithm of the abundances.
    # ndarray P_abunds:
    #               2D array containing the pressures from the pressure-temperature grid in mbar
    # ndarray T_abunds:
    #               2D array containing the temperatures from the pressure-temperature grid in K
    # ndarray Tinv:
    #               2D array containing 1/T_abunds (for computational efficiency if looping over huge number of points)
    # ndarray P_abunds:
    #               2D array containing np.log(P_abunds) (for computational efficiency if looping over huge number of
    #               points)
    # list, tuple or ndarray ncp:
    #               array containing number of pressure points for each temperature point. Has to be specified when
    #               reading in the file. Value should be
    #               ncp=[15,16,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,18,18,18,18,18,18,18,18,18,18,18,18,18,18,
    #               18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18]

    nt_abunds=logabunds.shape[0]
    np_abunds=logabunds.shape[1]

    # find P index
    jlowP = np.searchsorted(P_abunds[-1, :], Pp) - 1

    # print(P_abunds[-1,jlowP],sim.pc[k]*1e-2,P_abunds[-1,jlowP+1])

    # find T index
    jlowT = np.searchsorted(T_abunds[:, 0], Tp) - 1

    # print(T_abunds[jlowT,-1],Tp,T_abunds[jlowT+1,-1])

    # deal with P or T being out of range
    if (Tp < T_abunds[0, 0]):
        jlowT = 0
        print('Warning: Temperature out of range... extrapolating.')
    elif (Tp > T_abunds[-1, 0]):
        jlowT = nt_abunds - 2
        print('Warning: Temperature out of range... extrapolating.')
    if Pp < P_abunds[-1, 0]:
        jlowP = 0
        print('Warning: pressure too low... extrapolating')
    elif Pp > P_abunds[jlowT, ncp[jlowT] - 1]:
        jlowP = np_abunds - 2
        print('Warning: Layer %d pressure too high... extrapolating')

    tt = (1 / Tp - Tinv[jlowT, jlowP]) / (Tinv[jlowT + 1, jlowP] - Tinv[jlowT, jlowP])
    u = (np.log(Pp) - logP_abunds[jlowT, jlowP]) / (logP_abunds[jlowT, jlowP + 1] - logP_abunds[jlowT, jlowP])

    logX = (1 - tt) * (1 - u) * logabunds[jlowT, jlowP, ispecies] \
           + tt * (1 - u) * logabunds[jlowT + 1, jlowP, ispecies] \
           + tt * u * logabunds[jlowT + 1, jlowP + 1, ispecies] \
           + (1 - tt) * u * logabunds[jlowT, jlowP + 1, ispecies]

    return np.exp(logX)

logP_abunds=np.log(P_abunds)
Tinv=1/T_abunds
logabunds=np.log(abunds)

# Select the species used in this profile - the order should match optools.nml  
ispecies=[1,8,9,10,11,13,25] #1=H2, 8=He, 9=H2O, 10=CH4, 11=CO, 12=NH3, 13=N2, 25=CO2, 26=HCN
# The species do not need to completely match the species/order in optools. H2, He, N2 are not in optools as they are not strong opacity sources
speciesnames=['H2','He','H2O','CH4','CO','N2','CO2']
nsp = len(ispecies) 

X = np.zeros((ni, nsp))
logP_abunds=np.log(P_abunds)
Tinv=1/T_abunds
logabunds=np.log(abunds)

for j in range(nsp):
    for k in range(ni):
        X[k,j] = interpolate_abunds(P[k],T[k],ispecies[j],logabunds,P_abunds,T_abunds,Tinv,logP_abunds,ncp)

# Create a .prf file
head = open('../data/header.txt','r')
lines = head.readlines()

fname = 'MSteinrueck.prf'
prf = open(fname,'w')
prf.write(lines[0])  # height, P, T, mu, VMR
prf.write(lines[1])  # number of layers 
prf.write(str(ni) + '\n')
prf.write(lines[2])  # number and names of gases in the table (same order as VMR)
prf.write((str(nsp) + '\n'))
for n in range(nsp):
    prf.write(speciesnames[n] + '\n')
prf.write(lines[3])  # begiin profile 
prf.write(lines[4])  # n_lay, PG, TG, mug, VMR(:)
print(lines[4])
for n in range(ni):
        prf.write(str(n+1) + ' ' + str(P[n]) + ' ' + str(T[n]) + ' ' + str(mu_1D[n]) + ' ' + " ".join(str(l) for l in X[n,:]) + '\n')
