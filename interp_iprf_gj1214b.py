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

# ........ Read in abundances file .........
abundsfile=open('m+0.0_co1.0.data.11/full_abunds.txt','r')
header=abundsfile.readline()

for i in range(nt_abunds):
    for j in range(ncp[i]):
        line=abundsfile.readline()
        newline = line.strip()
        abunds[i,j,:]= newline.split('  ')
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
VMR_1D = np.zeros((nL,nspecies))
data = np.loadtxt(abundsfile,skiprows=1)
VMR_1D = data[:,:]

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
print(df['pressure'])

# convert 1D to 3D arrays
VMR = np.zeros((nL, nL, nspecies))
mu_15 = np.zeros((1,15))
mu_16 = np.zeros((1,16))
mu_17 = np.zeros((15,17))
mu_18 = np.zeros((43,18))

n = 0
for i in range(nt_abunds):
    for j in range(ncp[i]):
        VMR[i,j,:] = VMR_1D[n,:]
        n = n + 1

n=0
for i in range(1):
    for j in range(15):
        mu_15[i,j] = mu_1D[n]
        n = n + 1 

n=15
for i in range(1):
    for j in range(16):
        mu_16[i,j] = mu_1D[n]
        n = n + 1 

n=31
for i in range(15):
    for j in range(17):
        mu_17[i,j] = mu_1D[n]
        n = n + 1 

n=286
for i in range(143):
    for j in range(18):
        try:
            mu_18[i,j] = mu_1D[n]
            n = n + 1
        except IndexError:
            break
        
# Create a linear regression function for mu
X = df[['temperature','pressure']]
Y = df['mu(amu)'].values
regr = linear_model.LinearRegression()
regr.fit(X, Y)

'''
# Create a scipy interpolation function for mu
f_mu_15 = interpolate.interp2d(T_filtered[0], P_filtered[0:15], mu_15, kind='linear')
f_mu_16 = interpolate.interp2d(T_filtered[1], P_filtered[0:16], mu_16, kind='linear')
f_mu_17 = interpolate.interp2d(T_filtered[2:17], P_filtered[0:17], mu_17, kind='linear')
f_mu_18 = interpolate.interp2d(T_filtered[17:], P_filtered[0:18], mu_18, kind='linear')
'''

# Create a scipy interpolation function for VMR of each species
f_VMR = []
for i in range(nspecies):
    f_VMR.append(interpolate.interp2d(T_1D[::-1],P_1D[::-1],VMR[:,:,i], kind='linear'))

# Now loop across the whole 1D profile and interpolate to each P-T point in the GCM
mu_1D = np.zeros(ni)
VMR_1D = np.zeros((ni,nspecies))
for n in range(ni):
    #regression for mu
    mu_1D[n] = regr.predict([[T[n],P[n]]])
    '''
    # interpolate mu with switches 
    if T[n]<87.5:
        mu_1D[n] = f_mu_15(T[n],P[n], mu_15, kind = 'linear')
    elif 87.5 <= T[n] < 105:
        mu_1D[n] = f_mu_16(T[n],P[n], mu_16, kind = 'linear')
    elif 105 <= T[n] < 255:
        mu_1D[n] = f_mu_17(T[n],P[n], mu_17, kind = 'linear')
    else:
        mu_1D[n] = f_mu_18(T[n],P[n], mu_18, kind = 'linear')
    '''
    # interpolate VMRs
    for s in range(nspecies):
        VMR_1D[n,s] = 10.0**f_VMR[s](T[n],P[n])

# Select the species used in this profile - the order should match optools.nml  
ispecies=[1,8,9,10,11,13,25] #1=H2, 8=He, 9=H2O, 10=CH4, 11=CO, 12=NH3, 13=N2, 25=CO2, 26=HCN
speciesnames=['H$_2$','He','H$_2$O','CH$_4$','CO','N$_2$','CO$_2$']
nsp = len(ispecies)

# Create a .prf file
head = open('../data/header.txt','r')
lines = head.readlines()

fname = 'MSteinrueck.prf'
prf = open(fname,'w')
prf.write(lines[0])  # height, P, T, mu, VMR
prf.write(lines[1])  # number of layers 
prf.write(str(ni) + '\n')
prf.write(lines[2])  # number and names of gases in the table (same order as VMR)
for n in range(nsp):
    prf.write(str(ispecies) + '\n')
for n in range(nsp):
    prf.write(speciesnames[n] + '\n')
prf.write(lines[3])  # begiin profile 
prf.write(lines[4])  # n_lay, PG, TG, mug, VMR(:)
for n in range(ni):
        prf.write(str(n+1) + ' ' + str(P[n]) + ' ' + str(T[n]) + ' ' + str(mu_1D[n]) + ' ' + " ".join(str(l) for l in VMR_1D[n,:]) + '\n')
