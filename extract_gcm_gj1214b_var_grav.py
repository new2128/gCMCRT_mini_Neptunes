import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import scipy.io as sio
from scipy import interpolate
from netCDF4 import Dataset
from astropy.constants import sigma_sb,L_sun,au
from sklearn import linear_model

R = 8.31446261815324
kb = 1.380649e-23
amu = 1.66053906660e-27

# Altitude function with variable gravity 
def calc_alt_col_var_g(nlev,Plev,T,gs,Rd,Rp):
    alt = np.zeros(nlev)
    alt_new = np.zeros(nlev)
    grav = np.zeros(nlev)

    # Initial guess with constant gravity 
    alt[0] = 0.0
    for k in range(nlev-1):
        alt[k+1] = alt[k] + (Rd[k]*T[k])/gs * np.log(Plev[k]/Plev[k+1])

    alt_new[:] = alt[:]
    # Converge using alt[k] for gravity
    itera = 0
    converge = False
    while (converge == False) :

      # Current delta alt
      atdepth = alt[-1] - alt[0]

      # Perform hydrostatic calcuation
      alt_new[0] = 0.0
      for k in range(nlev-1):
        grav[k] = gs * (Rp / (Rp + alt[k]))**2
        alt_new[k+1] = alt_new[k] + (Rd[k]*T[k])/grav[k] * np.log(Plev[k]/Plev[k+1])
      grav[-1] = gs * (Rp / (Rp + alt[-1]))**2

      # New delta alt
      atdepth1 = alt_new[-1] - alt_new[0]

      # Convergence test
      itera = itera + 1
      xdepth  = 100.0 * abs((atdepth1-atdepth)/atdepth)
      #print(itera, atdepth, atdepth1, xdepth)
      if (xdepth < 1e-8):
        converge = True

      alt[:] = alt_new[:]

    return alt, grav

# Matlab file 
mat_fname = 'GJ1214b_solar_15552000.mat'
mat_contents = sio.loadmat(mat_fname,squeeze_me=True)
sim = mat_contents['sim']

# Hard code the haze particle radius and log-normal st. dev.)
r0 = 3.0 * 0.001  # 3 nm
sig = np.log(1.05)
rho_c = 1000.0

Rp = sim['rSphere']
g = sim['gravity']
Rd = sim['atm_kappa']*sim['atm_Cp']
plev = sim['pf'].item()
play = sim['pc'].item()
lons = sim['xi'].item()
lats = sim['yi'].item()

nlon = len(lons)
nlat = len(lats)
nlay = len(play)
nlev = nlay + 1
print(nlat, nlon, nlay) 

# Roll the longitudes to start at 0-360 rather than -180-180
rl = int(len(lons)/2)
print(rl)
lons = np.roll(lons,rl)
lons = np.where(lons > 0.0, lons, 360 + lons)
nlines = nlon * nlat * nlay

dat = mat_contents['dat']
T = dat['temp'].item()
u = dat['uci'].item()
v = dat['vci'].item()
w = dat['vert'].item()
q = dat['tracer'].item()[1]  # haze production rate for 3nm particles 

# Sort out NaNs at min and max latitude for all variables
T[:,0,:] = T[:,1,:]
T[:,-1,:] = T[:,-2,:]
u[:,0,:] = u[:,1,:]
u[:,-1,:] = u[:,-2,:]
v[:,0,:] = v[:,1,:]
v[:,-1,:] = v[:,-2,:]
w[:,0,:] = w[:,1,:]
w[:,-1,:] = w[:,-2,:]
q[:,0,:] = q[:,1,:]
q[:,-1,:] = q[:,-2,:]

# Array dimensions go as nlon, nlat, nlay

# Roll all the data arrays
for j in range(nlat):
  for k in range(nlay):
      T[:,j,k] = np.roll(T[:,j,k],rl)
      u[:,j,k] = np.roll(u[:,j,k],rl)
      v[:,j,k] = np.roll(v[:,j,k],rl)
      w[:,j,k] = np.roll(w[:,j,k],rl)
      q[:,j,k] = np.roll(q[:,j,k],rl)

# Interp to find mu at each P and T
ptfile = 'm+0.0_co1.0.data.11/cp_all'
data = np.loadtxt(ptfile,skiprows=1)
T_1D = data[:,0]
P_1D = data[:,1]
P_1D = P_1D * .001 # mbar to bar conversion
mu_1D = data[:,2]

# Create a dataframe from the temp, pressure, mu table
df = pd.DataFrame({'pressure':np.log10(P_1D), 'temperature':np.log10(T_1D), 'mu':mu_1D})
        
# Create a linear regression function for mu
X = df[['temperature','pressure']]
Y = df['mu']
regr = linear_model.LinearRegression()
regr.fit(X, Y)

# Now we can loop across the whole 3D profile
Rd = np.zeros((nlon,nlat,nlay))
for j in range(nlat):
    for i in range(nlon):
        for k in range(nlay):
          # interpolate mu
          Rd[i,j,k] = R/(10.0**(regr.predict([[np.log10(T[i,j,k]),np.log10(play[k]/1e5)]])) / 1000.0)
print('Rd example', Rd[2][2][2])
print('Rd example, changing height', Rd[2][2][5])
print('Rd example, changing lat/lon', Rd[5][5][5])

# Calculate the altitude of each column
alt = np.zeros((nlon,nlat,nlev))
glev = np.zeros((nlon,nlat,nlev))
alt_mid = np.zeros((nlon,nlat,nlay))
for i in range(nlon):
  for j in range(nlat):
      alt[i,j,:], glev[i,j,:] = calc_alt_col_var_g(nlev,plev[:],T[i,j,:],g,Rd[i,j,:],Rp)
      alt_mid[i,j,:] = (alt[i,j,0:nlay] + alt[i,j,1:nlev])/2.0
print('glev example', glev[2][2][2])
print('glev example, changing height', glev[2][2][5])
print('glev example, changing lat/lon', glev[5][5][5])

# Find the column index of maximum altitude to interpolate to
imax = np.unravel_index(np.argmax(alt, axis=None), alt.shape)
alt_grid = alt[imax[0],imax[1],:]
alt_grid_mid = (alt_grid[0:nlay] + alt_grid[1:nlev])/2.0

# Output the veritcal height grid
fname = 'MSteinrueck.hprf'
f = open(fname,'w')
for k in range(nlev):
    f.write(str(k+1) + ' ' + str((alt_grid[k] + Rp)*100.0) + '\n')

f.close()

# Interpolate the data variables to the new height grid
iT = np.zeros((nlon,nlat,nlay))
iP = np.zeros((nlon,nlat,nlay))
iu = np.zeros((nlon,nlat,nlay))
iv = np.zeros((nlon,nlat,nlay))
iw = np.zeros((nlon,nlat,nlay))
iq = np.zeros((nlon,nlat,nlay))
ig = np.zeros((nlon,nlat,nlay))
iRd = np.zeros((nlon,nlat,nlay))
for i in range(nlon):
  for j in range(nlat):
      iT[i,j,:] = np.interp(alt_grid_mid[:],alt_mid[i,j,:],T[i,j,:])
      iP[i,j,:] = 10.0**np.interp(alt_grid_mid[:],alt_mid[i,j,:],np.log10(play[:]),right=-12)
      iRd[i,j,:] = np.interp(alt_grid_mid[:],alt_mid[i,j,:],Rd[i,j,:])
      iu[i,j,:] = np.interp(alt_grid_mid[:],alt_mid[i,j,:],u[i,j,:])
      iv[i,j,:] = np.interp(alt_grid_mid[:],alt_mid[i,j,:],v[i,j,:])
      iw[i,j,:] = np.interp(alt_grid_mid[:],alt_mid[i,j,:],w[i,j,:])
      iq[i,j,:] = np.interp(alt_grid_mid[:],alt_mid[i,j,:],q[i,j,:],right=0,left=0)
      ig[i,j,:] = np.interp(alt_grid_mid[:],alt[i,j,:],glev[i,j,:])
      # Now perform extrapolation in height for max alt < alt_grid assuming hydrostatic Eq.
      for k in range(nlay):
          if (iP[i,j,k] == 1e-12):
            p0 = iP[i,j,k-1]
            z0 = alt_grid_mid[k-1]
            H0 = (iRd[i,j,k] * iT[i,j,k]) / ig[i,j,k]
            iP[i,j,k] = p0 * np.exp(-(alt_grid_mid[k] - z0)/H0)
            if (iP[i,j,k] < 1e-12):
              iP[i,j,k] = 1e-12

# Find cloud particle number density
rho_atm = np.zeros((nlon,nlat,nlay))
rho_atm[:,:,:] = iP[:,:,:]/(Rd * iT[:,:,:])
nd = np.zeros((nlon,nlat,nlay))
nd[:,:,:] = (3.0*iq[:,:,:]*rho_atm[:,:,:])/(4.0*np.pi*rho_c*(r0*1e-6)**3*np.exp(4.5*sig**2))

# Now output T-p profile in bar and K to an interpolation file (iprf) - after which we can interpolate to GGChem values to get VMRs
fname = 'MSteinrueck.iprf'
print('Outputting interpolatable T-p profile: ', fname)
f = open(fname,'w')
n = 0
for j in range(nlat):
    for i in range(nlon):
        for k in range(nlay):
            f.write(str(n+1) + ' ' +  str(iP[i,j,k]/1e5) + ' ' + str(iT[i,j,k]) + '\n')
            n = n + 1
f.close()

# Output the 3D cloud profiles
fname = 'MSteinrueck.clprf'
print('Outputting cloud profile: ', fname)
f = open(fname,'w')
f.write('\n')
f.write('\n')
f.write(str(nlines) + ' ' + '1' + '\n')
f.write('\n')
f.write('1' + '\n')
f.write('Haze' + '\n')
f.write('\n')
f.write('\n')
n = 0
for j in range(nlat):
    for i in range(nlon):
        for k in range(nlay):
            f.write(str(n+1) + ' ' + str(r0) + ' ' + str(nd[i,j,k]/1e6) + ' ' + '1.0' + '\n')
            n = n + 1
f.close()

