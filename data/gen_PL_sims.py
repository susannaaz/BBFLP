import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import os

from optparse import OptionParser
parser = OptionParser()
parser.add_option('--predir', dest='predir', 
                  default='/pscratch/sd/s/susannaz/BBMASTER/data/',
                  type=str, help='Output directory')
(o, args) = parser.parse_args()
predir = o.predir

os.system(f'mkdir -p {predir}PL')
os.system(f'mkdir -p {predir}val')
os.system(f'mkdir -p {predir}CMBl')
os.system(f'mkdir -p {predir}CMBr')

nside = 64
ls = np.arange(3*nside)
cl = 1/(ls+10)**2
np.savez(f'{predir}PL/cl_PL.npz', ls=ls,
         clTT=cl, clEE=cl, clEB=cl, clBE=cl, clBB=cl)

clTT = 1000/(ls+10)**0.5
clEE = 1/(ls+10)**0.5
clBB = 0.05/(ls+10)**0.5
cl0 = np.zeros(3*nside)
np.savez(f'{predir}val/cl_val.npz', ls=ls,
         clTT=clTT, clEE=clEE, clEB=cl0, clBE=cl0, clBB=clBB)

d = np.loadtxt(f'{predir}camb_lens_nobb.dat', unpack=True) #l,dtt,dee,dbb,dte
clTT_CMBl = np.zeros(3*nside)
clTT_CMBl[1:] = (2*np.pi*d[1]/(d[0]*(d[0]+1)))[:3*nside-1]
clEE_CMBl = np.zeros(3*nside)
clEE_CMBl[1:] = (2*np.pi*d[2]/(d[0]*(d[0]+1)))[:3*nside-1]
clBB_CMBl = np.zeros(3*nside)
clBB_CMBl[1:] = (2*np.pi*d[3]/(d[0]*(d[0]+1)))[:3*nside-1]
np.savez(f'{predir}CMBl/cl_CMBl.npz', ls=ls,
         clTT=clTT_CMBl, clEE=clEE_CMBl, clEB=cl0, clBE=cl0, clBB=clBB_CMBl)

dlr = np.loadtxt(f'{predir}camb_lens_r1.dat', unpack=True)
rfid = 0.01
clTT_CMBr = clTT_CMBl #?
clEE_CMBr = clEE_CMBl #? so you cancel it out in full cl?
clBB_CMBr = np.zeros(3*nside)
clBB_CMBr[1:] = (2*np.pi*rfid*(dlr[3]-d[3])/(d[0]*(d[0]+1)))[:3*nside-1]
np.savez(f'{predir}CMBr/cl_CMBr.npz', ls=ls,
         clTT=clTT_CMBr, clEE=clEE_CMBr, clEB=cl0, clBE=cl0, clBB=clBB_CMBr)

npol = 3
npix = hp.nside2npix(nside)
nsims = 200
for i in range(nsims):
    print(i)
    seed = 1000+i
    np.random.seed(seed)
    
    # Fiducial PL sims
    alm = hp.synalm(cl)
    alm0 = alm*0
    mp_T = hp.synfast(cl,nside,lmax=3*nside-1)
    mpE_Q, mpE_U = hp.alm2map_spin([alm, alm0], nside, spin=2, lmax=3*nside-1)
    mpB_Q, mpB_U = hp.alm2map_spin([alm0, alm], nside, spin=2, lmax=3*nside-1)
    mpsE_pl = np.zeros([npol, npix]); mpsE_pl[0] = mp_T; mpsE_pl[1] = mpE_Q; mpsE_pl[2] = mpE_U
    mpsB_pl = np.zeros([npol, npix]); mpsB_pl[0] = mp_T; mpsB_pl[1] = mpB_Q; mpsB_pl[2] = mpB_U
    hp.write_map(f'{predir}PL/plsim_{seed}_E.fits', mpsE_pl, overwrite=True)
    hp.write_map(f'{predir}PL/plsim_{seed}_B.fits', mpsB_pl, overwrite=True)

    # Alternative PL sims
    mp_T = hp.synfast(clTT,nside,lmax=3*nside-1)
    almE = hp.synalm(clEE)
    almB = hp.synalm(clBB)
    mp_Q, mp_U = hp.alm2map_spin([almE, almB], nside, spin=2, lmax=3*nside-1)
    mps_val = np.zeros([npol, npix]); mps_val[0] = mp_T; mps_val[1] = mp_Q; mps_val[2] = mp_U
    hp.write_map(f'{predir}val/valsim_{seed}.fits', mps_val, overwrite=True)
    
    # CMB sims (lensing only with Al=1)
    mpT = hp.synfast(clTT_CMBl,nside,lmax=3*nside-1)
    almE = hp.synalm(clEE_CMBl)
    almB = hp.synalm(clBB_CMBl)
    mp_Q, mp_U = hp.alm2map_spin([almE, almB], nside, spin=2, lmax=3*nside-1)
    mps_CMBl = np.zeros([npol, npix]); mps_CMBl[0] = mp_T; mps_CMBl[1] = mp_Q; mps_CMBl[2] = mp_U
    hp.write_map(f'{predir}CMBl/CMBl_{seed}.fits', mps_CMBl, overwrite=True)
    
    # CMB sims (no lensing, r=0.01)
    mpT = hp.synfast(clTT_CMBr,nside,lmax=3*nside-1)
    almE = hp.synalm(clEE_CMBr)
    almB = hp.synalm(clBB_CMBr)
    mp_Q, mp_U = hp.alm2map_spin([almE, almB], nside, spin=2, lmax=3*nside-1)
    mps_CMBr = np.zeros([npol, npix]); mps_CMBr[0] = mp_T; mps_CMBr[1] = mp_Q; mps_CMBr[2] = mp_U
    hp.write_map(f'{predir}CMBr/CMBr_{seed}.fits', mps_CMBr, overwrite=True)


