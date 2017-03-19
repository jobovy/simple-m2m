
#
# read output from densv2m1d_fstar_sample.py and run hom2m
#

import pyfits
import math
import numpy
import copy
import sys
# sys.path.append('../py')
from hom2m_sc1d import *
# import matplotlib.pyplot as plt
from scipy import stats
from galpy.util import bovy_coords
from galpy.util import bovy_plot
import matplotlib.cm as cm
from StringIO import StringIO
import seaborn as sns
from matplotlib import pylab
import matplotlib.pyplot as plt

numpy.random.seed(4)

# for observed density
# input file data
infile='cylinder_dens.fits'
rstar_hdulist=pyfits.open(infile)
rstar=rstar_hdulist[1].data
h_obs=rstar_hdulist[0].header['h_obs']
print 'h_obs=',h_obs
rstar_hdulist.close()

# set z_obs and density
z_obs=rstar['z_obs']
dens_obs=rstar['dens_obs']
dens_obs_noise=rstar['dens_obs_noise']

h_m2m= h_obs

# the input data has "unknown" zoffset so no need for z offset
zoff_obs=0.0
print ' dens =',dens_obs
print ' dens uncertainty =',dens_obs_noise

# bovy_plot.bovy_print(axes_labelsize=17.,text_fontsize=12.,xtick_labelsize=15.,ytick_labelsize=15.)
# plt.figure(figsize(6,4))
# bovy_plot.bovy_plot(z_obs,dens_obs,'ko',semilogy=True,
#  xlabel=r'$\tilde{z}$',ylabel=r'$\nu_{\mathrm{obs}}(\tilde{z})$',
#  xrange=[-.25,0.25],yrange=[0.1,30.])
# plt.errorbar(z_obs,dens_obs,yerr=dens_obs_noise,marker='None',ls='none',color='k')
# plt.show()

### read the velocity data
# input file data
infile='cylinder_v2m.fits'
rstar_hdulist=pyfits.open(infile)
rstar=rstar_hdulist[1].data
h_obs_v2m=rstar_hdulist[0].header['h_obs']
if h_obs_v2m!=h_obs:
  print 'h_obs_v2m is not consistent with h_obs! h_obs_v2m=',h_obs_v2m
  sys.exit()
rstar_hdulist.close()

# set z_obs and density
z_obsv2m=rstar['z_obs']
if numpy.all(numpy.not_equal(z_obsv2m,z_obs)):
  print 'z_obsv2m is not consistent with z_obs!'
  print ' z_obs, z_obsv2m=',z_obs,z_obsv2m
  sys.exit()
v2m_obs=rstar['v2m_obs']
v2m_obs_noise=rstar['v2m_obs_noise']
print '<v^2>^1/2=',numpy.sqrt(v2m_obs)
print '<v^2>_unc^1/2=',numpy.sqrt(v2m_obs_noise)

# vz for Sun from Schoenrich et al. 2010
vzsun=7.25

### initial model

n_m2m= 4000
# assume velocity km/s, distance kpc unit 
sigma_init= 18.0
E_m2m= numpy.random.exponential(scale=sigma_init**2.,size=n_m2m)
phi_m2m= numpy.random.uniform(size=n_m2m)*2.*numpy.pi
# although it is guess
# scale with 220 km/s and 8 kpc
sigma_true=sigma_init
# omega = sqrt(2)x(v (km/s))/z (kpc)
omega_true= 122.0
print 'omega_true=',omega_true
A_m2m= numpy.sqrt(2.*E_m2m)/omega_true
w_init= numpy.ones(n_m2m)
z_m2m= A_m2m*numpy.cos(phi_m2m)
z_out= numpy.linspace(-0.3,0.3,101)
vz_m2m= -omega_true*A_m2m*numpy.sin(phi_m2m)
# use zsun guess
zsun_guess=0.025
dens_init= compute_dens(z_m2m,zsun_guess,z_out,h_m2m,w_init)
v2m_init= compute_v2m(z_m2m,vz_m2m,zsun_guess,z_out,h_m2m,w_init)

# plot the observational data and initial model
bovy_plot.bovy_print(axes_labelsize=17.,text_fontsize=12.,xtick_labelsize=15.,ytick_labelsize=15.)
v2m_init= compute_v2m(z_m2m,vz_m2m,zsun_guess,z_out,h_m2m,w_init)

# plt.figsize(6,4)
plt.subplot(1,2,1)
bovy_plot.bovy_plot(z_out,dens_init,'-',semilogy=True,
                   xlabel=r'$\tilde{z}$',ylabel=r'$\nu_{\mathrm{obs}}(\tilde{z})$',
                   xrange=[-.25,0.25],yrange=[0.1,10.],gcf=True)
bovy_plot.bovy_plot(z_obs,dens_obs,'o',semilogy=True,overplot=True)
plt.errorbar(z_obs,dens_obs,yerr=dens_obs_noise,marker='None',ls='none',color=sns.color_palette()[1])
plt.yscale('log',nonposy='clip')
#
plt.subplot(1,2,2)
bovy_plot.bovy_plot(z_out,numpy.sqrt(v2m_init),'-',semilogy=True,
                   xlabel=r'$\tilde{z}$',ylabel=r'$\langle v^2\rangle^{1/2}$',
                    xrange=[-.25,0.25],yrange=[10.0,100.0],gcf=True)
bovy_plot.bovy_plot(z_obs,numpy.sqrt(v2m_obs),'o',semilogy=True,overplot=True)
bovy_plot.bovy_plot(z_out,numpy.sqrt(v2m_init),'-',semilogy=True,overplot=True,zorder=0)
plt.errorbar(z_obs,numpy.sqrt(v2m_obs),yerr=numpy.sqrt(v2m_obs_noise),marker='None',ls='none',color=sns.color_palette()[1])
# plt.yscale('log',nonposy='clip')

plt.show()

### run M2M with density and v^2 constraints
step= numpy.pi/3.*10.**-2.
step=0.2
nstep= 10000
eps= 10.**-4.
# x n_m2m, because density is normalised with n, somehow this is better
eps_vel= eps*n_m2m
# eps_vel= eps
mu= 1.
h_m2m= h_obs
nodens= False
omega_m2m= omega_true
zsun_m2m= zsun_guess
z_out= numpy.linspace(-0.3,0.3,101)
v2m_init= compute_v2m(z_m2m,vz_m2m,zsun_guess,z_out,h_m2m,w_init)
w_out,Q,wevol,windx= run_m2m_weights_wv2m(w_init,A_m2m,phi_m2m,omega_m2m,zsun_m2m,
                                         z_obs,dens_obs,dens_obs_noise,v2m_obs,v2m_obs_noise,
                                         nstep=nstep,step=step,mu=mu,eps=eps,h_m2m=h_m2m,nodens=nodens,
                                         output_wevolution=10,eps_vel=eps_vel)

### plot the result

z_m2m= A_m2m*numpy.cos(phi_m2m+nstep*step*omega_m2m)
vz_m2m= -A_m2m*omega_m2m*numpy.sin(phi_m2m+nstep*step*omega_m2m)
z_out= numpy.linspace(-0.3,0.3,101)
# use zsun_guess
dens_final= compute_dens(z_m2m,zsun_guess,z_out,h_m2m,w_out)
bovy_plot.bovy_print(axes_labelsize=17.,text_fontsize=12.,xtick_labelsize=15.,ytick_labelsize=15.)
# figsize(15,6)
plt.subplot(2,3,1)
bovy_plot.bovy_plot(z_out,dens_init,'-',semilogy=True,
  xlabel=r'$\tilde{z}$',ylabel=r'$\nu_{\mathrm{obs}}(\tilde{z})$',
  xrange=[-.25,0.25],yrange=[0.0,10.],gcf=True)
bovy_plot.bovy_plot(z_obs,dens_obs,'o',semilogy=True,overplot=True)
bovy_plot.bovy_plot(z_out,dens_final,'-',semilogy=True,overplot=True,zorder=0)
plt.errorbar(z_obs,dens_obs,yerr=dens_obs_noise,marker='None',ls='none',color=sns.color_palette()[1])
plt.yscale('log',nonposy='clip')
#
plt.subplot(2,3,2)
v2m_final= compute_v2m(z_m2m,vz_m2m,zsun_guess,z_out,h_obs,w_out)
bovy_plot.bovy_plot(z_out,numpy.sqrt(v2m_init),'-',semilogy=False,
                   xlabel=r'$\tilde{z}$',ylabel=r'$\langle v^2\rangle^{1/2}$',
                    xrange=[-.25,0.25],yrange=[0.0,50.0],gcf=True)
bovy_plot.bovy_plot(z_obs,numpy.sqrt(v2m_obs),'o',semilogy=False,overplot=True)
bovy_plot.bovy_plot(z_out,numpy.sqrt(v2m_final),'-',semilogy=False,overplot=True,zorder=0)
plt.errorbar(z_obs,numpy.sqrt(v2m_obs),yerr=numpy.sqrt(v2m_obs_noise),marker='None',ls='none',color=sns.color_palette()[1])
# plt.yscale('log',nonposy='clip')
#
plt.subplot(2,3,3)
for ii in range(len(wevol)):
    bovy_plot.bovy_plot(numpy.linspace(0.,1.,nstep)*nstep*step*omega_true/2./numpy.pi,wevol[ii],'-',
                        color=cm.viridis(A_m2m[windx][ii]/0.3),
                        yrange=[-0.2,numpy.amax(wevol)*1.1],
                        semilogx=True,xlabel=r'$\mathrm{orbits}$',ylabel=r'$w(t)$',gcf=True,overplot=ii>0)
#
plt.subplot(2,3,4)
bovy_plot.bovy_plot(A_m2m,w_out,'k.',xlabel=r'$z_{\mathrm{max}}$',ylabel=r'$w(z_{\mathrm{max}})$',
                   yrange=[0.,5.],gcf=True)
sindx= numpy.argsort(A_m2m)
w_expect= numpy.exp((A_m2m[sindx]*omega_m2m)**2./2.*(1./sigma_init**2.-1./sigma_true**2.))
w_expect/= numpy.sum(w_expect)/len(w_expect)
plt.plot(A_m2m[sindx],w_expect,lw=2.)
#
plt.subplot(2,3,5)
bovy_plot.bovy_plot(numpy.linspace(0.,1.,nstep)*nstep*step*omega_true/2./numpy.pi,numpy.sum(Q,axis=1),lw=3.,
                   loglog=True,xlabel=r'$\mathrm{orbits}$',ylabel=r'$F$',gcf=True)
plt.subplot(2,3,6)
_= plt.hist(vz_m2m,weights=w_out,histtype='step',lw=2.,normed=True,bins=31,zorder=1)
#_= plt.hist(vz_vmock,histtype='step',lw=2.,normed=True,bins=31,zorder=2)
_= plt.hist(vz_m2m,histtype='step',lw=2.,normed=True,bins=31,ls='--',zorder=0)
plt.xlabel(r'$v_z$')
plt.ylabel(r'$p(v_z)$')
print("Velocity dispersions: fit", \
      numpy.sqrt(numpy.sum(w_out*(vz_m2m-numpy.sum(w_out*vz_m2m)/numpy.sum(w_out))**2.)/numpy.sum(w_out)))

# plt.tight_layout()
plt.show()

### run M2M with variable zsun

step= numpy.pi/20.0
nstep= 40000
eps= 10.**-4.
eps_zo= eps/100.
mu= 1.
omega_m2m= omega_true
zsun_m2m= zsun_guess
print 'h_m2m (variable zsun)=',h_m2m
(w_out,zsun_out),Q,wevol,windx= run_m2m_weights_zsun_densv2m(
  w_init,A_m2m,phi_m2m,omega_m2m,zsun_m2m,
  z_obs,dens_obs,dens_obs_noise,v2m_obs,v2m_obs_noise,
  nstep=nstep,step=step,mu=mu,eps=eps,eps_vel=eps_vel,h_m2m=h_m2m,
  eps_zo=eps_zo,output_wevolution=10)

print("Zsun: fit, starting point",zsun_out[-1],zsun_m2m)
# plot
z_m2m= A_m2m*numpy.cos(phi_m2m+nstep*step*omega_m2m)
vz_m2m= -A_m2m*omega_m2m*numpy.sin(phi_m2m+nstep*step*omega_m2m)
z_out= numpy.linspace(-0.3,0.3,101)
dens_final= compute_dens(z_m2m,zsun_out[-1],z_out,h_m2m,w_out)
bovy_plot.bovy_print(axes_labelsize=17.,text_fontsize=12.,xtick_labelsize=15.,ytick_labelsize=15.)
# figsize(15,6)
plt.subplot(2,3,1)
bovy_plot.bovy_plot(z_out,dens_init,'-',semilogy=True,
                   xlabel=r'$\tilde{z}$',ylabel=r'$\nu_{\mathrm{obs}}(\tilde{z})$',
                   xrange=[-.25,0.25],yrange=[0.1,30.],gcf=True)
bovy_plot.bovy_plot(z_obs,dens_obs,'o',semilogy=True,overplot=True)
bovy_plot.bovy_plot(z_out,dens_final,'-',semilogy=True,overplot=True,zorder=0)
plt.errorbar(z_obs,dens_obs,yerr=dens_obs_noise,marker='None',ls='none',color=sns.color_palette()[1])
plt.yscale('log',nonposy='clip')
#
plt.subplot(2,3,2)
v2m_final= compute_v2m(z_m2m,vz_m2m,zsun_guess,z_out,h_obs,w_out)
bovy_plot.bovy_plot(z_out,numpy.sqrt(v2m_init),'-',semilogy=True,
                   xlabel=r'$\tilde{z}$',ylabel=r'$\langle v^2\rangle^{1/2}$',
                    xrange=[-.25,0.25],yrange=[10.0,100.0],gcf=True)
bovy_plot.bovy_plot(z_obs,numpy.sqrt(v2m_obs),'o',semilogy=True,overplot=True)
bovy_plot.bovy_plot(z_out,numpy.sqrt(v2m_final),'-',semilogy=True,overplot=True,zorder=0)
plt.errorbar(z_obs,numpy.sqrt(v2m_obs),yerr=numpy.sqrt(v2m_obs_noise),marker='None',ls='none',color=sns.color_palette()[1])
plt.yscale('log',nonposy='clip')
#
plt.subplot(2,3,3)
for ii in range(len(wevol)):
    bovy_plot.bovy_plot(numpy.linspace(0.,1.,nstep)*nstep*step*omega_true/2./numpy.pi,wevol[ii],'-',
                        color=cm.viridis(A_m2m[windx][ii]/0.3),
                        yrange=[-0.2,numpy.amax(wevol)*1.1],
                        semilogx=True,xlabel=r'$\mathrm{orbits}$',ylabel=r'$w(t)$',gcf=True,overplot=ii>0)
#
plt.subplot(2,3,4)
bovy_plot.bovy_plot(numpy.linspace(0.,1.,nstep)*nstep*step*omega_m2m/2./numpy.pi,numpy.sum(Q,axis=1),lw=3.,
                   loglog=True,xlabel=r'$\mathrm{orbits}$',ylabel=r'$F$',gcf=True)
# no plot for vz
plt.subplot(2,3,5)
_= plt.hist(vz_m2m,weights=w_out,histtype='step',lw=2.,normed=True,bins=31,zorder=1)
# _= plt.hist(vz_vmock,histtype='step',lw=2.,normed=True,bins=31,zorder=2)
_= plt.hist(vz_m2m,histtype='step',lw=2.,normed=True,bins=31,ls='--',zorder=0)
plt.xlabel(r'$v_z$')
plt.ylabel(r'$p(v_z)$')
print("Velocity dispersions: fit", \
      numpy.sqrt(numpy.sum(w_out*(vz_m2m-numpy.sum(w_out*vz_m2m)/numpy.sum(w_out))**2.)/numpy.sum(w_out)))
#
plt.subplot(2,3,6)
bovy_plot.bovy_plot(numpy.linspace(0.,1.,nstep)*nstep*step*omega_m2m/2./numpy.pi,zsun_out,'-',
                   xlabel=r'$\mathrm{orbits}$',ylabel=r'$z_\odot(t)$',gcf=True,semilogx=True,zorder=1)
plt.axhline(zsun_guess,color=sns.color_palette()[1],lw=2.,zorder=0)
#print("Velocity dispersions: mock, fit",numpy.std(vz_vmock),\
#      numpy.sqrt(numpy.sum(w_out*(vz_m2m-numpy.sum(w_out*vz_m2m)/numpy.sum(w_out))**2.)/numpy.sum(w_out)))
# plt.tight_layout()
print("Zsun: fit, starting point",zsun_out[-1],zsun_m2m)
plt.show()

