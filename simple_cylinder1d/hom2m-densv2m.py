
#
# read output from dens1d_fstar.py and run hom2m
#

import pyfits
import math
import numpy
import copy
import sys
sys.path.append('../py')
from hom2m import *
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

# input parameters
# We only observe the density at a few z
# z_obs= numpy.array([0.075,0.1,0.125,0.15,0.175,0.2,-0.075,-0.1,-0.125,-0.15,-0.175,-0.2])
z_obs= numpy.array([0.075,0.1,0.125,0.15,0.175,-0.075,-0.1,-0.125,-0.15,-0.175])
h_obs= 0.1
h_m2m= h_obs

# print input parameters

# input selected stellar data
infile='selstars_xyz.fits'
rstar_hdus=pyfits.open(infile)
rstar=rstar_hdus[1].data
rstar_hdus.close()

# read the data
# number of data points
z_mock=rstar['zpos']
n_mock=len(z_mock)

print 'number of selected stars for density =',n_mock

# bovy_plot.bovy_hist(numpy.fabs(z_mock),bins=11,normed=True,
#                    xlabel=r'$z$',ylabel=r'$\nu(z)$',lw=2.,histtype='step')
# pylab.gca().set_yscale('log')
# plt.show()
#plt.savefig('dens_hist.jpg')

# for observed density
# the input data has "unknown" zoffset so no need for z offset
zoff_obs=0.0
dens_obs= compute_dens(z_mock,zoff_obs,z_obs,h_obs)
def compute_nsbin(z,zsun,z_obs,h_obs,w=None):
    if w is None: w= numpy.ones_like(z)
    nsbin= numpy.zeros_like(z_obs)
    for jj,zo in enumerate(z_obs):
       nsbin[jj]+= numpy.sum(w[numpy.fabs(zo-z+zsun)<1.0*h_obs])
    return nsbin
# poisson error
nsbin_obs=compute_nsbin(z_mock,zoff_obs,z_obs,h_obs)
print ' Ns bin for density=',nsbin_obs
dens_obs_noise= dens_obs/numpy.sqrt(nsbin_obs)
print ' dens =',dens_obs
print ' dens uncertainty =',dens_obs_noise

# dens_obs_noise= numpy.sqrt(dens_obs)*0.2*numpy.sqrt(numpy.amax(dens_obs))\
#    /(numpy.fabs(z_obs**2)/numpy.amin(numpy.fabs(z_obs**2)))
# observation already has a noise
# dens_obs+= numpy.random.normal(size=dens_obs.shape)*dens_obs_noise

# bovy_plot.bovy_print(axes_labelsize=17.,text_fontsize=12.,xtick_labelsize=15.,ytick_labelsize=15.)
# plt.figure(figsize(6,4))
# bovy_plot.bovy_plot(z_obs,dens_obs,'ko',semilogy=True,
#  xlabel=r'$\tilde{z}$',ylabel=r'$\nu_{\mathrm{obs}}(\tilde{z})$',
#  xrange=[-.25,0.25],yrange=[0.1,30.])
# plt.errorbar(z_obs,dens_obs,yerr=dens_obs_noise,marker='None',ls='none',color='k')
# plt.show()

### read the velocity data
infile='selstars_xyzvz.fits'
rstar_hdus=pyfits.open(infile)
rstar=rstar_hdus[1].data
rstar_hdus.close()

# read the data
# number of data points
z_vmock=rstar['zpos']
vz_vmock=rstar['velz']
n_vmock=len(z_vmock)

print 'number of selected stars for <v^2> =',n_vmock
# vz for Sun from Schoenrich et al. 2010
vzsun=7.25

# for observed <v^2>
# relative to sun
vz_vmock=vz_vmock+vzsun
v2m_obs= compute_v2m(z_vmock,vz_vmock,zoff_obs,z_obs,h_obs)
nsbin_obs=compute_nsbin(z_vmock,zoff_obs,z_obs,h_obs)
print ' Ns bin for <v^2>=',nsbin_obs
v2m_obs_noise= v2m_obs/numpy.sqrt(nsbin_obs)
print ' <v^2> =',v2m_obs
print ' v^2 errors=',v2m_obs_noise
vmean=numpy.zeros_like(z_obs)
for jj,zo in enumerate(z_obs):
    vmean[jj]= numpy.sum(kernel(numpy.fabs(zo-z_vmock+zoff_obs),h_obs)*vz_vmock) \
         /numpy.sum(kernel(numpy.fabs(zo-z_vmock+zoff_obs),h_obs))
print ' <v>bin =',vmean

### initial model

n_m2m= 4000
# assume velocity km/s, distance kpc unit 
sigma_init= 15.0
E_m2m= numpy.random.exponential(scale=sigma_init**2.,size=n_m2m)
phi_m2m= numpy.random.uniform(size=n_m2m)*2.*numpy.pi
# although it is guess
# scale with 220 km/s and 8 kpc
sigma_true=15.0
# omega = sqrt(2)x(v (km/s))/z (kpc)
omega_true= 1.4*15.0/0.2
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
  xrange=[-.25,0.25],yrange=[0.1,10.],gcf=True)
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
_= plt.hist(vz_vmock,histtype='step',lw=2.,normed=True,bins=31,zorder=2)
_= plt.hist(vz_m2m,histtype='step',lw=2.,normed=True,bins=31,ls='--',zorder=0)
plt.xlabel(r'$v_z$')
plt.ylabel(r'$p(v_z)$')
print("Velocity dispersions: mock, fit",numpy.std(vz_vmock),\
      numpy.sqrt(numpy.sum(w_out*(vz_m2m-numpy.sum(w_out*vz_m2m)/numpy.sum(w_out))**2.)/numpy.sum(w_out)))

# plt.tight_layout()
plt.show()

### run M2M with variable zsun

step= numpy.pi/20.0
nstep= 10000
eps= 10.**-4.
eps_zo= eps/500.
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
_= plt.hist(vz_vmock,histtype='step',lw=2.,normed=True,bins=31,zorder=2)
_= plt.hist(vz_m2m,histtype='step',lw=2.,normed=True,bins=31,ls='--',zorder=0)
plt.xlabel(r'$v_z$')
plt.ylabel(r'$p(v_z)$')
print("Velocity dispersions: mock, fit",numpy.std(vz_vmock),\
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

