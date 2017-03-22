
#
# read output from dens1d_fstar.py and run hom2m
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
print ' Vzsun assumed=',vzsun

### initial model

# estimated zsun from above
zsun_obs=0.00287
print 'zsun_obs set =',zsun_obs

n_m2m= 4000
# assume velocity km/s, distance kpc unit 
sigma_init= 18.2
# although it is guess
# scale with 220 km/s and 8 kpc
sigma_true=sigma_init
# omega = sqrt(2)x(v (km/s))/z (kpc)
# omega_true= 1.4*15.0/0.2-10.0
omega_true= 70.0
omega_m2m=omega_true
E_m2m= numpy.random.exponential(scale=sigma_init**2.,size=n_m2m)
phi_m2m_omega= numpy.random.uniform(size=n_m2m)*2.*numpy.pi
A_m2m_omega= numpy.sqrt(2.*E_m2m)/omega_m2m
w_init= numpy.ones(n_m2m)
z_m2m= A_m2m_omega*numpy.cos(phi_m2m_omega)
z_out= numpy.linspace(-0.3,0.3,101)
dens_init= compute_dens(z_m2m,zsun_obs,z_out,h_m2m,w_init)
vz_m2m= -omega_m2m*A_m2m_omega*numpy.sin(phi_m2m_omega)
v2m_init= compute_v2m(z_m2m,vz_m2m,zsun_obs,z_out,h_m2m,w_init)

# plot the observational data and initial model
bovy_plot.bovy_print(axes_labelsize=17.,text_fontsize=12.,xtick_labelsize=15.,ytick_labelsize=15.)

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
                    xrange=[-.25,0.25],yrange=[10.0,50.0],gcf=True)
bovy_plot.bovy_plot(z_obs,numpy.sqrt(v2m_obs),'o',semilogy=True,overplot=True)
bovy_plot.bovy_plot(z_out,numpy.sqrt(v2m_init),'-',semilogy=True,overplot=True,zorder=0)
plt.errorbar(z_obs,numpy.sqrt(v2m_obs),yerr=numpy.sqrt(v2m_obs_noise),marker='None',ls='none',color=sns.color_palette()[1])
# plt.yscale('log',nonposy='clip')

plt.show()

### run M2M with variable omega using a fixed zsun
step=numpy.pi/20.0
nstep= 40000
eps= 10.**-6
# eps_vel= eps*n_m2m
eps_vel= eps
eps_omega= eps*100.0
skipomega= 40
mu= 1.0
h_m2m= h_obs
zsun_m2m=zsun_obs
opt_w_first= True
if opt_w_first:
    w_out,Q= run_m2m_weights_wv2m(w_init,A_m2m_omega,phi_m2m_omega,omega_m2m,zsun_m2m,
                                 z_obs,dens_obs,dens_obs_noise,v2m_obs,v2m_obs_noise,
                                 nstep=numpy.amin([nstep,10000]),
                                 step=step,mu=mu,eps=eps,eps_vel=eps_vel,
                                 h_m2m=h_m2m)
else:
    w_out= w_init
if True:
# delta_omega set to be 0.3 x omega_true
    (w_out,omega_out,z_m2m,vz_m2m),Q= \
     run_m2m_weights_omega_densv2m(w_out,A_m2m_omega,phi_m2m_omega, \
       omega_m2m,zsun_m2m,z_obs,dens_obs,dens_obs_noise,v2m_obs,v2m_obs_noise, \
       nstep=nstep,step=step,mu=mu,skipomega=skipomega, \
       eps=eps,eps_vel=eps_vel,h_m2m=h_m2m,eps_omega=eps_omega, \
       delta_omega=omega_true*0.3)

z_out= numpy.linspace(-0.3,0.3,101)
dens_final= compute_dens(z_m2m,zsun_obs,z_out,h_obs,w_out)

# output
print("Velocity dispersions: fit",\
   numpy.sqrt(numpy.sum(w_out*(vz_m2m-numpy.sum(w_out*vz_m2m)/numpy.sum(w_out))**2.)/numpy.sum(w_out)))
print("omega: fit, starting point",numpy.median(omega_out[-1000:-1]),omega_m2m)

# plot
bovy_plot.bovy_print(axes_labelsize=17.,text_fontsize=12.,xtick_labelsize=15.,ytick_labelsize=15.)

#figsize(15,6)
plt.subplot(2,3,1)
bovy_plot.bovy_plot(z_out,dens_init,'-',semilogy=True,
                   xlabel=r'$\tilde{z}$',ylabel=r'$\nu_{\mathrm{obs}}(\tilde{z})$',
                   xrange=[-.25,0.25],yrange=[0.1,10.],gcf=True)
bovy_plot.bovy_plot(z_obs,dens_obs,'o',semilogy=True,overplot=True)
bovy_plot.bovy_plot(z_out,dens_final,'-',semilogy=True,overplot=True,zorder=0)
plt.errorbar(z_obs,dens_obs,yerr=dens_obs_noise,marker='None',ls='none',color=sns.color_palette()[1])
plt.yscale('log',nonposy='clip')
plt.subplot(2,3,2)
v2m_final= compute_v2m(z_m2m,vz_m2m,zsun_obs,z_out,h_obs,w_out)
bovy_plot.bovy_plot(z_out,v2m_init,'-',semilogy=False,
                   xlabel=r'$\tilde{z}$',ylabel=r'$\langle v^2\rangle^{1/2}$',
                    xrange=[-.25,0.25],yrange=[0.0,100.0],gcf=True)
bovy_plot.bovy_plot(z_obs,numpy.sqrt(v2m_obs),'o',semilogy=False,overplot=True)
bovy_plot.bovy_plot(z_out,numpy.sqrt(v2m_final),'-',semilogy=False,overplot=True,zorder=0)
plt.errorbar(z_obs,numpy.sqrt(v2m_obs),yerr=numpy.sqrt(v2m_obs_noise),marker='None',ls='none',color=sns.color_palette()[1])
# plt.yscale('log',nonposy='clip')
plt.subplot(2,3,3)
bovy_plot.bovy_plot(numpy.linspace(0.,1.,nstep)*nstep*step*numpy.mean(omega_true)/2./numpy.pi,omega_out,'-',
                   xlabel=r'$\mathrm{orbits}$',ylabel=r'$\omega(t)$',gcf=True,semilogx=True,zorder=1,
                   yrange=[omega_m2m/1.3,numpy.amax(omega_out)*1.2])
plt.axhline(omega_true,color=sns.color_palette()[1],lw=2.,zorder=0)
plt.ylim(0.0,200.0)
plt.subplot(2,3,4)
bovy_plot.bovy_plot(A_m2m_omega,w_out,'k.',xlabel=r'$z_{\mathrm{max}}$',ylabel=r'$w(z_{\mathrm{max}})$',
                   yrange=[0.,5.],gcf=True)
sindx= numpy.argsort(A_m2m_omega)
w_expect= numpy.exp((A_m2m_omega[sindx]*omega_true)**2./2.*(1./sigma_init**2.-1./sigma_true**2.))
w_expect/= numpy.sum(w_expect)/len(w_expect)
plt.plot(A_m2m_omega[sindx],w_expect,lw=2.)
plt.subplot(2,3,5)
bovy_plot.bovy_plot(numpy.linspace(0.,1.,nstep)*nstep*step*omega_true/2./numpy.pi,numpy.sum(Q,axis=1),lw=1.,
                   semilogx=True,xlabel=r'$\mathrm{orbits}$',ylabel=r'$F$',gcf=True)
plt.subplot(2,3,6)
_= plt.hist(vz_m2m,weights=w_out,histtype='step',lw=2.,normed=True,bins=31,zorder=1)
# _= plt.hist(vz_vmock,histtype='step',lw=2.,normed=True,bins=31,zorder=2)
_= plt.hist(vz_m2m,histtype='step',lw=2.,normed=True,bins=31,ls='--',zorder=0)
plt.xlabel(r'$v_z$')
plt.ylabel(r'$p(v_z)$')

plt.show()
