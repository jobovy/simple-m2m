
#
# 1. read ../TGAS/TGASTycho_J2000.fits from CDS with J2000 coordinates
#  displace the data using the uncertainty
#  analyse Vz distribution for stars b small enough. 
#

import pyfits
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import stats
from galpy.util import bovy_coords
from galpy.util import bovy_plot
import matplotlib.cm as cm
import seaborn as sns
from StringIO import StringIO
from hom2m_sc1d import *
from astropy.modeling import models, fitting

# input parameters
eplxlim=0.1
# Vt mag limit
vtlim=11.5
# colour range, F0V and F9V
bvtcolmin=0.317
bvtcolmax=0.587
# A0V and A9V
# bvtcolmin=0.013
# bvtcolmax=0.279
# G0V and G9V
# bvtcolmin=0.650
# bvtcolmax=0.894
# RAVE radial velocity error limit
hrverrlim=10.0
hrvlim=200.0
# number of error sampling 
nsamp=100

# radius of a cylinder for density calculation (kpc)
# rxylim=0.1
rxylim=0.2
# parallax limit (mas)
plxmin=1.0/(0.3)
# glat limit
glatlim=20.0

numpy.random.seed(4)


# print input parameters
print ' eplxlim=',eplxlim
print ' Vtlim=',vtlim
print ' Bv-Vt colour range=',bvtcolmin,bvtcolmax
print ' Cylinder region radius within (kpc)=',rxylim
print ' min parallax (mas) =',plxmin,' max distance (kpc)=',1.0/plxmin
print ' |b|<',glatlim

# input data
infile='../TGAS/TGASTycho_J2000.fits'
rstar_hdus=pyfits.open(infile)
rstar=rstar_hdus[1].data
rstar_hdus.close()

# read the data
# number of data points
nstar_alls=len(rstar['SolID'])
print 'number of all TGAS stars=',nstar_alls

# select F stars with mag limit
# parallax error limit does not make a big difference
# mcsindx=np.where((rstar['e_Plx']/rstar['Plx']<eplxlim) \
# radial velocity error and |Vhel| should be limited. 
mcsindx=np.where((rstar['VTmag']<vtlim) \
  & (rstar['BTmag']-rstar['VTmag']>bvtcolmin) \
  & (rstar['BTmag']-rstar['VTmag']<bvtcolmax) \
  & (np.fabs(rstar['e_Plx']/rstar['Plx'])<eplxlim) & (rstar['Plx']>0.0) \
  & (np.fabs(rstar['_Glat'])<glatlim))

nstar_mcs=len(mcsindx[0])

print ' number of stars after mag, colour cut =',nstar_mcs

# extract the necessary info
# use J2000 position calculated by CDS
ra_mcs=rstar['_RAJ2000'][mcsindx]
dec_mcs=rstar['_DEJ2000'][mcsindx]
plx_mcs=rstar['Plx'][mcsindx]
eplx_mcs=rstar['e_Plx'][mcsindx]
vt_mcs=rstar['VTmag'][mcsindx]
bt_mcs=rstar['BTmag'][mcsindx]
pmra_mcs=rstar['pmRA'][mcsindx]
pmdec_mcs=rstar['pmDE'][mcsindx]
pmraerr_mcs=rstar['e_pmRA'][mcsindx]
pmdecerr_mcs=rstar['e_pmDE'][mcsindx]

# analyse vz distribution
nbin=70
vmin=-70.0
vmax=70.0
vzhist=np.zeros(nbin)
nvzhist_samp=np.zeros((nsamp,nbin))

# nbin=30
# vmin=-50.0
# vmax=50.0
# vzhist=np.zeros(nbin)
# nvzhist_samp=np.zeros((nsamp,nbin))

isamp=0
while isamp<nsamp:
# sample parallax values within the error
  plxsamp_mcs=plx_mcs+eplx_mcs*np.random.normal(size=plx_mcs.shape)
#  plxsamp_mcs=plx_mcs

# only choose parallax > plxmin
  psindx=np.where(plxsamp_mcs>plxmin)

  nstar_ps=len(psindx[0])

#  print 'number of stars after parallax limit=',nstar_ps

  ra_ps=ra_mcs[psindx]
  dec_ps=dec_mcs[psindx]
  plx_ps=plxsamp_mcs[psindx]
  eplx_ps=eplx_mcs[psindx]
  vt_ps=vt_mcs[psindx]
  bt_ps=bt_mcs[psindx]
  pmra_ps=pmra_mcs[psindx]
  pmdec_ps=pmdec_mcs[psindx]
  pmraerr_ps=pmraerr_mcs[psindx]
  pmdecerr_ps=pmdecerr_mcs[psindx]
  hrv_ps=np.zeros(nstar_ps)
  hrverr_ps=np.zeros(nstar_ps)

# displace the velocity data using errors
  pmra_ps=pmra_ps+pmraerr_ps*np.random.normal(size=pmra_ps.shape)
  pmdec_ps=pmdec_ps+pmdecerr_ps*np.random.normal(size=pmdec_ps.shape)
  hrv_ps=hrv_ps+hrverr_ps*np.random.normal(size=hrv_ps.shape)

# calculate x,y,z coordinate
# ra,dec -> glon
  Tlb=bovy_coords.radec_to_lb(ra_ps,dec_ps \
    ,degree=True,epoch=2000.0)
  glon_ps=Tlb[:,0]
  glat_ps=Tlb[:,1]
# deg to radian
  glon_ps*=(np.pi/180.0)
  glat_ps*=(np.pi/180.0)

# convert proper motion from mu_alpha*,delta to mu_l*,b using bovy_coords
  pmlon_ps=np.zeros(nstar_ps)
  pmlat_ps=np.zeros(nstar_ps)
  Tpmlb=bovy_coords.pmrapmdec_to_pmllpmbb(pmra_ps,pmdec_ps,ra_ps,dec_ps \
    ,degree=True,epoch=2000.0)
# deg to radian
  pmlon_ps=Tpmlb[:,0]
  pmlat_ps=Tpmlb[:,1]
# calculate vz
  Tvxvyvz=bovy_coords.vrpmllpmbb_to_vxvyvz(hrv_ps,pmlon_ps,pmlat_ps,glon_ps,glat_ps,1.0/plx_ps)
  velx_ps=Tvxvyvz[:,0]
  vely_ps=Tvxvyvz[:,1]
  velz_ps=Tvxvyvz[:,2]
  pmvconst=4.74047

# this is consistent with bovy_coords, but use bovy_coords
#  velz_ps=np.cos(glat_ps)*pmvconst*pmlat_ps/plx_ps \
#   +np.sin(glat_ps)*hrv_ps


# x,y,z position relative to the sun (kpc)
  rxy_ps=np.cos(glat_ps)/plx_ps

# cylinder selection
  csindx=np.where((rxy_ps<rxylim))
#  csindx=np.where((rxy_ps<rxylim) & (np.fabs(glat_ps)<45.0*np.pi/180.0))

# position for selected stars
  xpos_cs=rxy_ps[csindx]*np.cos(glon_ps[csindx])
  ypos_cs=rxy_ps[csindx]*np.sin(glon_ps[csindx])
  zpos_cs=np.sin(glat_ps[csindx])/plx_ps[csindx]
  plx_cs=plx_ps[csindx]
  vt_cs=vt_ps[csindx]
  bt_cs=bt_ps[csindx]
  velx_cs=velx_ps[csindx]
  vely_cs=vely_ps[csindx]
  velz_cs=velz_ps[csindx]
  glon_cs=glon_ps[csindx]
  glat_cs=glat_ps[csindx]
  vlon_cs=pmvconst*pmlon_ps[csindx]/plx_ps[csindx]
  vlat_cs=pmvconst*pmlat_ps[csindx]/plx_ps[csindx]
  pmraerr_cs=pmvconst \
            *np.sqrt((pmraerr_ps[csindx]**2+(pmra_ps[csindx]**2) \
            *(plx_ps[csindx]**(-2))*(eplx_ps[csindx]**2))\
            *(plx_ps[csindx]**(-2)))
  pmdecerr_cs=pmvconst \
           *np.sqrt((pmdecerr_ps[csindx]**2+(pmdec_ps[csindx]**2) \
            *(plx_ps[csindx]**(-2))*(eplx_ps[csindx]**2))\
            *(plx_ps[csindx]**(-2)))
  hrverr_cs=hrverr_ps[csindx]
  hrv_cs=hrv_ps[csindx]

  nstar_cs=len(xpos_cs)

# keep vz distribution
  velzhist=np.histogram(velz_cs,nbin,(vmin,vmax),density=True)
  if isamp==0:
# set vzhist
    i=0
    while i<nbin:
      vzhist[i]=0.5*(velzhist[1][i+1]+velzhist[1][i])
      i+=1

  nvzhist_samp[isamp,:]=velzhist[0]

# output ascii data for test
  if isamp==0:
    f=open('vzdist_starvz_samp.asc','w')
    i=0
    while i < nstar_cs:
      print >>f, "%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f" \
        %(xpos_cs[i],ypos_cs[i],zpos_cs[i], \
        plx_cs[i],bt_cs[i],vt_cs[i],velz_cs[i], \
        pmraerr_cs[i],pmdecerr_cs[i],hrverr_cs[i],hrv_cs[i],vlon_cs[i], \
        vlat_cs[i],velx_cs[i],vely_cs[i],glon_cs[i],glat_cs[i])
      i+=1
    f.close()

  if isamp==0:
    print ' mean velocity errors Vhel,Vra,Vdec=',np.mean(hrverr_cs) \
     ,np.mean(pmraerr_cs),np.mean(pmdecerr_cs)

  isamp+=1

# vz distribution
vzdist=np.mean(nvzhist_samp,axis=0)
vzdist_noise=np.std(nvzhist_samp,axis=0)

# 1 Gaussian fit
g_init=models.Gaussian1D(amplitude=0.04, mean=0, stddev=10.0)
fit_g=fitting.LevMarLSQFitter()
g=fit_g(g_init,vzhist,vzdist)
print g
# plot
plt.figure(figsize=(8,5))
plt.plot(vzhist,vzdist,'ko')
plt.errorbar(vzhist,vzdist,yerr=vzdist_noise,marker='None',ls='none',color=sns.color_palette()[1])
plt.plot(vzhist,g(vzhist))
plt.show()

# 2 Gaussian fit
gg_init = models.Gaussian1D(0.04,-7.0, 10.0) + models.Gaussian1D(0.001, -7.0, 30.0)
fitter = fitting.SLSQPLSQFitter()
gg_fit = fitter(gg_init,vzhist,vzdist)
print gg_fit
#print ' Gaussian fit, mean, stddev=',gg_fit.mean,gg_fit.stddev
# plot
plt.figure(figsize=(8,5))
plt.plot(vzhist,vzdist,'ko')
plt.errorbar(vzhist,vzdist,yerr=vzdist_noise,marker='None',ls='none',color=sns.color_palette()[1])
plt.plot(vzhist,gg_fit(vzhist))
plt.show()

