
#
# 1. read ../TGAS/TGASTycho_J2000.fits from CDS with J2000 coordinates
#  displace the data using the uncertainty
#  output the stars within the cylinder
#  calculate 1D density within a cylinder
# 2. read ../RAVE/DR5/ver3/RAVE_DR5xTGASJ2000_MatchAB.fits 
#    RAVE x TGAS with J2000 RA, DEC with matched flag A or B
#  output the stars' position and vz within the cylinder
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

# input parameters
# eplx/plx limit pi=parallax
eplxlim=0.15
# Vt mag limit
vtlim=11.5
# colour range, F0V and F9V
bvtcolmin=0.317
bvtcolmax=0.587
# RAVE radial velocity error limit
hrverrlim=10.0
hrvlim=200.0
# number of error sampling 
nsamp=100

# radius of a cylinder for density calculation (kpc)
# rxylim=0.1
rxylim=0.2
# nbin and z range max and min for North (_n) and South (_s)
# z_obs= numpy.array([0.075,0.1,0.125,0.15,0.175,-0.075,-0.1,-0.125,-0.15,-0.175])
z_obs= numpy.array([0.0,0.025,0.05,0.075,0.1,0.125,0.15,0.175,-0.025,-0.05,-0.075,-0.1,-0.125,-0.15,-0.175])
h_obs= 0.075
zoff_obs=0.0
zmax=h_obs+0.175

numpy.random.seed(4)

# parallax limit (mas)
plxmin=1.0/(np.sqrt(rxylim**2+zmax**2))

# print input parameters
print ' eplxlim=',eplxlim
print ' Vtlim=',vtlim
print ' Bv-Vt colour range=',bvtcolmin,bvtcolmax
print ' Cylinder region radius within (kpc)=',rxylim
print ' min parallax (mas) =',plxmin,' max distance (kpc)=',1.0/plxmin

print ' measure <density>'

# input data
infile='../TGAS/TGASTycho_J2000.fits'
rstar_hdus=pyfits.open(infile)
rstar=rstar_hdus[1].data
rstar_hdus.close()

# read the data
# number of data points
nstar_alls=len(rstar['SolID'])
print 'number of all TGAS stars=',nstar_alls

# select stars with mag and colour limit
mcsindx=np.where((rstar['VTmag']<vtlim) \
  & (rstar['BTmag']-rstar['VTmag']>bvtcolmin) \
  & (rstar['BTmag']-rstar['VTmag']<bvtcolmax))

nstar_mcs=len(mcsindx[0])

print ' number of stars with magnitude and colour =',nstar_mcs

# extract the necessary info
# use J2000 position calculated by CDS
ra_mcs=rstar['_RAJ2000'][mcsindx]
dec_mcs=rstar['_DEJ2000'][mcsindx]
plx_mcs=rstar['Plx'][mcsindx]
eplx_mcs=rstar['e_Plx'][mcsindx]
vt_mcs=rstar['VTmag'][mcsindx]
bt_mcs=rstar['BTmag'][mcsindx]

dens_obs_samp=np.zeros((nsamp,len(z_obs)))

isamp=0
while isamp<nsamp:
# sample parallax values within the error
  plxsamp_mcs=plx_mcs+eplx_mcs*np.random.normal(size=plx_mcs.shape)

# only choose parallax > plxmin
  psindx=np.where(plxsamp_mcs>plxmin)

#  print isamp,' number of stars after parallax limit =',len(psindx[0])

# extract the necessary info
  ra_ps=ra_mcs[psindx]
  dec_ps=dec_mcs[psindx]
  plx0_ps=plx_mcs[psindx]
  plx_ps=plxsamp_mcs[psindx]
  eplx_ps=eplx_mcs[psindx]
  vt_ps=vt_mcs[psindx]
  bt_ps=bt_mcs[psindx]

# for check
#  file_plx='plx_sample'+str(isamp)+'.asc'
#  f=open(file_plx,'w')
#  i=0
#  while i < len(psindx[0]):
#    print >>f, "%f %f %f" %(plx0_ps[i],plx_ps[i],eplx_ps[i])
#    i+=1
#  f.close()

# calculate x,y,z coordinate
# ra,dec -> glon
  Tlb=bovy_coords.radec_to_lb(ra_ps,dec_ps \
    ,degree=True,epoch=2000.0)
  glon_ps=Tlb[:,0]
  glat_ps=Tlb[:,1]
# deg to radian
  glon_ps*=(np.pi/180.0)
  glat_ps*=(np.pi/180.0)

# x,y,z position relative to the sun (kpc)
  rxy_ps=np.cos(glat_ps)/plx_ps

# cylinder selection
  csindx=np.where(rxy_ps<rxylim)

# position for selected stars
  xpos_cs=rxy_ps[csindx]*np.cos(glon_ps[csindx])
  ypos_cs=rxy_ps[csindx]*np.sin(glon_ps[csindx])
  zpos_cs=np.sin(glat_ps[csindx])/plx_ps[csindx]
  plx_cs=plx_ps[csindx]
  vt_cs=vt_ps[csindx]
  bt_cs=bt_ps[csindx]

  nstar_cs=len(xpos_cs)

#  print ' number of stars after cylinder selection =',nstar_cs

# output ascii data for test
#  file_cs='cylinder_star'+str(isamp)+'.asc'
#  f=open(file_cs,'w')
#  i=0
#  while i < nstar_cs:
#    print >>f, "%f %f %f %f %f %f" %(xpos_cs[i],ypos_cs[i],zpos_cs[i], \
#     plx_cs[i],bt_cs[i],vt_cs[i])
#    i+=1
#  f.close()

# calculate density
  dens_obs= compute_dens(zpos_cs,zoff_obs,z_obs,h_obs)

# save dens 
  dens_obs_samp[isamp,:]=dens_obs

#  print ' z,dens=',z_obs,dens_obs

# output ascii data for test
#  file_cs='cylinder_dens'+str(isamp)+'.asc'
#  f=open(file_cs,'w')
#  i=0
#  while i < len(z_obs):
#    print >>f, "%f %f" %(z_obs[i],dens_obs[i])
#    i+=1
#  f.close()

  isamp+=1

# get average and dispersion
dens_obs=np.mean(dens_obs_samp,axis=0)
dens_obs_noise=np.std(dens_obs_samp,axis=0)

print 'z,dens,noise=',z_obs,dens_obs,dens_obs_noise

# output ascii data for test
file_cs='cylinder_dens.asc'
f=open(file_cs,'w')
i=0
while i < len(z_obs):
  print >>f, "%f %f %f" %(z_obs[i],dens_obs[i],dens_obs_noise[i])
  i+=1
f.close()

# output fits file
# header
prihdr=pyfits.Header()
prihdr['h_obs']=h_obs
prihdu=pyfits.PrimaryHDU(header=prihdr)
tbhdu=pyfits.BinTableHDU.from_columns([\
  pyfits.Column(name='z_obs',format='E',array=z_obs),\
  pyfits.Column(name='dens_obs',format='E',array=dens_obs),\
  pyfits.Column(name='dens_obs_noise',format='E',array=dens_obs_noise)])
thdulist=pyfits.HDUList([prihdu,tbhdu])
thdulist.writeto('cylinder_dens.fits',clobber=True)

bovy_plot.bovy_print(axes_labelsize=17.,text_fontsize=12.,xtick_labelsize=15.,ytick_labelsize=15.)
# plt.figure(figsize(6,4))
bovy_plot.bovy_plot(z_obs,dens_obs,'ko',semilogy=True,
  xlabel=r'$\tilde{z}$',ylabel=r'$\nu_{\mathrm{obs}}(\tilde{z})$',
  xrange=[-.25,0.25],yrange=[0.1,10.],gcf=True)
plt.errorbar(z_obs,dens_obs,yerr=dens_obs_noise,marker='None',ls='none',color='k')
i=0

  
plt.show()

########## new sample for velocity tracers ##########

print ' measure <v^2>'

# compute <v>
def compute_vm(z,vz,zsun,z_obs,h_obs,w=None):
    if w is None: w= numpy.ones_like(z)
    vm= numpy.zeros_like(z_obs)
    for jj,zo in enumerate(z_obs):
        vm[jj]= numpy.sum(w*kernel(numpy.fabs(zo-z+zsun),h_obs)*vz) \
          /numpy.sum(w*kernel(numpy.fabs(zo-z+zsun),h_obs))
    return vm

# input data
infile='../RAVE/DR5/ver3/RAVE_DR5xTGASJ2000_MatchAB.fits'
rstar_hdus=pyfits.open(infile)
rstar=rstar_hdus[1].data
rstar_hdus.close()

# read the data
# number of data points
nstar_alls=len(rstar['SolID'])
print 'number of all TGASxRAVE stars=',nstar_alls

# select F stars with mag limit
# parallax error limit does not make a big difference
# mcsindx=np.where((rstar['e_Plx']/rstar['Plx']<eplxlim) \
# radial velocity error and |Vhel| should be limited. 
mcsindx=np.where((rstar['VTmag']<vtlim) \
  & (rstar['BTmag']-rstar['VTmag']>bvtcolmin) \
  & (rstar['BTmag']-rstar['VTmag']<bvtcolmax) \
  & (rstar['eHRV']<hrverrlim) & (np.fabs(rstar['HRV'])<hrvlim))

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
pmra_mcs=rstar['pmRA_1'][mcsindx]
pmdec_mcs=rstar['pmDE'][mcsindx]
pmraerr_mcs=rstar['e_pmRA'][mcsindx]
pmdecerr_mcs=rstar['e_pmDE'][mcsindx]
hrv_mcs=rstar['HRV'][mcsindx]
hrverr_mcs=rstar['eHRV'][mcsindx]
logg_mcs=rstar['logg_N_K'][mcsindx]

v2m_obs_samp=np.zeros((nsamp,len(z_obs)))

isamp=0
while isamp<nsamp:
# sample parallax values within the error
  plxsamp_mcs=plx_mcs+eplx_mcs*np.random.normal(size=plx_mcs.shape)

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
  hrv_ps=hrv_mcs[psindx]
  hrverr_ps=hrverr_mcs[psindx]
  logg_ps=logg_mcs[psindx]

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
  pmvconst=4.74047
  velz_ps=np.cos(glat_ps)*pmvconst*pmlat_ps/plx_ps \
   +np.sin(glat_ps)*hrv_ps

# x,y,z position relative to the sun (kpc)
  rxy_ps=np.cos(glat_ps)/plx_ps

# cylinder selection
  csindx=np.where(rxy_ps<rxylim)

# position for selected stars
  xpos_cs=rxy_ps[csindx]*np.cos(glon_ps[csindx])
  ypos_cs=rxy_ps[csindx]*np.sin(glon_ps[csindx])
  zpos_cs=np.sin(glat_ps[csindx])/plx_ps[csindx]
  plx_cs=plx_ps[csindx]
  vt_cs=vt_ps[csindx]
  bt_cs=bt_ps[csindx]
  velz_cs=velz_ps[csindx]
  glon_cs=glon_ps[csindx]
  glat_cs=glon_ps[csindx]
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
  logg_cs=logg_ps[csindx]

  nstar_cs=len(xpos_cs)

# output ascii data for test
#   f=open('cylinder_starvz_samp.asc','w')
#   i=0
#   while i < nstar_cs:
#    print >>f, "%f %f %f %f %f %f %f %f %f %f %f %f %f %f" \
#      %(xpos_cs[i],ypos_cs[i],zpos_cs[i], \
#      plx_cs[i],bt_cs[i],vt_cs[i],velz_cs[i], \
#        pmraerr_cs[i],pmdecerr_cs[i],hrverr_cs[i],hrv_cs[i],vlon_cs[i] \
#        ,vlat_cs[i],logg_cs[i])
#    i+=1
#  f.close()
  if isamp==0:
    print ' mean velocity errors Vhel,Vra,Vdec=',np.mean(hrverr_cs) \
     ,np.mean(pmraerr_cs),np.mean(pmdecerr_cs)

  v2m_obs= compute_v2m(zpos_cs,velz_cs,zoff_obs,z_obs,h_obs)
  vm_obs= compute_vm(zpos_cs,velz_cs,zoff_obs,z_obs,h_obs)
#  print ' <v^2> =',v2m_obs
#  print ' <v> =',vm_obs
# set v2m as sig_z^2
  v2m_obs=(v2m_obs-vm_obs**2)
#  print ' new <v^2>^1/3 ',np.sqrt(v2m_obs)

# save v2m
  v2m_obs_samp[isamp,:]=v2m_obs

  isamp+=1

# get average and dispersion
v2m_obs=np.mean(v2m_obs_samp,axis=0)
v2m_obs_noise=np.std(v2m_obs_samp,axis=0)

print 'v2m,err=',v2m_obs,v2m_obs_noise

file_cs='cylinder_v2m.asc'
f=open(file_cs,'w')
i=0
while i < len(z_obs):
  print >>f, "%f %f %f %f" %(z_obs[i],v2m_obs[i],vm_obs[i],v2m_obs_noise[i])
  i+=1
f.close()

# output fits file
# header
prihdr=pyfits.Header()
prihdr['h_obs']=h_obs
prihdu=pyfits.PrimaryHDU(header=prihdr)
tbhdu=pyfits.BinTableHDU.from_columns([\
  pyfits.Column(name='z_obs',format='E',array=z_obs),\
  pyfits.Column(name='v2m_obs',format='E',array=v2m_obs),\
  pyfits.Column(name='v2m_obs_noise',format='E',array=v2m_obs_noise)])
thdulist=pyfits.HDUList([prihdu,tbhdu])
thdulist.writeto('cylinder_v2m.fits',clobber=True)

bovy_plot.bovy_print(axes_labelsize=17.,text_fontsize=12.,xtick_labelsize=15.,ytick_labelsize=15.)
bovy_plot.bovy_plot(z_obs,numpy.sqrt(v2m_obs),'o',semilogy=False,
                   xlabel=r'$\tilde{z}$',ylabel=r'$\langle v^2\rangle^{1/2}$',
                    xrange=[-.25,0.25],yrange=[10.0,100.0],gcf=True)
plt.errorbar(z_obs,numpy.sqrt(v2m_obs),yerr=numpy.sqrt(v2m_obs_noise),marker='None',ls='none',color=sns.color_palette()[1])
# plt.yscale('log',nonposy='clip')

plt.show()

