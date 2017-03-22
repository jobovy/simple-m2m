
#
# 1. read ../TGAS/TGASTycho_J2000.fits from CDS with J2000 coordinates
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
from StringIO import StringIO

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
hrvlim=100.0

# radius of a cylinder for density calculation (kpc)
# rxylim=0.1
rxylim=0.2
# nbin and z range max and min for North (_n) and South (_s)
nbin_n=5
zmin_n=0.0
zmax_n=0.25
nbin_s=5
zmin_s=-0.25
zmax_s=0.0

# print input parameters
print ' eplxlim=',eplxlim
print ' Vtlim=',vtlim
print ' Bv-Vt colour range=',bvtcolmin,bvtcolmax
print ' Cylinder region radius within (kpc)=',rxylim
print ' z bin North nbin,zmin,zmax=',nbin_n,zmin_n,zmax_n
print ' z bin South nbin,zmin,zmax=',nbin_s,zmin_s,zmax_s

# input data
infile='../TGAS/TGASTycho_J2000.fits'
rstar_hdus=pyfits.open(infile)
rstar=rstar_hdus[1].data
rstar_hdus.close()

# read the data
# number of data points
nstar_alls=len(rstar['SolID'])
print 'number of all TGAS stars=',nstar_alls

# select only stars with parallax smaller than a limit 
epsindx=np.where((rstar['e_Plx']/rstar['Plx']<eplxlim) & (rstar['Plx']>0.0) \
  & (rstar['VTmag']<vtlim) & (rstar['BTmag']-rstar['VTmag']>bvtcolmin) \
  & (rstar['BTmag']-rstar['VTmag']<bvtcolmax))

nstar_eps=len(epsindx[0])

print ' number of stars after parallax cut of eplx/plx=',eplxlim,' =',nstar_eps

# extract the necessary info
# use J2000 position calculated by CDS
ra_eps=rstar['_RAJ2000'][epsindx]
dec_eps=rstar['_DEJ2000'][epsindx]
plx_eps=rstar['Plx'][epsindx]
vt_eps=rstar['VTmag'][epsindx]
bt_eps=rstar['BTmag'][epsindx]

# calculate x,y,z coordinate
# ra,dec -> glon
Tlb=bovy_coords.radec_to_lb(ra_eps,dec_eps \
  ,degree=True,epoch=2000.0)
glon_eps=Tlb[:,0]
glat_eps=Tlb[:,1]
# deg to radian
glon_eps*=(np.pi/180.0)
glat_eps*=(np.pi/180.0)

# x,y,z position relative to the sun (kpc)
rxy_eps=np.cos(glat_eps)/plx_eps

# cylinder selection
csindx=np.where(rxy_eps<rxylim)

# position for selected stars
xpos_cs=rxy_eps[csindx]*np.cos(glon_eps[csindx])
ypos_cs=rxy_eps[csindx]*np.sin(glon_eps[csindx])
zpos_cs=np.sin(glat_eps[csindx])/plx_eps[csindx]
plx_cs=plx_eps[csindx]
vt_cs=vt_eps[csindx]
bt_cs=bt_eps[csindx]

nstar_cs=len(xpos_cs)

print ' number of stars after cylinder selection =',nstar_cs

# output ascii data for test
f=open('cylinder_star.asc','w')
i=0
while i < nstar_cs:
  print >>f, "%f %f %f %f %f %f" %(xpos_cs[i],ypos_cs[i],zpos_cs[i], \
   plx_cs[i],bt_cs[i],vt_cs[i])
  i+=1
f.close()

# output fits data
tbhdu=pyfits.BinTableHDU.from_columns([\
  pyfits.Column(name='xpos',format='E',array=xpos_cs),\
  pyfits.Column(name='ypos',format='E',array=ypos_cs),\
  pyfits.Column(name='zpos',format='E',array=zpos_cs),\
  pyfits.Column(name='plx',format='E',array=plx_cs),\
  pyfits.Column(name='Bt',format='E',array=bt_cs),\
  pyfits.Column(name='Vt',format='E',array=vt_cs)])
tbhdu.writeto('selstars_xyz.fits',clobber=True)

# plot selected data point
# plt.scatter(xpos_cs,zpos_cs,c=vt_cs,s=3,vmin=0.0,vmax=11.0)
# plt.xlabel(r"X (kpc)",fontsize=18,fontname="serif")
# plt.ylabel(r"Z (kpc)",fontsize=18,fontname="serif")
# plt.axis([-0.1,0.1,-0.2,0.2],'scaled')
# cbar=plt.colorbar()
# cbar.set_label(r'VT mag')
# plt.xticks(fontsize=18)
# plt.yticks(fontsize=18)
# plt.show()

# histogram along z
zhist_n=np.histogram(zpos_cs,nbin_n,(zmin_n,zmax_n))
zhist_s=np.histogram(zpos_cs,nbin_s,(zmin_s,zmax_s))
nsz_n=zhist_n[0]
zhistbin_n=zhist_n[1]
nsz_s=zhist_s[0]
zhistbin_s=zhist_s[1]

# x data points
i=0
zbin_n=np.zeros(nbin_n)
while i<nbin_n:
  zbin_n[i]=0.5*(zhistbin_n[i]+zhistbin_n[i+1])
  i+=1
i=0
zbin_s=np.zeros(nbin_s)
while i<nbin_s:
  zbin_s[i]=0.5*(zhistbin_s[i]+zhistbin_s[i+1])
  i+=1

# show histogram
plt.scatter(zbin_n,nsz_n)
plt.scatter(zbin_s,nsz_s)
plt.xlabel(r"z (kpc)",fontsize=18,fontname="serif")
plt.ylabel(r"Ns",fontsize=18,fontname="serif")
#plt.axis([-0.25,0.25,0.0,100])
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()

########## new sample for velocity tracers ##########

# input data
infile='../RAVE/DR5/ver3/RAVE_DR5xTGASJ2000_MatchAB.fits'
rstar_hdus=pyfits.open(infile)
rstar=rstar_hdus[1].data
rstar_hdus.close()

# read the data
# number of data points
nstar_alls=len(rstar['SolID'])
print 'number of all TGASxRAVE stars=',nstar_alls

# select only stars with parallax smaller than a limit 
epsindx=np.where((rstar['e_Plx']/rstar['Plx']<eplxlim) & (rstar['Plx']>0.0) \
  & (rstar['VTmag']<vtlim) & (rstar['BTmag']-rstar['VTmag']>bvtcolmin) \
  & (rstar['BTmag']-rstar['VTmag']<bvtcolmax) & (rstar['eHRV']<hrverrlim) \
  & (np.fabs(rstar['HRV'])<hrvlim))

nstar_eps=len(epsindx[0])

print ' number of stars after parallax cut of eplx/plx=',eplxlim,' =',nstar_eps

# extract the necessary info
# use J2000 position calculated by CDS
ra_eps=rstar['_RAJ2000'][epsindx]
dec_eps=rstar['_DEJ2000'][epsindx]
plx_eps=rstar['Plx'][epsindx]
plxerr_eps=rstar['e_Plx'][epsindx]
vt_eps=rstar['VTmag'][epsindx]
bt_eps=rstar['BTmag'][epsindx]
pmra_eps=rstar['pmRA_1'][epsindx]
pmdec_eps=rstar['pmDE'][epsindx]
pmraerr_eps=rstar['e_pmRA'][epsindx]
pmdecerr_eps=rstar['e_pmDE'][epsindx]
hrv_eps=rstar['HRV'][epsindx]
hrverr_eps=rstar['eHRV'][epsindx]
logg_eps=rstar['logg_N_K'][epsindx]

# calculate x,y,z coordinate
# ra,dec -> glon
Tlb=bovy_coords.radec_to_lb(ra_eps,dec_eps \
  ,degree=True,epoch=2000.0)
glon_eps=Tlb[:,0]
glat_eps=Tlb[:,1]
# deg to radian
glon_eps*=(np.pi/180.0)
glat_eps*=(np.pi/180.0)

# convert proper motion from mu_alpha*,delta to mu_l*,b using bovy_coords
pmlon_eps=np.zeros(nstar_eps)
pmlat=np.zeros(nstar_eps)
Tpmlb=bovy_coords.pmrapmdec_to_pmllpmbb(pmra_eps,pmdec_eps,ra_eps,dec_eps \
  ,degree=True,epoch=2000.0)
# deg to radian
pmlon_eps=Tpmlb[:,0]
pmlat_eps=Tpmlb[:,1]
# calculate vz
pmvconst=4.74047
velz_eps=np.cos(glat_eps)*pmvconst*pmlat_eps/plx_eps \
 +np.sin(glat_eps)*hrv_eps

# x,y,z position relative to the sun (kpc)
rxy_eps=np.cos(glat_eps)/plx_eps

# cylinder selection
csindx=np.where(rxy_eps<rxylim)

# position for selected stars
xpos_cs=rxy_eps[csindx]*np.cos(glon_eps[csindx])
ypos_cs=rxy_eps[csindx]*np.sin(glon_eps[csindx])
zpos_cs=np.sin(glat_eps[csindx])/plx_eps[csindx]
plx_cs=plx_eps[csindx]
vt_cs=vt_eps[csindx]
bt_cs=bt_eps[csindx]
velz_cs=velz_eps[csindx]
glon_cs=glon_eps[csindx]
glat_cs=glon_eps[csindx]
vlon_cs=pmvconst*pmlon_eps[csindx]/plx_eps[csindx]
vlat_cs=pmvconst*pmlat_eps[csindx]/plx_eps[csindx]
pmraerr_cs=pmvconst \
            *np.sqrt((pmraerr_eps[csindx]**2+(pmra_eps[csindx]**2) \
            *(plx_eps[csindx]**(-2))*(plxerr_eps[csindx]**2))\
            *(plx_eps[csindx]**(-2)))
pmdecerr_cs=pmvconst \
           *np.sqrt((pmdecerr_eps[csindx]**2+(pmdec_eps[csindx]**2) \
            *(plx_eps[csindx]**(-2))*(plxerr_eps[csindx]**2))\
            *(plx_eps[csindx]**(-2)))
hrverr_cs=hrverr_eps[csindx]
hrv_cs=hrv_eps[csindx]
logg_cs=logg_eps[csindx]

nstar_cs=len(xpos_cs)

print ' number of stars after cylinder selection =',nstar_cs

# output ascii data for test
f=open('cylinder_starvz.asc','w')
i=0
while i < nstar_cs:
  print >>f, "%f %f %f %f %f %f %f %f %f %f %f %f %f %f" \
    %(xpos_cs[i],ypos_cs[i],zpos_cs[i], \
    plx_cs[i],bt_cs[i],vt_cs[i],velz_cs[i], \
      pmraerr_cs[i],pmdecerr_cs[i],hrverr_cs[i],hrv_cs[i],vlon_cs[i] \
      ,vlat_cs[i],logg_cs[i])
  i+=1
f.close()

# output fits data
tbhdu=pyfits.BinTableHDU.from_columns([\
  pyfits.Column(name='xpos',format='E',array=xpos_cs),\
  pyfits.Column(name='ypos',format='E',array=ypos_cs),\
  pyfits.Column(name='zpos',format='E',array=zpos_cs),\
  pyfits.Column(name='plx',format='E',array=plx_cs),\
  pyfits.Column(name='Bt',format='E',array=bt_cs),\
  pyfits.Column(name='Vt',format='E',array=vt_cs),\
  pyfits.Column(name='velz',format='E',array=velz_cs)])
tbhdu.writeto('selstars_xyzvz.fits',clobber=True)

# histogram along z
nsz_n=np.histogram(zpos_cs,nbin_n,(zmin_n,zmax_n))[0]
nsz_s=np.histogram(zpos_cs,nbin_s,(zmin_s,zmax_s))[0]


# vzm and vz2m
vzmhist_n=np.histogram(zpos_cs,nbin_n,(zmin_n,zmax_n),weights=velz_cs)[0]/nsz_n
vzmhist_s=np.histogram(zpos_cs,nbin_s,(zmin_s,zmax_s),weights=velz_cs)[0]/nsz_s
vz2mhist_n=np.histogram(zpos_cs,nbin_n,(zmin_n,zmax_n),weights=(velz_cs**2))[0]/nsz_n
vz2mhist_s=np.histogram(zpos_cs,nbin_s,(zmin_s,zmax_s),weights=(velz_cs**2))[0]/nsz_s
# sig_vz
sigvz_n=np.sqrt(vz2mhist_n-vzmhist_n**2)
sigvz_s=np.sqrt(vz2mhist_s-vzmhist_s**2)

print ' z bins North=',zbin_n
print ' vz dispersion North=',sigvz_n
print ' vz mean North=',vzmhist_n
print ' number of stars in bin in North=',nsz_n
print ' z bins South=',zbin_s
print ' vz dispersion South=',sigvz_s
print ' vz mean South=',vzmhist_s
print ' number of stars in bin in South=',nsz_s

# show histogram
plt.scatter(zpos_cs,velz_cs,marker='.')
plt.scatter(zbin_n,sigvz_n,marker='o',color='red',s=50)
plt.scatter(zbin_s,sigvz_s,marker='o',color='red',s=50)
plt.xlabel(r"z (kpc)",fontsize=18,fontname="serif")
plt.ylabel(r"Vz (km/s)",fontsize=18,fontname="serif")
#plt.axis([-0.25,0.25,0.0,100])
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()
