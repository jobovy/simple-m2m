
#
# read counts_vert_AthroughF_10bins_wext.sav
#  and convert it to HOM2M input
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
# from hom2m_sc1d import *
from astropy.modeling import models, fitting
import pickle

# set fake h_obs
h_obs=0.075

# select F stars
ii=1

with open('counts_vert_AthroughF_10bins_wext.sav','rb') as savefile:
    all_counts= pickle.load(savefile)
    all_counts_unc= pickle.load(savefile)
    all_effvol= pickle.load(savefile)
# convert to pc^-3
all_counts= all_counts[ii,:,0]*1.0e-9
all_counts_unc= all_counts_unc[ii,:,0]*1.0e-9
# convert to pc^3
all_effvol= all_effvol[ii,:,0]*1.0e9

print ' N data point=',len(all_counts)

zbins= np.arange(-0.4125,0.425,0.025)
# only selected range to match with v2m data
z_obs=np.arange(-0.4,0.4,0.025)

# boost density for hom2m, normalise arbitrary for HOM2M
dens_obs=all_counts[0:32]*1.5e3
dens_obs_noise=all_counts_unc[0:32]*1.5e3

print ' after selection N z_obs, dens=',len(z_obs),len(dens_obs)
print ' z_obs=',z_obs

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
                    xrange=[np.amin(z_obs)-0.1,np.amax(z_obs)+0.1],yrange=[1.0e-1,10.0],
  gcf=True)
plt.errorbar(z_obs,dens_obs,yerr=dens_obs_noise,marker='None',ls='none',color='k')
  
plt.show()

