import os, os.path
import sys
import csv
from optparse import OptionParser
sys.path.append("/home/bovy/Repos/tgas-completeness/py/")
import tqdm
import numpy
from extreme_deconvolution import extreme_deconvolution
import gaia_tools.load
from galpy.util import bovy_coords
import effsel # from tgas-completeness
from effsel import main_sequence_cut_r
numpy.random.seed(3)
def get_options():
    usage = "usage: %prog [options]"
    parser = OptionParser(usage=usage)
    # How to separate types, heights
    parser.add_option("--dpop",dest='dpop',default=10,type='int',
                      help="Width of stellar bins")
    parser.add_option("--dz",dest='dz',default=0.025,type='float',
                      help="Bins in height to consider (between -412.5 pc and 412.5 pc)")
    # Number of Gaussians
    parser.add_option("-k",dest='ngauss',default=3,type='int',
                      help="Number of Gaussians to use in each fit")
    # Number of bootstrap samples
    parser.add_option("-b",dest='nboot',default=20,type='int',
                      help="Number of bootstrap samples to use")
    # Re-start options
    parser.add_option("-r",action="store_true", 
                      dest="restart",default=False,
                      help="Restart, appends to existing output file")
    parser.add_option("--start",dest='start',default=0,type='int',
                      help="Index of the stellar type (for the given dpop) to start")
    parser.add_option("--startz",dest='startz',default=0,type='int',
                      help="Index of the vertical bin to start")
    # Output file
    parser.add_option("-o",dest='outfilename',
                      default=None,
                      help="Name of the file that will hold the output")
    return parser

def load_data():
    # Load TGAS and 2MASS
    tgas= gaia_tools.load.tgas()
    twomass= gaia_tools.load.twomass()
    jk= twomass['j_mag']-twomass['k_mag']
    dm= -5.*numpy.log10(tgas['parallax'])+10.
    mj= twomass['j_mag']-dm
    return (tgas,twomass,jk,dm,mj)

def load_spectraltypes():
    # Load spectral types, taken from tgas-completeness/TGAS-stellar-densities
    sp= effsel.load_spectral_types()
    sp_indx= numpy.array([(not 'O' in s)*(not 'L' in s)*(not 'T' in s)\
                          *(not 'Y' in s)\
                          *(not '.5V' in s) for s in sp['SpT']],
                         dtype='bool')
    # Cut out the small part where the color decreases
    sp_indx*= (numpy.roll((sp['JH']+sp['HK']),1)-(sp['JH']+sp['HK'])) <= 0.
    # Also cut out B 
    sp_indx*= numpy.array([(not 'B' in s) for s in sp['SpT']],dtype='bool') 
    # JK boundaries
    sub_indx= numpy.zeros_like(sp_indx)
    sub_indx[sp_indx]= True
    sub_indx[numpy.arange(len(sp_indx)) == (numpy.amin(numpy.arange(len(sp_indx))[sp_indx])-1)]= True
    sub_indx[numpy.arange(len(sp_indx)) == (numpy.amax(numpy.arange(len(sp_indx))[sp_indx])+1)]= True
    sp_jk= (sp['JH']+sp['HK'])[sub_indx]
    sp_jkmin= sp_jk+0.5*(numpy.roll(sp_jk,1)-sp_jk)
    sp_jkmax= sp_jk+0.5*(numpy.roll(sp_jk,-1)-sp_jk)
    sp_jk_bins= list(sp_jkmin[1:])
    sp_jk_mid= 0.5*(sp_jkmin+sp_jkmax)[1:-1]
    # MJ
    sp_mj= sp['M_J'][sub_indx]
    sp_mjmin= sp_mj+0.5*(numpy.roll(sp_mj,1)-sp_mj)
    sp_mjmax= sp_mj+0.5*(numpy.roll(sp_mj,-1)-sp_mj)
    sp_mj= sp_mj[1:-1]
    sp_mjmax= sp_mjmax[1:-1]
    sp_mjmin= sp_mjmin[1:-1]
    return (sp, sp_jk_bins)

def combined_sig2(amp,mean,covar):
    indx= numpy.sqrt(covar) < 30.
    tamp= amp[indx]/numpy.sum(amp[indx])
    return (numpy.sum(tamp*(covar+mean**2.)[indx])-numpy.sum(tamp*mean[indx])**2.)
def combined_k(amp,mean,covar):
    indx= numpy.sqrt(covar) < 30.
    tamp= amp[indx]/numpy.sum(amp[indx])
    tmean= numpy.sum(tamp*mean[indx])
    return (numpy.sum(tamp*(mean**4.+6.*mean**2.*covar+3.*covar**2.)[indx])
            -4.*tmean*numpy.sum(tamp*(mean**3.+3.*mean*covar)[indx])
            +6.*tmean**2.*numpy.sum(tamp*(mean**2.+covar)[indx])
            -3.*tmean**4.
           -3.*combined_sig2(amp,mean,covar)**2.)

def bootstrap(nboot,vrd,vrd_cov,proj,ngauss=2):
    out= numpy.empty((2,nboot))
    for ii in range(nboot):
        # Draw w/ replacement
        indx= numpy.floor(numpy.random.uniform(size=len(vrd))*len(vrd)).astype('int')
        ydata= vrd[indx]
        ycovar= vrd_cov[indx]
        initamp= numpy.random.uniform(size=ngauss)
        initamp/= numpy.sum(initamp)
        m= numpy.zeros(3)
        s= numpy.array([40.,40.,20.])
        initmean= []
        initcovar= []
        for jj in range(ngauss):
            initmean.append(m+numpy.random.normal(size=3)*s)
            initcovar.append(4.*s**2.*numpy.diag(numpy.ones(3)))
        initcovar= numpy.array(initcovar)
        initmean= numpy.array(initmean)
        lnL= extreme_deconvolution(ydata,ycovar,initamp,initmean,initcovar,projection=proj[indx])
        out[0,ii]= combined_sig2(initamp,initmean[:,2],initcovar[:,2,2])
        out[1,ii]= combined_k(initamp,initmean[:,2],initcovar[:,2,2])
    return out

def compute_projection(tgas):
    #First calculate the transformation matrix T
    epoch= None
    theta,dec_ngp,ra_ngp= bovy_coords.get_epoch_angles(epoch)
    Tinv= numpy.dot(numpy.array([[numpy.cos(ra_ngp),-numpy.sin(ra_ngp),0.],
                                 [numpy.sin(ra_ngp),numpy.cos(ra_ngp),0.],
                                 [0.,0.,1.]]),
                    numpy.dot(numpy.array([[-numpy.sin(dec_ngp),0.,numpy.cos(dec_ngp)],
                                           [0.,1.,0.],
                                           [numpy.cos(dec_ngp),0.,numpy.sin(dec_ngp)]]),
                              numpy.array([[numpy.cos(theta),numpy.sin(theta),0.],
                                           [numpy.sin(theta),-numpy.cos(theta),0.],
                                           [0.,0.,1.]])))
    # Calculate all projection matrices
    ra= tgas['ra']/180.*numpy.pi
    dec= tgas['dec']/180.*numpy.pi
    A1= numpy.array([[numpy.cos(dec),numpy.zeros(len(tgas)),numpy.sin(dec)],
                     [numpy.zeros(len(tgas)),numpy.ones(len(tgas)),numpy.zeros(len(tgas))],
                     [-numpy.sin(dec),numpy.zeros(len(tgas)),numpy.cos(dec)]])
    A2= numpy.array([[numpy.cos(ra),numpy.sin(ra),numpy.zeros(len(tgas))],
                     [-numpy.sin(ra),numpy.cos(ra),numpy.zeros(len(tgas))],
                     [numpy.zeros(len(tgas)),numpy.zeros(len(tgas)),numpy.ones(len(tgas))]])
    TAinv= numpy.empty((len(tgas),3,3))
    for jj in range(len(tgas)):
        TAinv[jj]= numpy.dot(numpy.dot(A1[:,:,jj],A2[:,:,jj]),Tinv)
    return TAinv[:,1:]
    
def compute_vradec_cov_mc(tgas,nmc):
    vradec_cov= numpy.empty((len(tgas),2,2))
    for ii in range(len(tgas)):
        # Construct covariance matrixx
        tcov= numpy.zeros((3,3))
        tcov[0,0]= tgas['parallax_error'][ii]**2./2. # /2 because of symmetrization below
        tcov[1,1]= tgas['pmra_error'][ii]**2./2.
        tcov[2,2]= tgas['pmdec_error'][ii]**2./2.
        tcov[0,1]= tgas['parallax_pmra_corr'][ii]*tgas['parallax_error'][ii]*tgas['pmra_error'][ii]
        tcov[0,2]= tgas['parallax_pmdec_corr'][ii]*tgas['parallax_error'][ii]*tgas['pmdec_error'][ii]
        tcov[1,2]= tgas['pmra_pmdec_corr'][ii]*tgas['pmra_error'][ii]*tgas['pmdec_error'][ii]
        # symmetrize
        tcov= (tcov+tcov.T)
        # Cholesky decomp.
        L= numpy.linalg.cholesky(tcov)
        tsam= numpy.tile((numpy.array([tgas['parallax'][ii],
                                       tgas['pmra'][ii],
                                       tgas['pmdec'][ii]])),(nmc,1)).T
        tsam+= numpy.dot(L,numpy.random.normal(size=(3,nmc)))
        tvradec= numpy.array([bovy_coords._K/tsam[0]*tsam[1],
                              bovy_coords._K/tsam[0]*tsam[2]])
        vradec_cov[ii]= numpy.cov(tvradec) 
        return vradec_cov

def measure_kinematics_onepop(tgas,twomass,jk,dm,mj,spii,zbins,options,
                              csvwriter,csvout):
    # Compute XYZ
    lb= bovy_coords.radec_to_lb(tgas['ra'],tgas['dec'],degree=True,epoch=None)
    XYZ= bovy_coords.lbd_to_XYZ(lb[:,0],lb[:,1],1./tgas['parallax'],
                                degree=True)
    # Generate vradec and projection matrix
    vradec= numpy.array([bovy_coords._K/tgas['parallax']*tgas['pmra'],
                         bovy_coords._K/tgas['parallax']*tgas['pmdec']])
    proj= compute_projection(tgas)
    # Sample from the joint (parallax,proper motion) uncertainty distribution 
    # to get the covariance matrix of the vradec, using MC sims
    nmc= 10001
    vradec_cov= compute_vradec_cov_mc(tgas,nmc)
    # Fit each zbin
    if spii == options.start:
        startz= options.startz
    else:
        startz= 0
    for ii in tqdm.trange(startz,len(zbins)-1):
        indx= (XYZ[:,2] > zbins[ii])\
              *(XYZ[:,2] <= zbins[ii+1])\
              *(numpy.sqrt(XYZ[:,0]**2.+XYZ[:,1]**2.) < 0.2)
        nstar= numpy.sum(indx)
        if numpy.sum(indx) < 30: continue
        # Basic XD fit
        ydata= vradec.T[indx]
        ycovar= numpy.zeros_like(vradec.T)[indx]
        initamp= numpy.random.uniform(size=options.ngauss)
        initamp/= numpy.sum(initamp)
        m= numpy.zeros(3)
        s= numpy.array([40.,40.,20.])
        initmean= []
        initcovar= []
        for jj in range(options.ngauss):
            initmean.append(m+numpy.random.normal(size=3)*s)
            initcovar.append(4.*s**2.*numpy.diag(numpy.ones(3)))
        initcovar= numpy.array(initcovar)
        initmean= numpy.array(initmean)
        lnL= extreme_deconvolution(ydata,ycovar,initamp,initmean,initcovar,
                                   projection=proj[indx])
        sig2z= combined_sig2(initamp,initmean[:,2],initcovar[:,2,2])
        kurtz= combined_k(initamp,initmean[:,2],initcovar[:,2,2])
        sam= bootstrap(options.nboot,
                       vradec.T[indx],vradec_cov[indx],proj[indx],
                       ngauss=options.ngauss)
        sig2z_err= 1.4826*numpy.median(numpy.fabs(sam[0]-numpy.median(sam[0])))
        kurtz_err= 1.4826*numpy.median(numpy.fabs(sam[1]-numpy.median(sam[1])))
        sig2kurtz_corr= numpy.corrcoef(sam)[0,1]
        csvwriter.writerow([spii,ii,nstar,
                            sig2z,sig2z_err,kurtz,kurtz_err,sig2kurtz_corr])
        csvout.flush()
    return None

def read_kinematics(filename,dpop=None,dz=0.025):
    # Reads the kinematics file as written by this module
    if dpop is None:
        try:
            dpop= int(filename.split('dpop')[1].split('.csv')[0])
        except:
            raise ValueError('dpop= not set and could not be gleaned from the filename')
    zbins= define_zbins(dz)
    npop= 45//dpop+1
    nz= len(zbins)
    print(dpop,npop,nz)
    out_sig2z= numpy.zeros((npop,nz))
    out_sig2z_err= numpy.zeros((npop,nz))
    out_sig2z[:,:]= numpy.nan
    out_sig2z_err[:,:]= numpy.nan
    with open(filename,'r') as csvfile:
        csvreader= csv.reader(csvfile,delimiter=',')
        for row in csvreader:
            out_sig2z[int(row[0])//dpop,int(row[1])]= float(row[3])
            out_sig2z_err[int(row[0])//dpop,int(row[1])]= float(row[4])
    return (out_sig2z,out_sig2z_err)

def define_zbins(dz):
    return numpy.arange(-0.4125,0.425,dz)

if __name__ == '__main__':
    parser= get_options()
    options,args= parser.parse_args()
    # Setup output
    if not options.restart and os.path.exists(options.outfilename):
        print("Output filename %s already exists, exiting ...")
        sys.exit(-1)
    csvout= open(options.outfilename,'a')
    csvwriter= csv.writer(csvout,delimiter=',')
    zbins= define_zbins(options.dz)
    tgas, twomass, jk, dm, mj= load_data()
    sp, sp_jk_bins= load_spectraltypes()
    for spii in range(options.start,45,options.dpop):
        # Select data for this bin
        jkmin= sp_jk_bins[spii]
        if spii+options.dpop > 45:
            jkmax= sp_jk_bins[45]
        else:
            jkmax= sp_jk_bins[spii+options.dpop]
        good_plx_indx= (tgas['parallax']/tgas['parallax_error'] > 10.)\
                       *(jk != 0.)
        good_sampling= good_plx_indx*(jk > jkmin)*(jk < jkmax)\
                       *(mj < main_sequence_cut_r(jk,tight=False,low=True))\
                       *(mj > main_sequence_cut_r(jk,tight=False))
        print("Found %i stars in TGAS with good parallaxes for stars in bin %i " \
              % (numpy.sum(good_sampling),spii))
        measure_kinematics_onepop(tgas[good_sampling],twomass[good_sampling],
                                  jk[good_sampling],dm[good_sampling],
                                  mj[good_sampling],spii,zbins,options,
                                  csvwriter,csvout)
        
