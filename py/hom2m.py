# hom2m.py: Simple Harmonic-Oscillator M2M implementation
import numpy
import copy
from scipy import interpolate, integrate
import tqdm

######################### COORDINATE TRANSFORMATIONS ##########################
def zvz_to_Aphi(z,vz,omega):
    A= numpy.sqrt(z**2.+vz**2./omega**2.)
    phi= numpy.arctan2(-vz/omega,z)
    return (A,phi)
def Aphi_to_zvz(A,phi,omega):
    z= A*numpy.cos(phi)
    vz= -A*omega*numpy.sin(phi)
    return (z,vz)
############################# ISOTHERMAL DF TOOLS #############################
def sample_iso(sigma,omega,n=1):
    E= numpy.random.exponential(scale=sigma**2.,size=n)
    phi= numpy.random.uniform(size=n)*2.*numpy.pi
    A= numpy.sqrt(2.*E)/omega
    return Aphi_to_zvz(A,phi,omega)
################################### KERNELS ###################################
def sph_kernel(r,h):
    out= numpy.zeros_like(r)
    out[(r >= 0.)*(r <= h/2.)]= 1.-6.*(r[(r >= 0.)*(r <= h/2.)]/h)**2.+6.*(r[(r >= 0.)*(r <= h/2.)]/h)**3.
    out[(r > h/2.)*(r <= h)]= 2.*(1-(r[(r > h/2.)*(r <= h)]/h))**3.
    out*= 4./3./h
    return out
def sph_kernel_deriv(r,h):
    out= numpy.zeros_like(r)
    out[(r >= 0.)*(r <= h/2.)]= -12.*r[(r >= 0.)*(r <= h/2.)]/h**2.+18.*r[(r >= 0.)*(r <= h/2.)]**2./h**3.
    out[(r > h/2.)*(r <= h)]= -6./h*(1-(r[(r > h/2.)*(r <= h)]/h))**2.
    out*= 4./3./h
    return out
def epanechnikov_kernel(r,h):
    out= numpy.zeros_like(r)
    out[(r >= 0.)*(r <= h)]= 3./4.*(1.-r[(r >= 0.)*(r <= h)]**2./h**2.)/h
    return out
def epanechnikov_kernel_deriv(r,h):
    out= numpy.zeros_like(r)
    out[(r >= 0.)*(r <= h)]= -3./2.*r[(r >= 0.)*(r <= h)]/h**3.
    return out

################################### OBSERVATIONS ##############################
### compute density at z_obs
def compute_dens(z,zsun,z_obs,h_obs,w=None,kernel=epanechnikov_kernel):
    if w is None: w= numpy.ones_like(z)/float(len(z))
    dens= numpy.zeros_like(z_obs)
    for jj,zo in enumerate(z_obs):
        dens[jj]= numpy.nansum(w*kernel(numpy.fabs(zo-z+zsun),h_obs))
    return dens
### compute_v2
def compute_v2(z,vz,zsun,z_obs,h_obs,w=None,kernel=epanechnikov_kernel):
    if w is None: w= numpy.ones_like(z)/float(len(z))
    v2= numpy.zeros_like(z_obs)
    for jj,zo in enumerate(z_obs):
        v2[jj]= numpy.nansum(w*kernel(numpy.fabs(zo-z+zsun),h_obs)*vz**2.)\
            /numpy.nansum(w*kernel(numpy.fabs(zo-z+zsun),h_obs))
    return v2
### compute_densv2
def compute_densv2(z,vz,zsun,z_obs,h_obs,w=None,kernel=epanechnikov_kernel):
    if w is None: w= numpy.ones_like(z)/float(len(z))
    densv2= numpy.zeros_like(z_obs)
    for jj,zo in enumerate(z_obs):
        densv2[jj]= numpy.nansum(w*kernel(numpy.fabs(zo-z+zsun),h_obs)*vz**2.)
    return densv2

################################### M2M PRIOR #################################
def prior(w_m2m,mu,w_prior,prior):
    """Evaluate the log prior for the different cases"""
    if prior.lower() == 'entropy':
        return prior_entropy(w_m2m,mu,w_prior)
    else:
        return prior_gamma(w_m2m,mu,w_prior)
def sample_prior(mu,w_prior,prior):
    if prior.lower() == 'entropy':
        if numpy.all(numpy.fabs(w_prior/w_prior[0]-1.) < 10.**-8.):
            return sample_entropy(mu,w_prior[0],n=len(w_prior))
        else:
            raise NotImplementedError("Sampling the entropy prior with different prior weights for different orbits not implemented yet")
    else:
        if numpy.all(numpy.fabs(w_prior/w_prior[0]-1.) < 10.**-8.):
            return sample_gamma(mu,w_prior[0],n=len(w_prior))
        else:
            raise NotImplementedError("Sampling the gamma prior with different prior weights for different orbits not implemented yet")

def prior_entropy(w_m2m,mu,w_prior):
    """Returns log prior for each weight individually"""
    return -mu*w_m2m*(numpy.log(w_m2m/w_prior)-1.)
def sample_entropy(mu,w_prior,n=1.):
    # Compute CDF
    ws= numpy.linspace(0.,10.*w_prior,1001)
    cdf= numpy.array([\
        integrate.quad(lambda x: (x/w_prior/numpy.exp(1.))**(-mu*x),0.,w)[0]
        for w in ws])
    cdf/= cdf[-1]
    cdf[cdf > 1.]= 1.
    ma_indx= (numpy.arange(len(ws))[cdf == 1.])[0]
    ip= interpolate.InterpolatedUnivariateSpline(cdf[:ma_indx],ws[:ma_indx],
                                                 k=3)
    out= numpy.random.uniform(size=n)
    return ip(out)

def prior_gamma(w_m2m,mu,w_prior):
    """Returns log prior for each weight individually"""
    return mu*(w_prior*numpy.log(w_m2m)-w_m2m)
def sample_gamma(mu,w_prior,n=1.):
    return numpy.random.gamma(mu*w_prior+1.,1./mu,size=n)

############################### M2M FORCE-OF-CHANGE ###########################
# All defined here as the straight d constraint / d parameter (i.e., does *not*
# include things like eps, weight)

# Due to the prior
def force_of_change_prior_weights(w_m2m,mu,w_prior,prior):
    if prior.lower() == 'entropy':
        return force_of_change_entropy_weights(w_m2m,mu,w_prior)
    else:
        return force_of_change_gamma_weights(w_m2m,mu,w_prior)

def force_of_change_entropy_weights(w_m2m,mu,w_prior):
    return -mu*numpy.log(w_m2m/w_prior)

def force_of_change_gamma_weights(w_m2m,mu,w_prior):
    return mu*(w_prior/w_m2m-1.)

#Due to the density
#For the weights
def force_of_change_density_weights(w_m2m,zsun_m2m,z_m2m,vz_m2m,
                                    z_obs,dens_obs,dens_obs_noise,
                                    h_m2m=0.02,kernel=epanechnikov_kernel,
                                    delta_m2m=None,Wij=None):
    """Computes the force of change for all of the weights"""
    delta_m2m_new= numpy.zeros_like(z_obs)
    if Wij is None:
        Wij= numpy.zeros((len(z_obs),len(z_m2m)))
        calc_Wij= True
    else:
        calc_Wij= False
    for jj,zo in enumerate(z_obs):
        if calc_Wij:
            Wij[jj]= kernel(numpy.fabs(zo-z_m2m+zsun_m2m),h_m2m)
        delta_m2m_new[jj]= (numpy.nansum(w_m2m*Wij[jj])-dens_obs[jj])/dens_obs_noise[jj]
    if delta_m2m is None: delta_m2m= delta_m2m_new
    return (-numpy.nansum(numpy.tile(delta_m2m/dens_obs_noise,(len(z_m2m),1)).T
                          *Wij,axis=0),
             delta_m2m_new)

# Due to the velocity
# For the weights
def force_of_change_densv2_weights(w_m2m,zsun_m2m,z_m2m,vz_m2m,
                                   z_obs,densv2_obs,densv2_obs_noise,
                                   h_m2m,kernel=epanechnikov_kernel,
                                   deltav2_m2m=None,Wij=None):
    """Computes the force of change for all of the weights due to the velocity constraint"""
    deltav2_m2m_new= numpy.zeros_like(z_obs)
    if Wij is None:
        Wij= numpy.zeros((len(z_obs),len(z_m2m)))
        calc_Wij= True
    else:
        calc_Wij= False
    for jj,zo in enumerate(z_obs):
        if calc_Wij:
            Wij[jj]= kernel(numpy.fabs(zo-z_m2m+zsun_m2m),h_m2m)*vz_m2m**2.
        deltav2_m2m_new[jj]= (numpy.nansum(w_m2m*Wij[jj])
                              -densv2_obs[jj])/densv2_obs_noise[jj]
    if deltav2_m2m is None: deltav2_m2m= deltav2_m2m_new
    return (-numpy.nansum(numpy.tile(deltav2_m2m/densv2_obs_noise,
                                  (len(z_m2m),1)).T*Wij,axis=0),
             deltav2_m2m_new)

# Due to v^2
def force_of_change_v2_weights(w_m2m,zsun_m2m,z_m2m,vz_m2m,
                               z_obs,v2_obs,v2_obs_noise,
                               h_m2m,kernel=epanechnikov_kernel,
                               deltav2_m2m=None,Wij=None):
    """Computes the force of change for all of the weights due to the velocity constraint"""
    deltav2_m2m_new= numpy.zeros_like(z_obs)
    if Wij is None:
        Wij= numpy.zeros((len(z_obs),len(z_m2m)))
        calc_Wij= True
    else:
        calc_Wij= False
    dens_m2m= numpy.zeros_like(z_obs)
    wv2_m2m= numpy.zeros_like(z_obs)
    for jj,zo in enumerate(z_obs):
        if calc_Wij:
            Wij[jj]= kernel(numpy.fabs(zo-z_m2m+zsun_m2m),h_m2m)
        dens_m2m[jj]=numpy.nansum(w_m2m*Wij[jj])
        wv2_m2m[jj]=numpy.nansum(w_m2m*Wij[jj]*(vz_m2m**2))
        deltav2_m2m_new[jj]= (wv2_m2m[jj]/dens_m2m[jj]-v2_obs[jj]) \
            /v2_obs_noise[jj]
    if deltav2_m2m is None: deltav2_m2m= deltav2_m2m_new
    return (-(numpy.nansum(numpy.tile(deltav2_m2m/(dens_m2m*v2_obs_noise), 
                                      (len(z_m2m),1)).T*(Wij),axis=0) 
              *(vz_m2m**2.)-numpy.nansum(numpy.tile(
                    deltav2_m2m*wv2_m2m/((dens_m2m**2)*v2_obs_noise),
                    (len(z_m2m),1)).T*(Wij),axis=0)),deltav2_m2m_new)

# Short-cuts
def force_of_change_weights(w_m2m,zsun_m2m,z_m2m,vz_m2m,
                            z_obs,dens_obs,dens_obs_noise,
                            densv2_obs,densv2_obs_noise,
                            prior,mu,w_prior,
                            h_m2m=0.02,kernel=epanechnikov_kernel,
                            delta_m2m=None,deltav2_m2m=None,
                            Wij=None,Wvz2ij=None,use_v2=False):
    """Computes the force of change for all of the weights"""
    fcw, delta_m2m_new=\
        force_of_change_density_weights(w_m2m,zsun_m2m,
                                        z_m2m,vz_m2m,
                                        z_obs,
                                        dens_obs,dens_obs_noise,
                                        h_m2m=h_m2m,kernel=kernel,
                                        delta_m2m=delta_m2m,Wij=Wij)
    # Add velocity constraint if given
    if not densv2_obs is None and not use_v2:
        fcwv2, deltav2_m2m_new= \
            force_of_change_densv2_weights(w_m2m,zsun_m2m,
                                           z_m2m,vz_m2m,
                                           z_obs,
                                           densv2_obs,densv2_obs_noise,
                                           h_m2m=h_m2m,
                                           kernel=kernel,
                                           deltav2_m2m=deltav2_m2m,
                                           Wij=Wvz2ij)
    elif not densv2_obs is None: # use_v2
        fcwv2, deltav2_m2m_new= \
            force_of_change_v2_weights(w_m2m,zsun_m2m,
                                       z_m2m,vz_m2m,
                                       z_obs,
                                       densv2_obs,densv2_obs_noise,
                                       h_m2m=h_m2m,
                                       kernel=kernel,
                                       deltav2_m2m=deltav2_m2m,
                                       Wij=Wij)
    else:
        fcwv2= 0.
        deltav2_m2m_new= 0.
    fcw+= fcwv2
    # Add prior
    fcw+= force_of_change_prior_weights(w_m2m,mu,w_prior,prior)
    return (fcw,delta_m2m_new,deltav2_m2m_new)

# zsun
def force_of_change_zsun(w_m2m,zsun_m2m,z_m2m,vz_m2m,
                         z_obs,dens_obs_noise,delta_m2m,
                         densv2_obs_noise,deltav2_m2m,
                         kernel=epanechnikov_kernel,
                         kernel_deriv=epanechnikov_kernel_deriv,
                         h_m2m=0.02,use_v2=False):
    """Computes the force of change for zsun"""
    if use_v2:
        dens_m2m= numpy.zeros_like(z_obs)
        wv2_m2m= numpy.zeros_like(z_obs)
        for jj,zo in enumerate(z_obs):
            Wij= kernel(numpy.fabs(zo-z_m2m+zsun_m2m),h_m2m)
            dens_m2m[jj]=numpy.nansum(w_m2m*Wij)
            wv2_m2m[jj]=numpy.nansum(w_m2m*Wij*(vz_m2m**2))

    out= 0.
    for jj,zo in enumerate(z_obs):
        dWij= kernel_deriv(numpy.fabs(zo-z_m2m+zsun_m2m),h_m2m)\
            *numpy.sign(zo-z_m2m+zsun_m2m)
        wdWj= numpy.nansum(w_m2m*dWij)
        if not numpy.isnan(delta_m2m[jj]/dens_obs_noise[jj]):
            out+= delta_m2m[jj]/dens_obs_noise[jj]*wdWj
        if not use_v2:
            if not numpy.isnan(deltav2_m2m[jj]/densv2_obs_noise[jj]):
                out+= deltav2_m2m[jj]/densv2_obs_noise[jj]\
                    *numpy.nansum(w_m2m*vz_m2m**2.*dWij)
        else:
            wv2mdWj= numpy.nansum(w_m2m*(vz_m2m**2)*dWij)
            if not numpy.isnan((deltav2_m2m[jj]/densv2_obs_noise[jj])):
                out+= (deltav2_m2m[jj]/densv2_obs_noise[jj]) \
                    *((wv2mdWj/dens_m2m[jj])-(wv2_m2m[jj]/(dens_m2m[jj]**2))*wdWj)
    return -out

# omega

# Function that returns the difference in z and vz for orbits starting at the 
# same (z,vz)_init integrated in potentials with different omega
def zvzdiff(z_init,vz_init,omega1,omega2,t):
    A1, phi1= zvz_to_Aphi(z_init,vz_init,omega1)
    A2, phi2= zvz_to_Aphi(z_init,vz_init,omega2)
    return (A2*numpy.cos(omega2*t+phi2)-A1*numpy.cos(omega1*t+phi1),
           -A2*omega2*numpy.sin(omega2*t+phi2)\
                +A1*omega1*numpy.sin(omega1*t+phi1)) 

def force_of_change_omega(w_m2m,zsun_m2m,omega_m2m,
                          z_m2m,vz_m2m,z_prev,vz_prev,
                          step,z_obs,dens_obs,dens_obs_noise,delta_m2m,
                          densv2_obs,densv2_obs_noise,deltav2_m2m,
                          h_m2m=0.02,kernel=epanechnikov_kernel,
                          delta_omega=0.3,use_v2=False):
    """Compute the force of change by direct finite difference 
    of the objective function"""
    dz, dvz= zvzdiff(z_prev,vz_prev,omega_m2m,omega_m2m+delta_omega,step)
    fcw, delta_m2m_do, deltav2_m2m_do= force_of_change_weights(\
        w_m2m,zsun_m2m,
        z_m2m+dz,vz_m2m+dvz,
        z_obs,dens_obs,dens_obs_noise,
        densv2_obs,densv2_obs_noise,
        'entropy',0.,1., # weights prior doesn't matter, so set to zero
        h_m2m=h_m2m,kernel=kernel,use_v2=use_v2)
#    return -numpy.nansum(\
#        delta_m2m*(delta_m2m_do-delta_m2m)/dens_obs_noise
#        +deltav2_m2m*(deltav2_m2m_do-deltav2_m2m)/densv2_obs_noise)\
#        /delta_omega

    return -numpy.nansum(\
        delta_m2m*(delta_m2m_do-delta_m2m)
        +deltav2_m2m*(deltav2_m2m_do-deltav2_m2m))\
        /delta_omega

################################ M2M OPTIMIZATION #############################
def precalc_kernel(z_init,vz_init,
                   omega_m2m,zsun_m2m,
                   z_obs,use_v2=False,step=0.001,nstep=1000,
                   kernel=epanechnikov_kernel,h_m2m=0.02):
    """
    NAME:
       precalc_kernel
    PURPOSE:
       Orbit average the kernels
    INPUT:
       z_init - initial z [N]
       vz_init - initial vz (rad) [N]
       omega_m2m - potential parameter omega
       zsun_m2m - Sun's height above the plane [N]
       use_v2= (False) if True, densv2_obs and densv2_obs_noise are actually <v^2> directly, not dens x <v^2>
       z_obs - heights at which the density observations are made
       step= stepsize of orbit integration
       nstep= number of steps to integrate the orbits for
       kernel= a smoothing kernel
       h_m2m= kernel size parameter for computing the observables
    OUTPUT:
       (Kdens,Kdensv2) - (density kernels [n_zobs,n_weights],
                          densityx v^2 kernels [n_zobs,n_weights])
    HISTORY:
       2017-03-14 - Written - Bovy (UofT/CCA)
       2017-03-25 - Added use_v2 for <v^2> case - Kawata (MSSL,UCL)
    """
    Kij= numpy.zeros((len(z_obs),len(z_init)))
    Kvz2ij= numpy.zeros((len(z_obs),len(z_init)))
    A_init, phi_init= zvz_to_Aphi(z_init,vz_init,omega_m2m)
    for ii in range(nstep):
        # Then update force
        phi_now= omega_m2m*ii*step+phi_init
        z_m2m= A_init*numpy.cos(phi_now)
        vz_m2m= -A_init*omega_m2m*numpy.sin(phi_now)
        # Compute kernel
        tW= numpy.empty((len(z_obs),len(z_init)))
        for jj,zo in enumerate(z_obs):
            tW[jj]= kernel(numpy.fabs(zo-z_m2m+zsun_m2m),h_m2m)
        Kij+= tW
        if use_v2:
            Kvz2ij+= ((tW*vz_m2m**2.).T/numpy.nansum(tW,axis=1)).T
        else:
            Kvz2ij+= tW*vz_m2m**2.
    return (Kij/nstep,Kvz2ij/nstep)

def fit_m2m(w_init,z_init,vz_init,
            omega_m2m,zsun_m2m,
            z_obs,dens_obs,dens_obs_noise,
            densv2_obs=None,densv2_obs_noise=None,use_v2=False,
            step=0.001,nstep=1000,
            eps=0.1,mu=1.,prior='entropy',w_prior=None,
            kernel=epanechnikov_kernel,
            kernel_deriv=epanechnikov_kernel_deriv,
            h_m2m=0.02,
            smooth=None,st96smooth=False,schwarzschild=False,
            output_wevolution=False,
            fit_zsun=False,fit_omega=False,
            skipomega=10,delta_omega=0.3):
    """
    NAME:
       fit_m2m
    PURPOSE:
       Run M2M optimization on the harmonic-oscillator data
    INPUT:
       w_init - initial weights [N]
       z_init - initial z [N]
       vz_init - initial vz (rad) [N]
       omega_m2m - potential parameter omega
       zsun_m2m - Sun's height above the plane [N]
       z_obs - heights at which the density observations are made
       dens_obs - observed densities
       dens_obs_noise - noise in the observed densities
       densv2_obs= (None) observed density x velocity-squareds (optional)
       densv2_obs_noise= (None) noise in the observed densities x velocity-squareds
       use_v2= (False) if True, densv2_obs and densv2_obs_noise are actually <v^2> directly, not dens x <v^2>
       step= stepsize of orbit integration
       nstep= number of steps to integrate the orbits for
       eps= M2M epsilon parameter (can be array when fitting zsun, omega; in that case eps[0] = eps_weights, eps[1] = eps_zsun, eps[1 or 2 based on fit_zsun] = eps_omega)
       mu= M2M entropy parameter mu
       prior= ('entropy' or 'gamma')
       w_prior= (None) prior weights (if None, equal to w_init)
       fit_zsun= (False) if True, also optimize zsun
       fit_omega= (False) if True, also optimize omega
       skipomega= only update omega every skipomega steps
       delta_omega= (0.3) difference in omega to use to compute derivative of objective function wrt omega
       kernel= a smoothing kernel
       kernel_deriv= the derivative of the smoothing kernel
       h_m2m= kernel size parameter for computing the observables
       smooth= smoothing parameter alpha (None for no smoothing)
       st96smooth= (False) if True, smooth the constraints (Syer & Tremaine 1996), if False, smooth the objective function and its derivative (Dehnen 2000)
       schwarzschild= (False) if True, first compute orbit-averaged kernels and use those rather than the point-estimates [only for basic weights fit]
       output_wevolution= if set to an integer, return the time evolution of this many randomly selected weights
    OUTPUT:
       (w_out,[zsun_out, [omega_out]],Q_out,[wevol,rndindx]) - 
              (output weights [N],
              [Solar offset [nstep] optional],
              [omega [nstep] optional when fit_omega],
              [z_m2m [N] final z, optional when fit_omega],
              [vz_m2m [N] final vz, optional when fit_omega],
              objective function as a function of time [nstep],
              [weight evolution for randomly selected weights,index of random weights])
    HISTORY:
       2016-12-06 - Written - Bovy (UofT/CCA)
       2016-12-08 - Added output_wevolution - Bovy (UofT/CCA)
       2016-12-16 - Added explicit noise in the observed densities - Bovy (UofT/CCA)
       2017-02-22 - Refactored to more general function - Bovy (UofT/CCA)
       2017-02-27 - Add st96smooth option and make Dehnen smoothing the default - Bovy (UofT/CCA)
       2017-03-17 - Added zsun fit - Bovy (CCA/UofT)
       2017-03-21 - Added omega fit - Bovy (CCA/UofT)
    """
    w_out= copy.deepcopy(w_init)
    zsun_out= numpy.empty(nstep)
    omega_out= numpy.empty(nstep)
    if w_prior is None:
        w_prior= w_init
    # Parse eps
    if isinstance(eps,float):
        eps= [eps]
        if fit_zsun: eps.append(eps[0])
        if fit_omega: eps.append(eps[0])
    Q_out= []
    A_init, phi_init= zvz_to_Aphi(z_init,vz_init,omega_m2m)
    A_now= copy.copy(A_init)
    phi_now= copy.copy(phi_init)
    if output_wevolution:
        rndindx= numpy.random.permutation(len(w_out))[:output_wevolution]
        wevol= numpy.zeros((output_wevolution,nstep))
    # Use orbit-averaged weights?
    if schwarzschild:
        if fit_zsun: raise NotImplementedError('Schwarzschild-like fitting is only implemented for the basic weights fit, not with fit_zsun=True')
        if fit_omega: raise NotImplementedError('Schwarzschild-like fitting is only implemented for the basic weights fit, not with fit_omega=True')
        Kij, Kvz2ij= precalc_kernel(z_init,vz_init,
                                    omega_m2m,zsun_m2m,
                                    z_obs,step=step,nstep=nstep,
                                    kernel=kernel,h_m2m=h_m2m)
    else:
        Kij= None
        Kvz2ij= None
    # Compute force of change for first iteration
    fcw, delta_m2m_new, deltav2_m2m_new= \
        force_of_change_weights(w_init,zsun_m2m,z_init,vz_init,
                                z_obs,dens_obs,dens_obs_noise,
                                densv2_obs,densv2_obs_noise,
                                prior,mu,w_prior,
                                h_m2m=h_m2m,kernel=kernel,
                                Wij=Kij,Wvz2ij=Kvz2ij,use_v2=use_v2)
    fcw*= w_init
    fcz= 0.
    if fit_zsun:
        fcz= force_of_change_zsun(w_init,zsun_m2m,z_init,vz_init,
                                  z_obs,dens_obs_noise,delta_m2m_new,
                                  densv2_obs_noise,deltav2_m2m_new,
                                  kernel=kernel,kernel_deriv=kernel_deriv,
                                  h_m2m=h_m2m,use_v2=use_v2)
    if not smooth is None:
        delta_m2m= delta_m2m_new
        deltav2_m2m= deltav2_m2m_new
    else:
        delta_m2m= None
        deltav2_m2m= None
    if not smooth is None and not st96smooth:
        if not densv2_obs is None:
            Q= numpy.hstack((delta_m2m**2.,deltav2_m2m**2.))
        else:
            Q= delta_m2m**2.
    # setup skipomega omega counter and prev. (z,vz) for F(omega)
    ocounter= skipomega-1 # Causes F(omega) to be computed in the 1st step
    z_prev, vz_prev= Aphi_to_zvz(A_init,phi_init-skipomega*step*omega_m2m,
                                 omega_m2m) #Rewind for first step
    for ii in range(nstep):
        # Update weights first
        w_out+= eps[0]*step*fcw
        w_out[w_out < 10.**-16.]= 10.**-16.
        # then zsun
        if fit_zsun: 
            zsun_m2m+= eps[1]*step*fcz 
            zsun_out[ii]= zsun_m2m
        # then omega (skipped in the first step, so undeclared vars okay)
        if fit_omega and ocounter == skipomega:
            domega= eps[1+fit_zsun]*step*skipomega*fco
            max_domega= delta_omega/30.
            if numpy.fabs(domega) > max_domega:
                domega= max_domega*numpy.sign(domega)
            omega_m2m+= domega
            # Keep (z,vz) the same in new potential
            A_now, phi_now= zvz_to_Aphi(z_m2m,vz_m2m,omega_m2m)
            ocounter= 0
        # (Store objective function)
        if not smooth is None and st96smooth:
            if not densv2_obs is None:
                Q_out.append(numpy.hstack((delta_m2m**2.,deltav2_m2m**2.)))
            else:
                Q_out.append(delta_m2m**2.)
        elif not smooth is None:
            Q_out.append(copy.deepcopy(Q))
        else:
            if not densv2_obs is None:
                Q_out.append(numpy.hstack((delta_m2m_new**2.,
                                           deltav2_m2m_new**2.)))
            else:
                Q_out.append(delta_m2m_new**2.)
        # Then update force
        phi_now+= omega_m2m*step
        z_m2m, vz_m2m= Aphi_to_zvz(A_now,phi_now,omega_m2m)
        # Compute force of change
        if smooth is None or not st96smooth:
            # Turn these off
            tdelta_m2m= None
            tdeltav2_m2m= None
        else:
            tdelta_m2m= delta_m2m
            tdeltav2_m2m= deltav2_m2m
        fcw_new, delta_m2m_new, deltav2_m2m_new= \
            force_of_change_weights(w_out,zsun_m2m,z_m2m,vz_m2m,
                                    z_obs,dens_obs,dens_obs_noise,
                                    densv2_obs,densv2_obs_noise,
                                    prior,mu,w_prior,
                                    h_m2m=h_m2m,kernel=kernel,
                                    delta_m2m=tdelta_m2m,
                                    deltav2_m2m=tdeltav2_m2m,
                                    Wij=Kij,Wvz2ij=Kvz2ij,use_v2=use_v2)
        fcw_new*= w_out
        if fit_zsun:
            if smooth is None or not st96smooth:
                tdelta_m2m= delta_m2m_new
                tdeltav2_m2m= deltav2_m2m_new
            fcz_new= force_of_change_zsun(w_out,zsun_m2m,z_m2m,vz_m2m,
                                          z_obs,dens_obs_noise,tdelta_m2m,
                                          densv2_obs_noise,tdeltav2_m2m,
                                          kernel=kernel,
                                          kernel_deriv=kernel_deriv,
                                          h_m2m=h_m2m,use_v2=use_v2)
        if fit_omega:
            omega_out[ii]= omega_m2m
            # Update omega in this step?
            ocounter+= 1
            if ocounter == skipomega:
                if not fit_zsun and (smooth is None or not st96smooth):
                    tdelta_m2m= delta_m2m_new
                    tdeltav2_m2m= deltav2_m2m_new
                fco_new= force_of_change_omega(w_out,zsun_m2m,omega_m2m,
                                               z_m2m,vz_m2m,z_prev,vz_prev,
                                               step*skipomega,
                                               z_obs,dens_obs,dens_obs_noise,
                                               tdelta_m2m,
                                               densv2_obs,densv2_obs_noise,
                                               tdeltav2_m2m,
                                               h_m2m=h_m2m,kernel=kernel,
                                               delta_omega=delta_omega,
                                               use_v2=use_v2)
                z_prev= copy.copy(z_m2m)
                vz_prev= copy.copy(vz_m2m)
        # Increment smoothing
        if not smooth is None and st96smooth:
            delta_m2m+= step*smooth*(delta_m2m_new-delta_m2m)
            deltav2_m2m+= step*smooth*(deltav2_m2m_new-deltav2_m2m)
            fcw= fcw_new
            if fit_zsun: fcz= fcz_new
            if fit_omega and ocounter == skipomega: fco= fco_new
        elif not smooth is None:
            if not densv2_obs is None:
                Q_new= numpy.hstack((delta_m2m_new**2.,deltav2_m2m_new**2.))
            else:
                Q_new= delta_m2m_new**2.
            Q+= step*smooth*(Q_new-Q)
            fcw+= step*smooth*(fcw_new-fcw)
            if fit_zsun: fcz+= step*smooth*(fcz_new-fcz)
            if fit_omega and ocounter == skipomega:
                fco+= step*skipomega*smooth*(fco_new-fco)
        else:
            fcw= fcw_new
            if fit_zsun: fcz= fcz_new
            if fit_omega and ocounter == skipomega: fco= fco_new
        # Record random weights if requested
        if output_wevolution:
            wevol[:,ii]= w_out[rndindx]
    out= (w_out,)
    if fit_zsun: out= out+(zsun_out,)
    if fit_omega:
        out= out+(omega_out,)
        out= out+Aphi_to_zvz(A_now,phi_now,omega_m2m)
    out= out+(numpy.array(Q_out),)
    if output_wevolution:
        out= out+(wevol,rndindx,)
    return out


def sample_m2m(nsamples,
               w_init,z_init,vz_init,
               omega_m2m,zsun_m2m,
               z_obs,dens_obs,dens_obs_noise,**kwargs):
    """
    NAME:
       sample_m2m
    PURPOSE:
       Sample parameters using M2M optimization for the weights and Metropolis-Hastings for the other parameters on the harmonic-oscillator data
    INPUT:
       nsamples - number of samples from the ~PDF
       fix_weights= (False) if True, don't sample the weights

       zsun parameters:
          sig_zsun= (0.005) if sampling zsun (fit_zsun=True), proposal stepsize for steps in zsun
          nmh_zsun= (20) number of MH steps to do for zsun for each weights sample
          nstep_zsun= (500) number of steps to average the likelihood over for zsun MH

       omega parameters:
          sig_omega= (0.2) if sampling omega (fit_omega=True), proposal stepsize for steps in omega
          nmh_omega= (20) number of MH steps to do for omega for each weights sample
          nstep_omega= (500) number of steps to average the likelihood over for omega MH; also the number of steps taken to change omega adiabatically
          nstepadfac_omega= (10) use nstepadfac_omega x nstep_omega steps to adiabatically change the frequency to the proposed value

       Rest of the parameters are the same as for fit_m2m
    OUTPUT:
       (w_out,[zsun_out],Q_out,z,vz) - 
               (output weights [nsamples,N],
               [Solar offset [nsamples],
               objective function [nsamples,nobs],
               positions at the final step of each sample [nsamples,N],
               velocities at the final step of each sample [nsamples,N])
    HISTORY:
       2017-03-15 - Written - Bovy (UofT/CCA)
       2017-03-17 - Added zsun - Bovy (UofT/CCA)
    """
    nw= len(w_init)
    w_out= numpy.empty((nsamples,nw))
    z_out= numpy.empty_like(w_out)
    vz_out= numpy.empty_like(w_out)
    eps= kwargs.get('eps',0.1)
    nstep= kwargs.get('nstep',1000)
    fix_weights= kwargs.pop('fix_weights',False)
    # zsun
    fit_zsun= kwargs.get('fit_zsun',False)
    kwargs['fit_zsun']= False # Turn off for weights fits
    sig_zsun= kwargs.pop('sig_zsun',0.005)
    nmh_zsun= kwargs.pop('nmh_zsun',20)
    nstep_zsun= kwargs.pop('nstep_zsun',499*fit_zsun+1)
    if fit_zsun: 
        zsun_out= numpy.empty((nsamples))
        nacc_zsun= 0
    # omega
    fit_omega= kwargs.get('fit_omega',False)
    kwargs['fit_omega']= False # Turn off for weights fits
    sig_omega= kwargs.pop('sig_omega',0.005)
    nmh_omega= kwargs.pop('nmh_omega',20)
    nstep_omega= kwargs.pop('nstep_omega',nstep_zsun*fit_zsun\
                                +500*(1-fit_zsun))
    nstepadfac_omega= kwargs.pop('nstepadfac_omega',10)
    if fit_omega: 
        omega_out= numpy.empty((nsamples))
        nacc_omega= 0
    # Copy some kwargs that we need to re-use
    densv2_obs= copy.deepcopy(kwargs.get('densv2_obs',None))
    if not densv2_obs is None:
        Q_out= numpy.empty((nsamples,len(dens_obs)+len(densv2_obs)))
    else:
        Q_out= numpy.empty((nsamples,len(dens_obs)))
    # Setup orbits
    A_now, phi_now= zvz_to_Aphi(z_init,vz_init,omega_m2m)
    z_m2m= z_init
    vz_m2m= vz_init
    for ii in tqdm.tqdm(range(nsamples)):
        if not fix_weights:
            # Draw new observations
            tdens_obs= dens_obs\
                +numpy.random.normal(size=len(dens_obs))*dens_obs_noise
            if not densv2_obs is None:
                kwargs['densv2_obs']= densv2_obs\
                    +numpy.random.normal(size=len(densv2_obs))\
                    *kwargs.get('densv2_obs_noise')
            tout= fit_m2m(kwargs['w_prior'],z_m2m,vz_m2m,omega_m2m,zsun_m2m,
                          z_obs,tdens_obs,dens_obs_noise,
                          **kwargs)
            # Keep track of orbits
            phi_now+= omega_m2m*kwargs.get('nstep',1000)\
                *kwargs.get('step',0.001)
            z_m2m, vz_m2m= Aphi_to_zvz(A_now,phi_now,omega_m2m)
            if not densv2_obs is None: # Need to switch back to original data
                kwargs['densv2_obs']= densv2_obs
        else:
            tout= [w_init]
        # Compute average chi^2
        if fit_zsun:
            tnstep= nstep_zsun
        else:
            tnstep= nstep_omega
        kwargs['nstep']= tnstep
        kwargs['eps']= 0. # Don't change weights
        dum= fit_m2m(tout[0],z_m2m,vz_m2m,omega_m2m,zsun_m2m,
                     z_obs,dens_obs,dens_obs_noise,
                     **kwargs)
        kwargs['nstep']= nstep
        kwargs['eps']= eps
        tQ= numpy.mean(dum[1],axis=0)
        # Keep track of orbits
        phi_now+= omega_m2m*tnstep*kwargs.get('step',0.001)
        z_m2m, vz_m2m= Aphi_to_zvz(A_now,phi_now,omega_m2m)
        if fit_zsun:
            # Rewind orbit, so we use same part for all zsun/omega
            phi_now-= omega_m2m*nstep_zsun*kwargs.get('step',0.001)
            z_m2m, vz_m2m= Aphi_to_zvz(A_now,phi_now,omega_m2m)
            for jj in range(nmh_zsun):
                # Do a MH step
                zsun_new= zsun_m2m+numpy.random.normal()*sig_zsun
                kwargs['nstep']= nstep_zsun
                kwargs['eps']= 0. # Don't change weights
                dum= fit_m2m(tout[0],z_m2m,vz_m2m,omega_m2m,zsun_new,
                             z_obs,dens_obs,dens_obs_noise,
                             **kwargs)
                kwargs['nstep']= nstep
                kwargs['eps']= eps
                acc= (numpy.nansum(tQ)
                      -numpy.mean(numpy.nansum(dum[1],axis=1)))/2.
                if acc > numpy.log(numpy.random.uniform()):
                    zsun_m2m= zsun_new
                    tQ= numpy.mean(dum[1],axis=0)
                    nacc_zsun+= 1               
            zsun_out[ii]= zsun_m2m
            # update orbit
            phi_now+= omega_m2m*nstep_zsun*kwargs.get('step',0.001)
            z_m2m, vz_m2m= Aphi_to_zvz(A_now,phi_now,omega_m2m)
        if fit_zsun and nstep_zsun != nstep_omega:
            # Need to compute average obj. function for nstep_omega
            kwargs['nstep']= nstep_omega
            kwargs['eps']= 0. # Don't change weights
            dum= fit_m2m(tout[0],z_m2m,vz_m2m,omega_m2m,zsun_m2m,
                         z_obs,dens_obs,dens_obs_noise,
                         **kwargs)
            kwargs['nstep']= nstep
            kwargs['eps']= eps
            tQ= numpy.mean(dum[1],axis=0)
            # Keep track of orbits
            phi_now+= omega_m2m*nstep_omega*kwargs.get('step',0.001)
            z_m2m, vz_m2m= Aphi_to_zvz(A_now,phi_now,omega_m2m)
        if fit_omega:
            for jj in range(nmh_omega):
                # Do a MH step
                omega_new= omega_m2m+numpy.random.normal()*sig_omega
                # Slowly change the orbits from omega to omega_new, by 
                # integrating backward
                z_cur= copy.copy(z_m2m)
                vz_cur= copy.copy(vz_m2m)
                for kk in range(nstep_omega*nstepadfac_omega):
                    omega_cur= omega_m2m+(omega_new-omega_m2m)\
                        *kk/float(nstep_omega*nstepadfac_omega-1)
                    A_cur, phi_cur= zvz_to_Aphi(z_cur,vz_cur,omega_cur)
                    phi_cur-= omega_cur*kwargs.get('step',0.001)
                    z_cur, vz_cur= Aphi_to_zvz(A_cur,phi_cur,omega_cur)
                # and forward again!
                phi_cur+= omega_cur*kwargs.get('step',0.001)\
                    *nstep_omega*(nstepadfac_omega-1)
                z_cur, vz_cur= Aphi_to_zvz(A_cur,phi_cur,omega_cur)
                kwargs['nstep']= nstep_omega
                kwargs['eps']= 0. # Don't change weights
                dum= fit_m2m(tout[0],z_cur,vz_cur,omega_new,zsun_m2m,
                             z_obs,dens_obs,dens_obs_noise,
                             **kwargs)
                kwargs['nstep']= nstep
                kwargs['eps']= eps
                acc= (numpy.nansum(tQ)\
                          -numpy.mean(numpy.nansum(dum[1],axis=1)))/2.
                if acc > numpy.log(numpy.random.uniform()):
                    omega_m2m= omega_new
                    tQ= numpy.mean(dum[1],axis=0)
                    nacc_omega+= 1
                    # Update phase-space positions
                    phi_cur+= omega_new*nstep_omega*kwargs.get('step',0.001)
                    A_now= A_cur
                    phi_now= phi_cur
                    z_m2m, vz_m2m= Aphi_to_zvz(A_now,phi_now,omega_m2m)
            omega_out[ii]= omega_m2m
        w_out[ii]= tout[0]
        Q_out[ii]= tQ
        z_out[ii]= z_m2m
        vz_out[ii]= vz_m2m
    out= (w_out,)
    if fit_zsun: out= out+(zsun_out,)
    if fit_omega: out= out+(omega_out,)
    out= out+(Q_out,z_out,vz_out,)
    if fit_zsun: print("MH acceptance ratio for zsun was %.2f" \
                           % (nacc_zsun/float(nmh_zsun*nsamples)))
    if fit_omega: print("MH acceptance ratio for omega was %.2f" \
                            % (nacc_omega/float(nmh_omega*nsamples)))
    return out

