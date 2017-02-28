# hom2m.py: Simple Harmonic-Oscillator M2M implementation
import numpy
import copy
import simplex

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
#if True:
#    kernel= epanechnikov_kernel
#    kernel_deriv= epanechnikov_kernel_deriv
#else:
#    kernel= sph_kernel
#    kernel_deriv= sph_kernel_deriv

################################### OBSERVATIONS ##############################
### compute density at z_obs
def compute_dens(z,zsun,z_obs,h_obs,w=None,kernel=epanechnikov_kernel):
    if w is None: w= numpy.ones_like(z)/float(len(z))
    dens= numpy.zeros_like(z_obs)
    for jj,zo in enumerate(z_obs):
        dens[jj]= numpy.sum(w*kernel(numpy.fabs(zo-z+zsun),h_obs))
    return dens
### compute_v2
def compute_v2(z,vz,zsun,z_obs,h_obs,w=None,kernel=epanechnikov_kernel):
    if w is None: w= numpy.ones_like(z)/float(len(z))
    v2= numpy.zeros_like(z_obs)
    for jj,zo in enumerate(z_obs):
        v2[jj]= numpy.sum(w*kernel(numpy.fabs(zo-z+zsun),h_obs)*vz**2.)\
            /numpy.sum(w*kernel(numpy.fabs(zo-z+zsun),h_obs))
    return v2
### compute_densv2
def compute_densv2(z,vz,zsun,z_obs,h_obs,w=None,kernel=epanechnikov_kernel):
    if w is None: w= numpy.ones_like(z)/float(len(z))
    densv2= numpy.zeros_like(z_obs)
    for jj,zo in enumerate(z_obs):
        densv2[jj]= numpy.sum(w*kernel(numpy.fabs(zo-z+zsun),h_obs)*vz**2.)
    return densv2

############################### M2M FORCE-OF-CHANGE ###########################
# All defined here as the straight d constraint / d parameter (i.e., does *not*
# include things like eps, weight)

# Due to the prior
def force_of_change_entropy_weights(w_m2m,mu,w_prior):
    return -mu*(numpy.log(w_m2m/w_prior)+1.)

def force_of_change_dirichlet_weights(w_m2m,mu,w_prior):
    return w_prior/w_m2m

#Due to the density
#For the weights
def force_of_change_density_weights(w_m2m,zsun_m2m,z_m2m,vz_m2m,
                                    z_obs,dens_obs,dens_obs_noise,
                                    h_m2m=0.02,kernel=epanechnikov_kernel,
                                    delta_m2m=None):
    """Computes the force of change for all of the weights"""
    delta_m2m_new= numpy.zeros_like(z_obs)
    Wij= numpy.zeros((len(z_obs),len(z_m2m)))
    for jj,zo in enumerate(z_obs):
        Wij[jj]= kernel(numpy.fabs(zo-z_m2m+zsun_m2m),h_m2m)
        delta_m2m_new[jj]= (numpy.sum(w_m2m*Wij[jj])-dens_obs[jj])/dens_obs_noise[jj]
    if delta_m2m is None: delta_m2m= delta_m2m_new
    return (-numpy.sum(numpy.tile(delta_m2m/dens_obs_noise,(len(z_m2m),1)).T
                       *Wij,axis=0),
             delta_m2m_new)

# Due to the velocity
# For the weights
def force_of_change_densv2_weights(w_m2m,zsun_m2m,z_m2m,vz_m2m,
                                   z_obs,densv2_obs,densv2_obs_noise,
                                   h_m2m,kernel=epanechnikov_kernel,
                                   deltav2_m2m=None):
    """Computes the force of change for all of the weights due to the velocity constraint"""
    deltav2_m2m_new= numpy.zeros_like(z_obs)
    Wij= numpy.zeros((len(z_obs),len(z_m2m)))
    for jj,zo in enumerate(z_obs):
        Wij[jj]= kernel(numpy.fabs(zo-z_m2m+zsun_m2m),h_m2m)
        deltav2_m2m_new[jj]= (numpy.sum(w_m2m*Wij[jj]*vz_m2m**2.)
                              -densv2_obs[jj])/densv2_obs_noise[jj]
    if deltav2_m2m is None: deltav2_m2m= deltav2_m2m_new
    return (-numpy.sum(numpy.tile(deltav2_m2m/densv2_obs_noise,
                                  (len(z_m2m),1)).T*Wij,axis=0)*vz_m2m**2.,
             deltav2_m2m_new)

# Short-cuts
def force_of_change_weights(w_m2m,zsun_m2m,z_m2m,vz_m2m,
                            z_obs,dens_obs,dens_obs_noise,
                            densv2_obs,densv2_obs_noise,
                            prior,mu,w_prior,
                            h_m2m=0.02,kernel=epanechnikov_kernel,
                            delta_m2m=None,deltav2_m2m=None):
    """Computes the force of change for all of the weights"""
    fcw, delta_m2m_new=\
        force_of_change_density_weights(w_m2m,zsun_m2m,
                                        z_m2m,vz_m2m,
                                        z_obs,
                                        dens_obs,dens_obs_noise,
                                        h_m2m=h_m2m,kernel=kernel,
                                        delta_m2m=delta_m2m)
    # Add velocity constraint if given
    if not densv2_obs is None:
        fcwv2, deltav2_m2m_new= \
            force_of_change_densv2_weights(w_m2m,zsun_m2m,
                                           z_m2m,vz_m2m,
                                           z_obs,
                                           densv2_obs,densv2_obs_noise,
                                           h_m2m=h_m2m,
                                           kernel=kernel,
                                           deltav2_m2m=deltav2_m2m)
    else:
        fcwv2= 0.
        deltav2_m2m_new= None
    fcw+= fcwv2
    # Add prior
    if prior.lower() == 'entropy':
        fcw+= force_of_change_entropy_weights(w_m2m,mu,w_prior)
    else:
        fcw+= force_of_change_dirichlet_weights(w_m2m,mu,w_prior)
    return (fcw,delta_m2m_new,deltav2_m2m_new)

########################### M2M SECOND DERIVATIVES ############################
# All of these are minus times the actual second derivative (so they plug into 
# the calculation of the Hessian)

# Prior is diagonal, so these return just the diagonal

# Due to the prior
def force_of_change_entropy_weights_deriv(w_m2m,mu,w_prior):
    return mu/w_m2m

def force_of_change_dirichlet_weights_deriv(w_m2m,mu,w_prior):
    return mu*w_prior/w_m2m**2.

# The likelihood is full 2D, option to just return the diagonal

#Due to the density
#For the weights
def force_of_change_density_weights_deriv(w_m2m,zsun_m2m,z_m2m,vz_m2m,
                                          z_obs,dens_obs,dens_obs_noise,
                                          h_m2m=0.02,
                                          kernel=epanechnikov_kernel,
                                          diag=False):
    if diag:
        out= numpy.empty((len(z_m2m)))
    else:
        out= numpy.empty((len(z_m2m),len(z_m2m)))
    for jj,zo in enumerate(z_obs):
        Wij= numpy.atleast_2d(kernel(numpy.fabs(zo-z_m2m+zsun_m2m),h_m2m))
        if diag:
            out+= Wij[0]**2./dens_obs_noise[jj]**2.
        else:
            out+= numpy.dot(Wij.T,Wij)/dens_obs_noise[jj]**2.
    return out

# Due to the velocity
# For the weights
def force_of_change_densv2_weights_deriv(w_m2m,zsun_m2m,z_m2m,vz_m2m,
                                         z_obs,densv2_obs,densv2_obs_noise,
                                         h_m2m,kernel=epanechnikov_kernel,
                                         diag=False):
    if diag:
        out= numpy.empty((len(z_m2m)))
    else:
        out= numpy.empty((len(z_m2m),len(z_m2m)))
    for jj,zo in enumerate(z_obs):
        Wij= numpy.atleast_2d(vz_m2m**2.\
                                  *kernel(numpy.fabs(zo-z_m2m+zsun_m2m),h_m2m))
        if diag:
            out+= Wij[0]**2./densv2_obs_noise[jj]**2.
        else:
            out+= numpy.dot(Wij.T,Wij)/densv2_obs_noise[jj]**2.
    return out

################################ M2M OPTIMIZATION #############################
def run_m2m(w_init,z_init,vz_init,
            omega_m2m,zsun_m2m,
            z_obs,dens_obs,dens_obs_noise,
            densv2_obs=None,densv2_obs_noise=None,
            step=0.001,nstep=1000,
            eps=0.1,mu=1.,prior='entropy',
            kernel=epanechnikov_kernel,h_m2m=0.02,
            smooth=None,st96smooth=False,
            output_wevolution=False):
    """
    NAME:
       run_m2m
    PURPOSE:
       Run M2M on the harmonic-oscillator data to optimize
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
       step= stepsize of orbit integration
       nstep= number of steps to integrate the orbits for
       eps= M2M epsilon parameter
       mu= M2M entropy parameter mu
       prior= ('entropy' or 'dirichlet')
       kernel= a smoothing kernel
       h_m2m= kernel size parameter for computing the observables
       smooth= smoothing parameter alpha (None for no smoothing)
       st96smooth= (False) if True, smooth the constraints (Syer & Tremaine 1996), if False, smooth the objective function and its derivative (Dehnen 2000)
       output_wevolution= if set to an integer, return the time evolution of this many randomly selected weights
    OUTPUT:
       (w_out,Q_out,[wevol,rndindx]) - (output weights [N],objective function as a function of time,
                                       [weight evolution for randomly selected weights,index of random weights])
    HISTORY:
       2016-12-06 - Written - Bovy (UofT/CCA)
       2016-12-08 - Added output_wevolution - Bovy (UofT/CCA)
       2016-12-16 - Added explicit noise in the observed densities - Bovy (UofT/CCA)
       2017-02-22 - Refactored to more general function - Bovy (UofT/CCA)
       2017-02-27 - Add st96smooth option and make Dehnen smoothing the default - Bovy (UofT/CCA)
    """
    w_out= copy.deepcopy(w_init)
    Q_out= []
    A_init, phi_init= zvz_to_Aphi(z_init,vz_init,omega_m2m)
    if output_wevolution:
        rndindx= numpy.random.permutation(len(w_out))[:output_wevolution]
        wevol= numpy.zeros((output_wevolution,nstep))
    # Compute force of change for first iteration
    fcw, delta_m2m_new, deltav2_m2m_new= \
        force_of_change_weights(w_init,zsun_m2m,z_init,vz_init,
                                z_obs,dens_obs,dens_obs_noise,
                                densv2_obs,densv2_obs_noise,
                                prior,mu,w_init,
                                h_m2m=h_m2m,kernel=kernel)
    fcw*= w_init
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
    for ii in range(nstep):
        # Update weights first
        w_out+= eps*step*fcw
        w_out[w_out < 0.]= 10.**-16.
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
        phi_now= omega_m2m*ii*step+phi_init
        z_m2m= A_init*numpy.cos(phi_now)
        vz_m2m= -A_init*omega_m2m*numpy.sin(phi_now)
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
                                    prior,mu,w_init,
                                    h_m2m=h_m2m,kernel=kernel,
                                    delta_m2m=tdelta_m2m,
                                    deltav2_m2m=tdeltav2_m2m)
        fcw_new*= w_out
        # Increment smoothing
        if not smooth is None and st96smooth:
            delta_m2m+= step*smooth*(delta_m2m_new-delta_m2m)
            deltav2_m2m+= step*smooth*(deltav2_m2m_new-deltav2_m2m)
            fcw= fcw_new
        elif not smooth is None:
            if not densv2_obs is None:
                Q_new= numpy.hstack((delta_m2m_new**2.,deltav2_m2m_new**2.))
            else:
                Q_new= delta_m2m_new**2.
            Q+= step*smooth*(Q_new-Q)
            fcw+= step*smooth*(fcw_new-fcw)
        else:
            fcw= fcw_new
        # Record random weights if requested
        if output_wevolution:
            wevol[:,ii]= w_out[rndindx]
    out= (w_out,numpy.array(Q_out))
    if output_wevolution:
        out= out+(wevol,rndindx,)
    return out

def estimate_hessian_m2m(w_out,z_init,vz_init,
                         omega_m2m,zsun_m2m,
                         z_obs,dens_obs,dens_obs_noise,
                         densv2_obs=None,densv2_obs_noise=None,
                         w_prior=None,
                         step=0.001,nstep=1000,mu=1.,
                         h_m2m=0.02,kernel=epanechnikov_kernel,
                         prior='entropy',diag=False):
    """
    NAME:
       estimate_hessian__m2m
    PURPOSE:
       Estimate the Hessian for an M2M optimization
    INPUT:
       w_out - weights [N]
       z_init - initial z [N]
       vz_init - initial vz (rad) [N]
       omega_m2m - potential parameter omega
       zsun_m2m - Sun's height above the plane [N]
       z_obs - heights at which the density observations are made
       dens_obs - observed densities
       dens_obs_noise - noise in the observed densities
       densv2_obs= (None) observed density x velocity-squareds (optional)
       densv2_obs_noise= (None) noise in the observed densities x velocity-squareds
       w_prior= (None) prior weights
       step= stepsize of orbit integration
       nstep= number of steps to integrate the orbits for
       mu= M2M entropy parameter mu
       h_m2m= kernel size parameter for computing the observables
       kernel= a smoothing kernel
       prior= ('entropy' or 'dirichlet')
       diag= (False) if True, only copute the diagonal of the Hessian
    OUTPUT:
       Hessian
    HISTORY:
       2017-02-24 - Written - Bovy (UofT/CCA)
    """
    if w_prior is None:
        w_prior= numpy.ones_like(w_out)/float(len(w_out))
    A_init, phi_init= zvz_to_Aphi(z_init,vz_init,omega_m2m)
    if diag:
        out= numpy.empty((len(w_out)))
    else:
        out= numpy.empty((len(w_out),len(w_out)))
    # To easily deal with the shape of the matrix for the prior
    if diag:
        prior_shape= lambda x: x
    else:
        prior_shape= lambda x: numpy.diagflat(x)
    for ii in range(nstep):
        # Compute current (z,vz)
        phi_now= omega_m2m*ii*step+phi_init
        z_m2m, vz_m2m= Aphi_to_zvz(A_init,phi_now,omega_m2m)
        # Evaluate second derivatives
        out+= force_of_change_density_weights_deriv(\
            w_out,zsun_m2m,z_m2m,vz_m2m,
            z_obs,dens_obs,dens_obs_noise,
            h_m2m,kernel=kernel,diag=diag)
        # Add velocity constraint if given
        if not densv2_obs is None:
            out+= force_of_change_densv2_weights_deriv(\
                w_out,zsun_m2m,z_m2m,vz_m2m,
                z_obs,densv2_obs,densv2_obs_noise,
                h_m2m,kernel=kernel,diag=diag)
        # Add prior
        if prior.lower() == 'entropy':
            out+= prior_shape(force_of_change_entropy_weights_deriv(\
                    w_out,mu,w_prior))
        else:
            out+= prior_shape(force_of_change_dirichlet_weights_deriv(\
                    w_out,mu,w_prior))
    return 0.5*(out+out.T)/nstep # numerical inaccuracy

def run_m2m_hmc(w_init,z_init,vz_init,
                omega_m2m,zsun_m2m,
                z_obs,dens_obs,dens_obs_noise,
                densv2_obs=None,densv2_obs_noise=None,
                mi=None,w_prior=None,
                step=0.001,nstep=1000,nleap=10,nobj=50,
                eps=0.1,mu=1.,
                h_m2m=0.02,kernel=epanechnikov_kernel,
                prior='entropy'):
    """
    NAME:
       run_m2m_hmc
    PURPOSE:
       Run M2M HMC on the harmonic-oscillator data
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
       mi= (None) masses for the HMC algorithm
       w_prior= (None) prior weights
       step= stepsize of orbit integration
       nstep= sets integration time: number of orbit steps = nstep x nleap, st nstep = nsamples
       nleap= number of orbit steps between momentum resamplings
       nobj= number of orbit steps to use to average the objective function
       eps= M2M epsilon parameter
       mu= M2M entropy parameter mu
       h_m2m= kernel size parameter for computing the observables
       kernel= a smoothing kernel
       prior= ('entropy' or 'dirichlet')
    OUTPUT:
       (w_out,Q_out,[wevol,rndindx]) - (output weights [N],objective function as a function of time,
                                       [weight evolution for randomly selected weights,index of random weights])
    HISTORY:
       2016-02-27 - Written - Bovy (UofT/CCA)
    """
    w_out= numpy.empty((nstep,len(w_init)))
    if mi is None: mi= numpy.ones_like(w_init)
    sqmi= numpy.sqrt(mi)
    if w_prior is None: w_prior= numpy.ones_like(w_init)/float(len(w_init))
    Q_out= []
    A_init, phi_init= zvz_to_Aphi(z_init,vz_init,omega_m2m)
    phi_now= phi_init
    # Compute initial force of change and initial objective function
    w_cur= copy.deepcopy(w_init)
    Vb4= 0.
    fcw= 0.
    for kk in range(nobj):
        phi_now+= omega_m2m*step
        z_m2m= z_init#A_init*numpy.cos(phi_now)
        vz_m2m= vz_init#-A_init*omega_m2m*numpy.sin(phi_now)
        fcwt, delta_m2m, deltav2_m2m= \
            force_of_change_weights(w_cur,zsun_m2m,z_m2m,vz_m2m,
                                    z_obs,dens_obs,dens_obs_noise,
                                    densv2_obs,densv2_obs_noise,
                                    prior,mu,w_init,
                                    h_m2m=h_m2m,kernel=kernel)
        # Compute potential energy
        if densv2_obs is None:
            deltav2_m2m= 0.
        Vb4+= 0.5*numpy.sum(delta_m2m**2.+deltav2_m2m**2.)
        fcw+= fcwt
    Vb4/= nobj
    fcw/= nobj
    if prior.lower() == 'entropy':
        Vb4+= +mu*numpy.sum(w_cur*numpy.log(w_cur/w_prior))
    else:
        Vb4+= -mu*numpy.sum(w_prior*numpy.log(w_cur))
    naccept= 0.
    for ii in range(nstep):
        # Sample momentum
        pw= numpy.random.normal(size=len(w_init))*sqmi
        # Compute Hamiltonian
        Hb4= 0.5*numpy.sum(pw**2./mi)+Vb4
        new_w= copy.deepcopy(w_cur)
        # Perform leapfrog integration
        pwi= copy.deepcopy(pw)
        pw+= step/2.*eps*fcw # half a step to get started
        for jj in range(1,nleap+1):
            new_w+= step*eps*pw/mi
            pw[new_w<0.]*= -1. # bounce
            new_w[new_w<0]*= -1.
#            phi_now+= omega_m2m*step
#            z_m2m= A_init*numpy.cos(phi_now)
#            vz_m2m= -A_init*omega_m2m*numpy.sin(phi_now)
            new_grad, delta_m2m, deltav2_m2m=\
                force_of_change_weights(new_w,zsun_m2m,z_m2m,vz_m2m,
                                        z_obs,dens_obs,dens_obs_noise,
                                        densv2_obs,densv2_obs_noise,
                                        prior,mu,w_init,
                                        h_m2m=h_m2m,kernel=kernel)
            pw+= step*eps*new_grad/(1.+ (jj == nleap)) #Full steps, excl. last
        # Compute Hamiltonian after leapfrogging
        V= 0.
        new_grad= 0.
        for kk in range(nobj):
            phi_now+= omega_m2m*step
            z_m2m= z_init#A_init*numpy.cos(phi_now)
            vz_m2m= vz_init#-A_init*omega_m2m*numpy.sin(phi_now)
            new_gradt, delta_m2m, deltav2_m2m=\
                force_of_change_weights(new_w,zsun_m2m,z_m2m,vz_m2m,
                                        z_obs,dens_obs,dens_obs_noise,
                                        densv2_obs,densv2_obs_noise,
                                        prior,mu,w_init,
                                        h_m2m=h_m2m,kernel=kernel)
            # Compute potential energy
            if densv2_obs is None:
                deltav2_m2m= 0.
            V+= 0.5*numpy.sum(delta_m2m**2.+deltav2_m2m**2.)
            new_grad+= new_gradt
        V/= nobj
        new_grad/= nobj
        if prior.lower() == 'entropy':
            V+= +mu*numpy.sum(new_w*numpy.log(new_w/w_prior))
        else:
            V+= -mu*numpy.sum(w_prior*numpy.log(new_w))
        H= 0.5*numpy.sum(pw**2./mi)+V
        dH= H-Hb4
        #print(dH,V-Vb4,0.5*numpy.sum((pw**2.-pwi**2.)/mi),V)
        dH= dH * ( dH > 0 )
        #Metropolis accept
        if numpy.random.uniform() < numpy.exp(-dH):
            w_cur= copy.deepcopy(new_w)
            Vb4= V
            fcw= new_grad
            naccept+= 1.
        w_out[ii]= w_cur
        Q_out.append(Vb4)
    return (w_out,naccept,Q_out)

### zsun force of change

def force_of_change_zsun(w_m2m,zsun_m2m,z_m2m,vz_m2m,
                         eps,mu,w_prior,
                         z_obs,dens_obs,dens_obs_noise,
                         h_m2m=0.02,
                         delta_m2m=None):
    """Computes the force of change for zsun"""
    out= 0.
    for jj,zo in enumerate(z_obs):
        dWij= kernel_deriv(numpy.fabs(zo-z_m2m+zsun_m2m),h_m2m)*numpy.sign(zo-z_m2m+zsun_m2m)
        out+= delta_m2m[jj]/dens_obs_noise[jj]*numpy.sum(w_m2m*dWij)/len(z_m2m)
    return -eps*out

### run M2M with zsun change

def run_m2m_weights_zsun(w_init,A_init,phi_init,
                         omega_m2m,zsun_m2m,
                         z_obs,dens_obs,dens_obs_noise,
                         step=0.001,nstep=1000,
                         eps=0.1,eps_zo=0.001,mu=1.,
                         h_m2m=0.02,
                         smooth=None,
                         output_wevolution=False):
    """
    NAME:
       run_m2m_weights_zsun
    PURPOSE:
       Run M2M on the harmonic-oscillator data to optimize the weights of the orbits as well as the Sun's height
    INPUT:
       w_init - initial weights [N]
       z_init - initial z [N]
       vz_init - initial vz [N]
       omega_m2m - potential parameter omega
       zsun_m2m - Sun's height above the plane [N]
       z_obs - heights at which the density observations are made
       dens_obs - observed densities
       dens_obs_noise - noise in the observed densities
       step= stepsize of orbit integration
       nstep= number of steps to integrate the orbits for
       eps= M2M epsilon parameter
       eps_zo= M2M epsilon parameter for zsun force of change
       mu= M2M entropy parameter mu
       h_m2m= kernel size parameter for computing the observables
       smooth= smoothing parameter alpha (None for no smoothing)
       output_wevolution= if set to an integer, return the time evolution of this many randomly selected weights
    OUTPUT:
       ((w_out,zsun_out),Q_out,[wevol,rndindx]) - ((output weights [N],output zsun [nstep]),
                                                    objective function as a function of time,
                                                   [weight evolution for randomly selected weights,
                                                   index of random weights])
    HISTORY:
       2016-12-06 - Written - Bovy (UofT/CCA)
       2016-12-09 - Added output_wevolution - Bovy (UofT/CCA)
       2016-12-16 - Added explicit noise in the observed densities - Bovy (UofT/CCA)
    """
    w_out= copy.deepcopy(w_init)
    zsun_out= []
    Q_out= []
    if output_wevolution:
        rndindx= numpy.random.permutation(len(w_out))[:output_wevolution]
        wevol= numpy.zeros((output_wevolution,nstep))
    if not smooth is None:
        # Smooth the constraints
        phi_now= phi_init
        z_m2m= A_init*numpy.cos(phi_now)
        vz_m2m= -A_init*omega_m2m*numpy.sin(phi_now) # unnecessary
        dens_init= compute_dens(z_m2m,zsun_m2m,z_obs,h_m2m,w=w_init)
        delta_m2m= (dens_init-dens_obs)/dens_obs_noise
    else:
        delta_m2m= None
    for ii in range(nstep):
        # Compute current (z,vz)
        phi_now= omega_m2m*ii*step+phi_init
        z_m2m= A_init*numpy.cos(phi_now)
        vz_m2m= -A_init*omega_m2m*numpy.sin(phi_now) # unnecessary
        fcw, delta_m2m_new= force_of_change_weights(w_out,zsun_m2m,
                                                    z_m2m,vz_m2m,
                                                    eps,mu,w_init,
                                                    z_obs,dens_obs,dens_obs_noise,
                                                    h_m2m=h_m2m,
                                                    delta_m2m=delta_m2m)
        if smooth is None:
            fcz= force_of_change_zsun(w_out,zsun_m2m,
                                      z_m2m,vz_m2m,
                                      eps_zo,mu,w_init,
                                      z_obs,dens_obs,dens_obs_noise,
                                      h_m2m=h_m2m,delta_m2m=delta_m2m_new)
        else:
            fcz= force_of_change_zsun(w_out,zsun_m2m,
                                      z_m2m,vz_m2m,
                                      eps_zo,mu,w_init,
                                      z_obs,dens_obs,dens_obs_noise,
                                      h_m2m=h_m2m,delta_m2m=delta_m2m)
        #print(fcz)
        w_out+= step*fcw
        w_out/= numpy.sum(w_out)/len(A_init)
        w_out[w_out < 0.]= 10.**-16.
        zsun_m2m+= step*fcz
        zsun_out.append(zsun_m2m)
        Q_out.append(delta_m2m_new**2.)
        # Increment smoothing
        if not smooth is None:
            delta_m2m= step*smooth*(delta_m2m_new-delta_m2m)
        if output_wevolution:
            wevol[:,ii]= w_out[rndindx]
    out= ((w_out,numpy.array(zsun_out)),numpy.array(Q_out))
    if output_wevolution:
        out= out+(wevol,rndindx,)
    return out


### force_of_change_densv2_weights

### run_m2m_weights_wdensv2

def run_m2m_weights_wdensv2(w_init,A_init,phi_init,
                        omega_m2m,zsun_m2m,
                        z_obs,
                        dens_obs,dens_obs_noise,
                        densv2_obs,densv2_obs_noise,
                        step=0.001,nstep=1000,
                        eps=0.1,eps_vel=0.1,mu=1.,
                        h_m2m=0.02,
                        smooth=None,nodens=False,
                        output_wevolution=False):
    """
    NAME:
       run_m2m_weights_wv2
    PURPOSE:
       Run M2M on the harmonic-oscillator data to optimize just the weights of the orbits, using both density and
       velocity data
    INPUT:
       w_init - initial weights [N]
       A_init - initial zmax [N]
       phi_init - initial angle (rad) [N]
       omega_m2m - potential parameter omega
       zsun_m2m - Sun's height above the plane [N]
       z_obs - heights at which the density observations are made
       dens_obs - observed densities
       dens_obs_noise - noise in the observed densities
       densv2_obs - observed density x velocity-squareds
       densv2_obs_noise - noise in the observed densities x velocity-squareds
       step= stepsize of orbit integration
       nstep= number of steps to integrate the orbits for
       eps= M2M epsilon parameter
       eps_vel= M2M epsilon parameter for the velocity change
       mu= M2M entropy parameter mu
       h_m2m= kernel size parameter for computing the observables
       smooth= smoothing parameter alpha (None for no smoothing)
       nodens= if True, don't fit the density data (also doesn't include entropy term)
       output_wevolution= if set to an integer, return the time evolution of this many randomly selected weights
    OUTPUT:
       (w_out,Q_out,[wevol,rndindx]) - (output weights [N],objective function as a function of time,
                                       [weight evolution for randomly selected weights,index of random weights])
    HISTORY:
       2016-12-07 - Written - Bovy (UofT/CCA)
       2016-12-09 - Added output_wevolution - Bovy (UofT/CCA)
       2016-12-16 - Added explicit noise in the observed densities and densities x velocity-squared - Bovy (UofT/CCA)
    """
    w_out= copy.deepcopy(w_init)
    Q_out= []
    if output_wevolution:
        rndindx= numpy.random.permutation(len(w_out))[:output_wevolution]
        wevol= numpy.zeros((output_wevolution,nstep))
    if not smooth is None:
        # Smooth the constraints
        phi_now= phi_init
        z_m2m= A_init*numpy.cos(phi_now)
        vz_m2m= -A_init*omega_m2m*numpy.sin(phi_now) # unnecessary
        dens_init= compute_dens(z_m2m,zsun_m2m,z_obs,h_m2mw=w_init)
        delta_m2m= (dens_init-dens_obs)/dens_obs_noise
        densv2_init= compute_densv2(z_m2m,vz_m2m,zsun_m2m,z_obs,w=w_init)
        deltav2_m2m= (densv2_init-densv2_obs)/densv2_obs_noise
    else:
        delta_m2m= None
        deltav2_m2m= None
    for ii in range(nstep):
        # Compute current (z,vz)
        phi_now= omega_m2m*ii*step+phi_init
        z_m2m= A_init*numpy.cos(phi_now)
        vz_m2m= -A_init*omega_m2m*numpy.sin(phi_now) # unnecessary
        fcw, delta_m2m_new= force_of_change_weights(w_out,zsun_m2m,
                                                    z_m2m,vz_m2m,
                                                    eps,mu,w_init,
                                                    z_obs,dens_obs,dens_obs_noise,
                                                    h_m2m=h_m2m,
                                                    delta_m2m=delta_m2m)
        fcwv2, deltav2_m2m_new= force_of_change_weights_densv2(w_out,zsun_m2m,
                                                           z_m2m,vz_m2m,
                                                           eps_vel,mu,w_init,
                                                           z_obs,
                                                           densv2_obs,densv2_obs_noise,
                                                           h_m2m=h_m2m,
                                                           deltav2_m2m=deltav2_m2m)
        w_out+= step*fcw*(1.-nodens)+step*fcwv2
        w_out/= numpy.sum(w_out)/len(A_init)
        w_out[w_out < 0.]= 10.**-16.
        if not smooth is None:
            Q_out.append(delta_m2m**2.*(1.-nodens)+deltav2_m2m**2.)
        else:
            Q_out.append(delta_m2m_new**2.*(1.-nodens)+deltav2_m2m_new**2.)
        # Increment smoothing
        if not smooth is None:
            delta_m2m+= step*smooth*(delta_m2m_new-delta_m2m)
            deltav2_m2m+= step*smooth*(deltav2_m2m_new-deltav2_m2m)
        if output_wevolution:
            wevol[:,ii]= w_out[rndindx]
    out= (w_out,numpy.array(Q_out))
    if output_wevolution:
        out= out+(wevol,rndindx,)
    return out

### compute_v2
# compute <v^2>

def compute_v2m(z,vz,zsun,z_obs,h_obs,w=None):
    if w is None: w= numpy.ones_like(z)
    v2m= numpy.zeros_like(z_obs)
    for jj,zo in enumerate(z_obs):
        v2m[jj]= numpy.sum(w*kernel(numpy.fabs(zo-z+zsun),h_obs)*vz**2.) \
          /numpy.sum(w*kernel(numpy.fabs(zo-z+zsun),h_obs))
    return v2m

# computer Number of stars or model particles within h_obs

#def compute_nsbin(z,zsun,z_obs,h_obs,w=None):
#    if w is None: w= numpy.ones_like(z)
#    dens= numpy.zeros_like(z_obs)
#    for jj,zo in enumerate(z_obs):
#       dens[jj]+= numpy.sum(w[numpy.fabs(zo-z+zsun)<h_obs])
#    return nsbin

### force_of_change_weights_v2m
def force_of_change_weights_v2m(w_m2m,zsun_m2m,z_m2m,vz_m2m,
                               eps,mu,w_prior,
                               z_obs,v2m_obs,v2m_obs_noise,
                               h_m2m=0.02,
                               deltav2m_m2m=None):
    """Computes the force of change for all of the weights due to the velocity constraint"""
    deltav2m_m2m_new= numpy.zeros_like(z_obs)
    Wij= numpy.zeros((len(z_obs),len(z_m2m)))
    dens_m2m= numpy.zeros_like(z_obs)
    wv2m_m2m= numpy.zeros_like(z_obs)
    for jj,zo in enumerate(z_obs):
        Wij[jj]= kernel(numpy.fabs(zo-z_m2m+zsun_m2m),h_m2m)
        dens_m2m[jj]=numpy.sum(w_m2m*Wij[jj])
        wv2m_m2m[jj]=numpy.sum(w_m2m*Wij[jj]*(vz_m2m**2))
        deltav2m_m2m_new[jj]= (wv2m_m2m[jj]/dens_m2m[jj]-v2m_obs[jj])/v2m_obs_noise[jj]
    if deltav2m_m2m is None: deltav2m_m2m= deltav2m_m2m_new
    if numpy.any(dens_m2m==0.0)==True:
        print ' dens_m2m has zero. dens_m2m==0 ',np.where(dens_m2m==0)
        print ' zsun= ',len(zsun_m2m),zsun_m2m
        sys.exit()
    return (-eps*w_m2m*(numpy.sum(numpy.tile(deltav2m_m2m/(dens_m2m*v2m_obs_noise),(len(z_m2m),1)).T*(Wij),axis=0) 
        *(vz_m2m**2.)-numpy.sum(numpy.tile(
        deltav2m_m2m*wv2m_m2m/((dens_m2m**2)*v2m_obs_noise),(len(z_m2m),1)).T*(Wij),axis=0)),
        deltav2m_m2m_new)

### run_m2m_weights_wv2m

def run_m2m_weights_wv2m(w_init,A_init,phi_init,
                        omega_m2m,zsun_m2m,
                        z_obs,
                        dens_obs,dens_obs_noise,
                        v2m_obs,v2m_obs_noise,
                        step=0.001,nstep=1000,
                        eps=0.1,eps_vel=0.1,mu=1.,
                        h_m2m=0.02,
                        smooth=None,nodens=False,
                        output_wevolution=False):
    """
    NAME:
       run_m2m_weights_wv2m
    PURPOSE:
       Run M2M on the harmonic-oscillator data to optimize just the weights of the orbits, using <v^2>
    INPUT:
       w_init - initial weights [N]
       A_init - initial zmax [N]
       phi_init - initial angle (rad) [N]
       omega_m2m - potential parameter omega
       zsun_m2m - Sun's height above the plane [N]
       z_obs - heights at which the density observations are made
       dens_obs - observed densities
       dens_obs_noise - noise in the observed densities
       v2m_obs - observed velocity-squareds
       v2m_obs_noise - noise in the observed velocity-squareds
       step= stepsize of orbit integration
       nstep= number of steps to integrate the orbits for
       eps= M2M epsilon parameter
       eps_vel= M2M epsilon parameter for the velocity change
       mu= M2M entropy parameter mu
       h_m2m= kernel size parameter for computing the observables
       smooth= smoothing parameter alpha (None for no smoothing)
       nodens= if True, don't fit the density data (also doesn't include entropy term)
       output_wevolution= if set to an integer, return the time evolution of this many randomly selected weights
    OUTPUT:
       (w_out,Q_out,[wevol,rndindx]) - (output weights [N],objective function as a function of time,
                                       [weight evolution for randomly selected weights,index of random weights])
    HISTORY:
       2016-12-07 - Written - Bovy (UofT/CCA)
       2016-12-09 - Added output_wevolution - Bovy (UofT/CCA)
       2016-12-16 - Added explicit noise in the observed densities and densities x velocity-squared - Bovy (UofT/CCA)
       2017-01-31 - modified to use <v^2> - Kawata (MSSL,UCL)
    """
    w_out= copy.deepcopy(w_init)
    Q_out= []
    if output_wevolution:
        rndindx= numpy.random.permutation(len(w_out))[:output_wevolution]
        wevol= numpy.zeros((output_wevolution,nstep))
    if not smooth is None:
        # Smooth the constraints
        phi_now= phi_init
        z_m2m= A_init*numpy.cos(phi_now)
        vz_m2m= -A_init*omega_m2m*numpy.sin(phi_now) # unnecessary
        dens_init= compute_dens(z_m2m,zsun_m2m,z_obs,h_m2m,w=w_init)
        delta_m2m= (dens_init-dens_obs)/dens_obs_noise
        v2m_init= compute_v2m(z_m2m,vz_m2m,zsun_m2m,z_obs,h_m2m,w=w_init)
        deltav2m_m2m= (v2m_init-v2m_obs)/v2m_obs_noise
    else:
        delta_m2m= None
        deltav2m_m2m= None
    for ii in range(nstep):
        # Compute current (z,vz)
        phi_now= omega_m2m*ii*step+phi_init
        z_m2m= A_init*numpy.cos(phi_now)
        vz_m2m= -A_init*omega_m2m*numpy.sin(phi_now) # unnecessary
        fcw, delta_m2m_new= force_of_change_weights(w_out,zsun_m2m,
                                                    z_m2m,vz_m2m,
                                                    eps,mu,w_init,
                                                    z_obs,dens_obs,dens_obs_noise,
                                                    h_m2m=h_m2m,
                                                    delta_m2m=delta_m2m)
        fcwv2m, deltav2m_m2m_new= force_of_change_weights_v2m(w_out,zsun_m2m,
                                                           z_m2m,vz_m2m,
                                                           eps_vel,mu,w_init,
                                                           z_obs,
                                                           v2m_obs,v2m_obs_noise,
                                                           h_m2m=h_m2m,
                                                           deltav2m_m2m=deltav2m_m2m)
        w_out+= step*fcw*(1.-nodens)+step*fcwv2m
        w_out/= numpy.sum(w_out)/len(A_init)
        w_out[w_out < 0.]= 10.**-16.
        if not smooth is None:
            Q_out.append(delta_m2m**2.*(1.-nodens)+deltav2m_m2m**2.)
        else:
            Q_out.append(delta_m2m_new**2.*(1.-nodens)+deltav2m_m2m_new**2.)
        # Increment smoothing
        if not smooth is None:
            delta_m2m+= step*smooth*(delta_m2m_new-delta_m2m)
            deltav2m_m2m+= step*smooth*(deltav2m_m2m_new-deltav2m_m2m)
        if output_wevolution:
            wevol[:,ii]= w_out[rndindx]
    out= (w_out,numpy.array(Q_out))
    if output_wevolution:
        out= out+(wevol,rndindx,)
    return out

### force_of_change_zsun_densv2m

def force_of_change_zsun_densv2m(w_m2m,zsun_m2m,z_m2m,vz_m2m,
                         eps,mu,w_prior,
                         z_obs,dens_obs,dens_obs_noise,v2m_obs,v2m_obs_noise,
                         eps_velw=1.0,h_m2m=0.02,
                         delta_m2m=None,deltav2m_m2m=None):
    """Computes the force of change for zsun from density and <v^2> constraints"""
    dens_m2m= numpy.zeros_like(z_obs)
    wv2m_m2m= numpy.zeros_like(z_obs)
    for jj,zo in enumerate(z_obs):
        Wij= kernel(numpy.fabs(zo-z_m2m+zsun_m2m),h_m2m)
        dens_m2m[jj]=numpy.sum(w_m2m*Wij)
        wv2m_m2m[jj]=numpy.sum(w_m2m*Wij*(vz_m2m**2))

    
    out= 0.
    for jj,zo in enumerate(z_obs):
        # from density
        dWij= kernel_deriv(numpy.fabs(zo-z_m2m+zsun_m2m),h_m2m)*numpy.sign(zo-z_m2m+zsun_m2m)
        wdWj= numpy.sum(w_m2m*dWij)
        out+= delta_m2m[jj]/dens_obs_noise[jj]*wdWj/len(z_m2m)
        # from velocity
        wv2mdWj= numpy.sum(w_m2m*(vz_m2m**2)*dWij)
        out+= eps_velw*(deltav2m_m2m[jj]/v2m_obs_noise[jj]) \
              *((wv2mdWj/dens_m2m[jj])-(wv2m_m2m[jj]/(dens_m2m[jj]**2))*wdWj)
    return -eps*out

### run M2M with zsun change

def run_m2m_weights_zsun_densv2m(w_init,A_init,phi_init,
                        omega_m2m,zsun_m2m,
                        z_obs,dens_obs,dens_obs_noise,v2m_obs,v2m_obs_noise,
                        step=0.001,nstep=1000,
                        eps=0.1,eps_vel=0.1,eps_zo=0.001,mu=1.,
                        h_m2m=0.02,
                        smooth=None,
                        output_wevolution=False):
    """
    NAME:
       run_m2m_weights_zsun_densv2m
    PURPOSE:
       Run M2M on the harmonic-oscillator data to optimize the weights of the orbits as well as the Sun's height constrained from both density and <v^2>
    INPUT:
       w_init - initial weights [N]
       A_init - initial zmax [N]
       phi_init - initial angle (rad) [N]
       omega_m2m - potential parameter omega
       zsun_m2m - Sun's height above the plane [N]
       z_obs - heights at which the density observations are made
       dens_obs - observed densities
       dens_obs_noise - noise in the observed densities
       v2m_obs - observed velocity-squareds
       v2m_obs_noise - noise in the observed velocity-squareds
       step= stepsize of orbit integration
       nstep= number of steps to integrate the orbits for
       eps= M2M epsilon parameter
       eps_vel= M2M epsilon parameter for the velocity change
       eps_zo= M2M epsilon parameter for zsun force of change
       mu= M2M entropy parameter mu
       h_m2m= kernel size parameter for computing the observables
       smooth= smoothing parameter alpha (None for no smoothing)
       output_wevolution= if set to an integer, return the time evolution of this many randomly selected weights
    OUTPUT:
       ((w_out,zsun_out),Q_out,[wevol,rndindx]) - ((output weights [N],output zsun [nstep]),
                                                    objective function as a function of time,
                                                   [weight evolution for randomly selected weights,
                                                   index of random weights])
    HISTORY:
       2016-12-06 - Written - Bovy (UofT/CCA)
       2016-12-09 - Added output_wevolution - Bovy (UofT/CCA)
       2016-12-16 - Added explicit noise in the observed densities - Bovy (UofT/CCA)
       2017-02-07 - modified from run_m2m_weights_zsun - Kawata (MSSL/UCL)
    """
    w_out= copy.deepcopy(w_init)
    zsun_out= []
    Q_out= []
    if output_wevolution:
        rndindx= numpy.random.permutation(len(w_out))[:output_wevolution]
        wevol= numpy.zeros((output_wevolution,nstep))
    if not smooth is None:
        # Smooth the constraints
        phi_now= phi_init
        z_m2m= A_init*numpy.cos(phi_now)
        vz_m2m= -A_init*omega_m2m*numpy.sin(phi_now) # unnecessary
        dens_init= compute_dens(z_m2m,zsun_m2m,z_obs,h_m2m,w=w_init)
        delta_m2m= (dens_init-dens_obs)/dens_obs_noise
        v2m_init=compute_v2m(z_m2m,vz_m2m,zsun_m2m,z_obs,h_m2m,w=w_init)
        deltav2m_m2m= (v2m_init-v2m_obs)/v2m_obs_noise
    else:
        delta_m2m= None
        deltav2m_m2m= None
    for ii in range(nstep):
        # Compute current (z,vz)
        phi_now= omega_m2m*ii*step+phi_init
        z_m2m= A_init*numpy.cos(phi_now)
        vz_m2m= -A_init*omega_m2m*numpy.sin(phi_now)
        fcw, delta_m2m_new= force_of_change_weights(w_out,zsun_m2m,
                                                    z_m2m,vz_m2m,
                                                    eps,mu,w_init,
                                                    z_obs,dens_obs,dens_obs_noise,
                                                    h_m2m=h_m2m,
                                                    delta_m2m=delta_m2m)
        fcwv2m, deltav2m_m2m_new= force_of_change_weights_v2m(w_out,zsun_m2m,
                                                    z_m2m,vz_m2m,
                                                    eps_vel,mu,w_init,
                                                    z_obs,
                                                    v2m_obs,v2m_obs_noise,
                                                    h_m2m=h_m2m,
                                                    deltav2m_m2m=deltav2m_m2m)
# constraints from density and velocity
        if smooth is None:
            fcz= force_of_change_zsun_densv2m(w_out,zsun_m2m,
                                      z_m2m,vz_m2m,
                                      eps_zo,mu,w_init,
                                      z_obs,dens_obs,dens_obs_noise,
                                      v2m_obs,v2m_obs_noise,eps_velw=1.0,
                                      h_m2m=h_m2m,delta_m2m=delta_m2m_new,deltav2m_m2m=deltav2m_m2m_new)
        else:
            fcz= force_of_change_zsun_densv2m(w_out,zsun_m2m,
                                      z_m2m,vz_m2m,
                                      eps_zo,mu,w_init,
                                      z_obs,dens_obs,dens_obs_noise,
                                      v2m_obs,v2m_obs_noise,eps_velw=1.0,
                                      h_m2m=h_m2m,delta_m2m=delta_m2m,deltav2m_m2m=deltav2m_m2m_new)
        w_out+= step*fcw+step*fcwv2m
        w_out/= numpy.sum(w_out)/len(A_init)
        w_out[w_out < 0.]= 10.**-16.
        zsun_m2m+= step*fcz
        zsun_out.append(zsun_m2m)
        # print 'step,fcz=',ii,fcz,zsun_m2m
        if not smooth is None:
            Q_out.append(delta_m2m**2.+deltav2m_m2m**2.)
        else:
            Q_out.append(delta_m2m_new**2.+deltav2m_m2m_new**2.)
        # Increment smoothing
        if not smooth is None:
            delta_m2m= step*smooth*(delta_m2m_new-delta_m2m)
            deltav2m_m2m+= step*smooth*(deltav2m_m2m_new-deltav2m_m2m)
        if output_wevolution:
            wevol[:,ii]= w_out[rndindx]
    out= ((w_out,numpy.array(zsun_out)),numpy.array(Q_out))
    if output_wevolution:
        out= out+(wevol,rndindx,)
    return out

# Functions that return the difference in z and vz for orbits starting at the same (z,vz)_init integrated
# in potentials with different omega
def zdiff(z_init,vz_init,omega1,omega2,t):
    A1= numpy.sqrt(z_init**2.+vz_init**2./omega1**2.)
    A2= numpy.sqrt(z_init**2.+vz_init**2./omega2**2.)
    phi1= numpy.arctan2(-vz_init/omega1,z_init)
    phi2= numpy.arctan2(-vz_init/omega2,z_init)
    return A2*numpy.cos(omega2*t+phi2)-A1*numpy.cos(omega1*t+phi1)
def vzdiff(z_init,vz_init,omega1,omega2,t):
    A1= numpy.sqrt(z_init**2.+vz_init**2./omega1**2.)
    A2= numpy.sqrt(z_init**2.+vz_init**2./omega2**2.)
    phi1= numpy.arctan2(-vz_init/omega1,z_init)
    phi2= numpy.arctan2(-vz_init/omega2,z_init)
    return -A2*omega2*numpy.sin(omega2*t+phi2)+A1*omega1*numpy.sin(omega1*t+phi1)

##### run force of change for omega from density and v^2

def force_of_change_omega_densv2m(w_m2m,zsun_m2m,omega_m2m,
                          z_m2m,vz_m2m,z_prev,vz_prev,
                          step,eps_omega,eps,eps_vel,mu,w_prior,
                          z_obs,dens_obs,dens_obs_noise,v2m_obs,v2m_obs_noise,
                          h_m2m=0.02,delta_omega=0.3,
                          delta_m2m=None,deltav2m_m2m=None):
    # Compute the force of change by directly finite difference of the objective function
    # REAL HACK FOLLOWS!
    fcw, delta_m2m_do= force_of_change_weights(w_out,zsun_m2m,
                                               z_m2m+zdiff(z_prev,vz_prev,omega_m2m,omega_m2m+delta_omega,step),
                                               vz_m2m+vzdiff(z_prev,vz_prev,omega_m2m,omega_m2m+delta_omega,step),
                                               eps,mu,w_init,
                                               z_obs,dens_obs,dens_obs_noise,
                                               h_m2m=h_m2m,
                                               delta_m2m=None)
    fcwv2, deltav2m_m2m_do= force_of_change_weights_v2m(w_out,zsun_m2m,
                                                      z_m2m+zdiff(z_prev,vz_prev,omega_m2m,omega_m2m+delta_omega,step),
                                                      vz_m2m+vzdiff(z_prev,vz_prev,omega_m2m,omega_m2m+delta_omega,step),
                                                      eps_vel,mu,w_init,
                                                      z_obs,v2m_obs,v2m_obs_noise,
                                                      h_m2m=h_m2m,
                                                      deltav2m_m2m=None)

    return -2.*eps_omega\
        *numpy.sum(delta_m2m*(delta_m2m_do-delta_m2m)+deltav2m_m2m*(deltav2m_m2m_do-deltav2m_m2m))/delta_omega

##### run M2M fitting Omega with density and v^2 constraints.

def run_m2m_weights_omega_densv2m(w_init,A_init,phi_init,
                          omega_m2m,zsun_m2m,
                          z_obs,dens_obs,dens_obs_noise,v2m_obs,v2m_obs_noise,
                          step=0.001,nstep=1000,skipomega=10,
                          eps=0.1,eps_vel=0.1,eps_omega=0.001,mu=1.,
                          h_m2m=0.02,delta_omega=0.3,
                          smooth=None,
                          output_wevolution=False):
    """
    NAME:
       run_m2m_weights_omega
    PURPOSE:
       Run M2M on the harmonic-oscillator data to optimize the weights of the orbits and the potential parameter
       omega using both density and velocity data
    INPUT:
       w_init - initial weights [N]
       A_init - initial zmax [N]
       phi_init - initial angle (rad) [N]
       omega_m2m - potential parameter omega
       zsun_m2m - Sun's height above the plane [N]
       z_obs - heights at which the density observations are made
       dens_obs - observed densities
       dens_obs_noise - noise in the observed densities
       v2m_obs - observed velocity-squareds
       v2m_obs_noise - noise in the observed velocity-squareds
       step= stepsize of orbit integration
       nstep= number of steps to integrate the orbits for
       skipomega= only update omega every skipomega steps
       eps= M2M epsilon parameter
       eps_vel= M2M epsilon parameter for the velocity change
       eps_omega= M2M epsilon parameter for omega force of change
       mu= M2M entropy parameter mu
       h_m2m= kernel size parameter for computing the observables
       delta_omega= (0.3) difference in omega to use to compute derivative of objective function wrt omega
       smooth= smoothing parameter alpha (None for no smoothing)
       nodens= if True, don't fit the density data (also doesn't include entropy term)
       output_wevolution= if set to an integer, return the time evolution of this many randomly selected weights
    OUTPUT:
       ((w_out,omega_out,z_out,vz_out),Q_out,[wevol,rndindx]) - 
               ((output weights [N],output omega [nstep],final z [N], final vz [N]),
                objective function as a function of time,
                [weight evolution for randomly selected weights,index of random weights])
    HISTORY:
       2016-12-07 - Written - Bovy (UofT/CCA)
       2016-12-09 - Added output_wevolution - Bovy (UofT/CCA)
       2016-12-16 - Added explicit noise in the observed densities and densities x velocity-squared - Bovy (UofT/CCA)
       2017-02-17 - Modified to use v2m_obs - Kawata (MSSL/UCL)
    """
    w_out= copy.deepcopy(w_init)
    Q_out= []
    omega_out= []
    if output_wevolution:
        rndindx= numpy.random.permutation(len(w_out))[:output_wevolution]
        wevol= numpy.zeros((output_wevolution,nstep))
    if not smooth is None:
        # Smooth the constraints
        phi_now= phi_init
        z_m2m= A_init*numpy.cos(phi_now)
        vz_m2m= -A_init*omega_m2m*numpy.sin(phi_now) # unnecessary
        dens_init= compute_dens(z_m2m,zsun_m2m,z_obs,h_m2m,w=w_init)
        delta_m2m= (dens_init-dens_obs)/dens_obs_noise
        v2m_init= compute_v2m(z_m2m,vz_m2m,zsun_m2m,z_obs,h_m2m,w=w_init)
        deltav2m_m2m= (v2m_init-v2m_obs)/v2m_obs_noise
    else:
        delta_m2m= None
        deltav2m_m2m= None
    z_m2m= A_init*numpy.cos(phi_init)
    vz_m2m= -A_init*omega_m2m*numpy.sin(phi_init)
    # Initialize
    z_m2m_prev= A_init*numpy.cos(phi_init)
    vz_m2m_prev= -A_init*omega_m2m*numpy.sin(phi_init)
    ocounter= 0
    for ii in range(nstep):
        # Compute current (z,vz)
        phi_prev= numpy.arctan2(-vz_m2m,z_m2m*omega_m2m)
        A_now= numpy.sqrt(z_m2m**2.+vz_m2m**2./omega_m2m**2.)
        phi_now= phi_prev+omega_m2m*step
        z_m2m= A_now*numpy.cos(phi_now)
        vz_m2m= -A_now*omega_m2m*numpy.sin(phi_now)
        fcw, delta_m2m_new= force_of_change_weights(w_out,zsun_m2m,
                                                    z_m2m,vz_m2m,
                                                    eps,mu,w_init,
                                                    z_obs,dens_obs,dens_obs_noise,
                                                    h_m2m=h_m2m,
                                                    delta_m2m=delta_m2m)
        fcwv2, deltav2m_m2m_new= force_of_change_weights_v2m(w_out,zsun_m2m,
                                                           z_m2m,vz_m2m,
                                                           eps_vel,mu,w_init,
                                                           z_obs,v2m_obs,v2m_obs_noise,
                                                           h_m2m=h_m2m,
                                                           deltav2m_m2m=deltav2m_m2m)
        w_out+= step*fcw+step*fcwv2
        w_out/= numpy.sum(w_out)/len(A_init)
        w_out[w_out < 0.]= 10.**-16. 
        Q_out.append(delta_m2m_new**2.+deltav2m_m2m_new**2.)
        # Increment smoothing
        if not smooth is None:
            delta_m2m= step*smooth*(delta_m2m_new-delta_m2m)
            deltav2m_m2m= step*smooth*(deltav2m_m2m_new-deltav2m_m2m)
        if output_wevolution:
            wevol[:,ii]= w_out[rndindx]
        # Update omega in this step?
        ocounter+= 1
        if ocounter != skipomega:
            omega_out.append(omega_m2m)
            continue
        ocounter= 0
        if smooth is None:
            fcomega= force_of_change_omega_densv2m(w_out,zsun_m2m,omega_m2m,
                                           z_m2m,vz_m2m,z_m2m_prev,vz_m2m_prev,
                                           step*skipomega,eps_omega,eps,eps_vel,mu,w_init,
                                           z_obs,dens_obs,dens_obs_noise,v2m_obs,v2m_obs_noise,
                                           h_m2m=h_m2m,delta_omega=delta_omega,
                                           delta_m2m=delta_m2m_new,deltav2m_m2m=deltav2m_m2m_new)
        else:
            fcomega= force_of_change_omega_densv2m(w_out,zsun_m2m,omega_m2m,
                                           z_m2m,vz_m2m,z_m2m_prev,vz_m2m_prev,
                                           step*skipomega,eps_omega,eps,eps_vel,mu,w_init,
                                           z_obs,dens_obs,dens_obs_noise,v2m_obs,v2m_obs_noise,
                                           h_m2m=h_m2m,delta_omega=delta_omega,
                                           delta_m2m=delta_m2m_new,deltav2m_m2m=deltav2m_m2m)
        domega= step*skipomega*fcomega
        maxdomega= delta_omega/10.
        if numpy.fabs(domega) > maxdomega: domega= maxdomega*numpy.sign(domega)
        omega_m2m+= domega
        z_m2m_prev= copy.copy(z_m2m)
        vz_m2m_prev= copy.copy(vz_m2m)
        omega_out.append(omega_m2m)
    out= ((w_out,numpy.array(omega_out),z_m2m,vz_m2m),numpy.array(Q_out))
    if output_wevolution:
        out= out+(wevol,rndindx,)
    return out
