
import math
import numpy
import copy

# define kernel
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
if True:
    kernel= epanechnikov_kernel
    kernel_deriv= epanechnikov_kernel_deriv
else:
    kernel= sph_kernel
    kernel_deriv= sph_kernel_deriv

### M2M force of change definitions

def force_of_change_weights(w_m2m,zsun_m2m,z_m2m,vz_m2m,
                            eps,mu,w_prior,
                            z_obs,dens_obs,dens_obs_noise,
                            h_m2m=0.02,
                            delta_m2m=None):
    """Computes the force of change for all of the weights"""
    delta_m2m_new= numpy.zeros_like(z_obs)
    Wij= numpy.zeros((len(z_obs),len(z_m2m)))
    for jj,zo in enumerate(z_obs):
        Wij[jj]= kernel(numpy.fabs(zo-z_m2m+zsun_m2m),h_m2m)
        delta_m2m_new[jj]= (numpy.sum(w_m2m*Wij[jj])/len(z_m2m)-dens_obs[jj])/dens_obs_noise[jj]
    if delta_m2m is None: delta_m2m= delta_m2m_new
    return (-eps*w_m2m*(numpy.sum(numpy.tile(delta_m2m/dens_obs_noise,(len(z_m2m),1)).T*Wij,axis=0)\
                        +mu*(numpy.log(w_m2m/w_prior)+1.)),delta_m2m_new)

### M2M cycle definition

def run_m2m_weights(w_init,A_init,phi_init,
                    omega_m2m,zsun_m2m,
                    z_obs,dens_obs,dens_obs_noise,
                    step=0.001,nstep=1000,
                    eps=0.1,mu=1.,
                    h_m2m=0.02,
                    smooth=None,
                    output_wevolution=False):
    """
    NAME:
       run_m2m_weights
    PURPOSE:
       Run M2M on the harmonic-oscillator data to optimize just the weights of the orbits
    INPUT:
       w_init - initial weights [N]
       A_init - initial zmax [N]
       phi_init - initial angle (rad) [N]
       omega_m2m - potential parameter omega
       zsun_m2m - Sun's height above the plane [N]
       z_obs - heights at which the density observations are made
       dens_obs - observed densities
       dens_obs_noise - noise in the observed densities
       step= stepsize of orbit integration
       nstep= number of steps to integrate the orbits for
       eps= M2M epsilon parameter
       mu= M2M entropy parameter mu
       h_m2m= kernel size parameter for computing the observables
       smooth= smoothing parameter alpha (None for no smoothing)
       output_wevolution= if set to an integer, return the time evolution of this many randomly selected weights
    OUTPUT:
       (w_out,Q_out,[wevol,rndindx]) - (output weights [N],objective function as a function of time,
                                       [weight evolution for randomly selected weights,index of random weights])
    HISTORY:
       2016-12-06 - Written - Bovy (UofT/CCA)
       2016-12-08 - Added output_wevolution - Bovy (UofT/CCA)
       2016-12-16 - Added explicit noise in the observed densities - Bovy (UofT/CCA)
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
        dens_init= compute_dens(z_m2m,zsun_m2m,z_obs,w=w_init)
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
                                                    z_obs,
                                                    dens_obs,dens_obs_noise,
                                                    h_m2m=h_m2m,
                                                    delta_m2m=delta_m2m)
        w_out+= step*fcw
        w_out/= numpy.sum(w_out)/len(A_init)
        w_out[w_out < 0.]= 10.**-16.
        if not smooth is None:
            Q_out.append(delta_m2m**2.)
        else:
            Q_out.append(delta_m2m_new**2.)
        # Increment smoothing
        if not smooth is None:
            delta_m2m+= step*smooth*(delta_m2m_new-delta_m2m)
        # Record random weights if requested
        if output_wevolution:
            wevol[:,ii]= w_out[rndindx]
    out= (w_out,numpy.array(Q_out))
    if output_wevolution:
        out= out+(wevol,rndindx,)
    return out

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
       A_init - initial zmax [N]
       phi_init - initial angle (rad) [N]
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
        dens_init= compute_dens(z_m2m,zsun_m2m,z_obs,w=w_init)
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

