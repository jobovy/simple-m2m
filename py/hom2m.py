
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

### compute density at z_obs

def compute_dens(z,zsun,z_obs,h_obs,w=None):
    if w is None: w= numpy.ones_like(z)
    dens= numpy.zeros_like(z_obs)
    for jj,zo in enumerate(z_obs):
        dens[jj]= numpy.sum(w*kernel(numpy.fabs(zo-z+zsun),h_obs))/len(z)
    return dens

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
       h_m2m= kernel size parameter for computing the observables, this should be same as h_obs
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


### compute_densv2

def compute_densv2(z,vz,zsun,z_obs,h_obs,w=None):
    if w is None: w= numpy.ones_like(z)
    densv2= numpy.zeros_like(z_obs)
    for jj,zo in enumerate(z_obs):
        densv2[jj]= numpy.sum(w*kernel(numpy.fabs(zo-z+zsun),h_obs)*vz**2.)/len(z)
    return densv2

### force_of_change_weights_densv2

def force_of_change_weights_densv2(w_m2m,zsun_m2m,z_m2m,vz_m2m,
                               eps,mu,w_prior,
                               z_obs,densv2_obs,densv2_obs_noise,
                               h_m2m,
                               deltav2_m2m=None):
    """Computes the force of change for all of the weights due to the velocity constraint"""
    deltav2_m2m_new= numpy.zeros_like(z_obs)
    Wij= numpy.zeros((len(z_obs),len(z_m2m)))
    for jj,zo in enumerate(z_obs):
        Wij[jj]= kernel(numpy.fabs(zo-z_m2m+zsun_m2m),h_m2m)
        deltav2_m2m_new[jj]= (numpy.sum(w_m2m*Wij[jj]*vz_m2m**2.)/len(z_m2m)-densv2_obs[jj])/densv2_obs_noise[jj]
    if deltav2_m2m is None: deltav2_m2m= deltav2_m2m_new
    return (-eps*w_m2m*numpy.sum(numpy.tile(deltav2_m2m/densv2_obs_noise,(len(z_m2m),1)).T*Wij,axis=0)*vz_m2m**2.,
            deltav2_m2m_new)

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
       2017-02-07 - modified from run_m2m_weights_zsun
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

