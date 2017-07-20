# wendym2m.py: M2M with wendy, a 1D N-body code
import copy
import numpy
import wendy
import hom2m

########################## SELF-GRAVITATING DISK TOOLS ########################
def sample_sech2(sigma,totmass,n=1):
    # compute zh based on sigma and totmass
    zh= sigma**2./totmass # twopiG = 1. in our units
    x= numpy.arctanh(2.*numpy.random.uniform(size=n)-1)*zh*2.
    v= numpy.random.normal(size=n)*sigma
    v-= numpy.mean(v) # stabilize
    m= numpy.ones_like(x)*totmass/n
    return (x,v,m)

############################### M2M FORCE-OF-CHANGE ###########################
# All defined here as the straight d constraint / d parameter (i.e., does *not*
# include things like eps, weight)

def force_of_change_weights(w_m2m,zsun_m2m,z_m2m,vz_m2m,
                            data_dicts,
                            prior,mu,w_prior,
                            h_m2m=0.02,
                            kernel=hom2m.epanechnikov_kernel,
                            delta_m2m=None):
    """Computes the force of change for all of the weights"""
    fcw= numpy.zeros_like(w_m2m)
    delta_m2m_new= []
    if delta_m2m is None: delta_m2m= [None for d in data_dicts]
    for ii,data_dict in enumerate(data_dicts):
        if data_dict['type'].lower() == 'dens':
            tfcw, tdelta_m2m_new=\
                hom2m.force_of_change_density_weights(\
                    numpy.sum(w_m2m[:,data_dict['pops']],axis=1),
                    zsun_m2m,z_m2m,vz_m2m,
                    data_dict['zobs'],data_dict['obs'],data_dict['unc'],
                    h_m2m=h_m2m,kernel=kernel,delta_m2m=delta_m2m[ii])
        elif data_dict['type'].lower() == 'v2':
            tfcw, tdelta_m2m_new=\
                hom2m.force_of_change_v2_weights(\
                    numpy.sum(w_m2m[:,data_dict['pops']],axis=1),
                    zsun_m2m,z_m2m,vz_m2m,
                    data_dict['zobs'],data_dict['obs'],data_dict['unc'],
                    h_m2m=h_m2m,kernel=kernel,deltav2_m2m=delta_m2m[ii])
        else:
            raise ValueError("'type' of measurement in data_dict not understood")
        fcw[:,data_dict['pops']]+= numpy.atleast_2d(tfcw).T
        delta_m2m_new.extend(tdelta_m2m_new)
    # Add prior

    fcw+= hom2m.force_of_change_prior_weights(w_m2m,mu,w_prior,prior)
    return (fcw,delta_m2m_new)

################################ M2M OPTIMIZATION #############################
def parse_data_dict(data_dicts):
    """
    NAME:
       parse_data_dict
    PURPOSE:
       parse the data_dict input to M2M routines
    INPUT:
       data_dicts - list of data_dicts
    OUTPUT:
       cleaned-up version of data_dicts
    HISTORY:
       2017-07-20 - Written - Bovy (UofT)
    """
    for data_dict in data_dicts:
        if isinstance(data_dict['pops'],int):
            data_dict['pops']= [data_dict['pops']]
    return data_dict

def fit_m2m(w_init,z_init,vz_init,
            omega_m2m,zsun_m2m,
            data_dicts,
            step=0.001,nstep=1000,
            eps=0.1,mu=1.,prior='entropy',w_prior=None,
            kernel=hom2m.epanechnikov_kernel,
            kernel_deriv=hom2m.epanechnikov_kernel_deriv,
            h_m2m=0.02,
            npop=1,
            smooth=None,st96smooth=False,
            output_wevolution=False,
            fit_zsun=False,fit_omega=False,
            skipomega=10,delta_omega=0.3):
    """
    NAME:
       fit_m2m
    PURPOSE:
       Run M2M optimization for wendy M2M
    INPUT:
       w_init - initial weights [N] or [N,npop]
       z_init - initial z [N]
       vz_init - initial vz (rad) [N]
       omega_m2m - potential parameter omega
       zsun_m2m - Sun's height above the plane [N]
       data_dicts - list of dictionaries that hold the data, these are described in more detail below
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
       npop= (1) number of theoretical populations
       smooth= smoothing parameter alpha (None for no smoothing)
       st96smooth= (False) if True, smooth the constraints (Syer & Tremaine 1996), if False, smooth the objective function and its derivative (Dehnen 2000)
       output_wevolution= if set to an integer, return the time evolution of this many randomly selected weights
    DATA DICTIONARIES:
       The data dictionaries have the following form:
           'type': type of measurement: 'dens', 'v2'
           'pops': the theoretical populations included in this measurement; 
                   single number or list
           'zobs': vertical height of the observation
           'zrange': width of vertical bin relative to some fiducial value (used to scale h_m2m, which should therefore be appropriate for the fiducial value)
           'obs': the actual observation
           'unc': the uncertainty in the observation
       of these, zobs, obs, and unc can be arrays for mulitple measurements
    OUTPUT:
       (w_out,[zsun_out, [omega_out]],z_m2m,vz_m2m,Q_out,[wevol,rndindx]) - 
              (output weights [N],
              [Solar offset [nstep] optional],
              [omega [nstep] optional when fit_omega],
              z_m2m [N] final z,
              vz_m2m [N] final vz,
              objective function as a function of time [nstep],
              [weight evolution for randomly selected weights,index of random weights])
    HISTORY:
       2017-07-20 - Started from hom2m.fit_m2m - Bovy (UofT)
    """
    if len(w_init.shape) == 1:
        w_out= numpy.empty((len(w_init),npop))
        w_out[:,:]= numpy.tile(copy.deepcopy(w_init),(npop,1)).T
    else:
        w_out= copy.deepcopy(w_init)
    zsun_out= numpy.empty(nstep)
    omega_out= numpy.empty(nstep)
    if w_prior is None:
        w_prior= w_out
    # Parse data_dict
    data_dict= parse_data_dict(data_dicts)
    # Parse eps
    if isinstance(eps,float):
        eps= [eps]
        if fit_zsun: eps.append(eps[0])
        if fit_omega: eps.append(eps[0])
    Q_out= []
    if output_wevolution:
        rndindx= numpy.random.permutation(len(w_out))[:output_wevolution]
        wevol= numpy.zeros((output_wevolution,npop,nstep))
    # Compute force of change for first iteration
    fcw, delta_m2m_new= \
        force_of_change_weights(w_out,zsun_m2m,z_init,vz_init,
                                data_dicts,prior,mu,w_prior,
                                h_m2m=h_m2m,kernel=kernel)
    fcw*= w_out
    fcz= 0.
    if fit_zsun:
        fcz= force_of_change_zsun(w_init,zsun_m2m,z_init,vz_init,
                                  z_obs,dens_obs_noise,delta_m2m_new,
                                  densv2_obs_noise,deltav2_m2m_new,
                                  kernel=kernel,kernel_deriv=kernel_deriv,
                                  h_m2m=h_m2m,use_v2=use_v2)
    if not smooth is None:
        delta_m2m= delta_m2m_new
    else:
        delta_m2m= [None for d in data_dicts]
    if not smooth is None and not st96smooth:
        Q= [d**2 for d in delta_m2m**2.]
    # setup skipomega omega counter and prev. (z,vz) for F(omega)
    #ocounter= skipomega-1 # Causes F(omega) to be computed in the 1st step
    #z_prev, vz_prev= Aphi_to_zvz(A_init,phi_init-skipomega*step*omega_m2m,
    #                             omega_m2m) #Rewind for first step
    z_m2m, vz_m2m= z_init, vz_init
    for ii in range(nstep):
        # Update weights first
        if True:
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
            Q_out.append([d**2. for d in delta_m2m])
        elif not smooth is None:
            Q_out.append(copy.deepcopy(Q))
        else:
            Q_out.append([d**2. for d in delta_m2m_new])
        # Then update the dynamics
        mass= numpy.sum(w_out,axis=1)
        # (temporary?) way to deal with small masses
        relevant_particles_index= mass > (numpy.median(mass[mass > 10.**-9.])*10.**-6.)
        if numpy.any(mass[relevant_particles_index] < (10.**-8.*numpy.median(mass[relevant_particles_index]))):
            print(numpy.sum(mass[relevant_particles_index] < (10.**-8.*numpy.median(mass[relevant_particles_index]))))
        g= wendy.nbody(z_m2m[relevant_particles_index],
                       vz_m2m[relevant_particles_index],
                       mass[relevant_particles_index],
                       step,maxcoll=10000000)
        tz_m2m, tvz_m2m= next(g)
        z_m2m[relevant_particles_index]= tz_m2m
        vz_m2m[relevant_particles_index]= tvz_m2m
        z_m2m-= numpy.sum(mass*z_m2m)/numpy.sum(mass)
        vz_m2m-= numpy.sum(mass*vz_m2m)/numpy.sum(mass)
        # Compute force of change
        if smooth is None or not st96smooth:
            # Turn these off
            tdelta_m2m= None
        else:
            tdelta_m2m= delta_m2m
        fcw_new, delta_m2m_new= \
            force_of_change_weights(w_out,zsun_m2m,z_m2m,vz_m2m,
                                    data_dicts,prior,mu,w_prior,
                                    h_m2m=h_m2m,kernel=kernel,
                                    delta_m2m=tdelta_m2m)
        fcw_new*= w_out
        if fit_zsun:
            if smooth is None or not st96smooth:
                tdelta_m2m= delta_m2m_new
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
            delta_m2m= [d+step*smooth*(dn-d) 
                        for d,dn in zip(delta_m2m,delta_m2m_new)]
            fcw= fcw_new
            if fit_zsun: fcz= fcz_new
            if fit_omega and ocounter == skipomega: fco= fco_new
        elif not smooth is None:
            Q_new= [d**2. for d in delta_m2m_new]
            Q= [q+step*smooth*(qn-q) for q,qn in zip(Q,Q_new)]
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
            wevol[:,:,ii]= w_out[rndindx]
    out= (w_out,)
    if fit_zsun: out= out+(zsun_out,)
    if fit_omega:
        out= out+(omega_out,)
    out= out+(z_m2m,vz_m2m,)
    out= out+(numpy.array(Q_out),)
    if output_wevolution:
        out= out+(wevol,rndindx,)
    return out


