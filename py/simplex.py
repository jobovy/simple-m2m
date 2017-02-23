# simplex.py: functions to deal with the simplex, following Betancourt 2012 
#             (and a little bit of inspiration from the stan user's manual)
import numpy
def _logit(x):
    return numpy.log(x/(1.-x))
def _inv_logit(x):
    return 1./(1.+numpy.exp(-x))

def simplex_to_Rn(x):
    """
    NAME:
       simplex_to_Rn
    PURPOSE:
       Transform the simplex x to R^n
    INPUT:
       x \in simplex
    OUTPUT:
       y \in R^n
    HISTORY:
       2017-02-21 - Written - Bovy (UofT/CCA)
    """
    z= numpy.zeros(len(x)-1)
    for ii in range(len(x)-1):
        z[ii]= (1.-x[ii]/numpy.prod(z[:ii]))
    return _logit(z)-_logit(numpy.arange(1,len(z)+1)[::-1]/\
                             numpy.arange(2,len(z)+2).astype('float')[::-1])

def Rn_to_simplex(y,_retz=False):
    """
    NAME:
       Rn_to_simplex
    PURPOSE:
       Transform R^n to the simplex using a stick-breaking algorithm
    INPUT:
       y \in R^n
    OUTPUT:
       x \in simplex
    HISTORY:
       2017-02-21 - Written - Bovy (UofT/CCA)
    """
    z= _inv_logit(y+
                  _logit(numpy.arange(1,len(y)+1)[::-1]/\
                             numpy.arange(2,len(y)+2).astype('float')[::-1]))
    tmp1= numpy.empty(len(y)+1)
    tmp1[0]= 0.
    tmp1[1:]= z
    tmp2= numpy.roll(1.-tmp1,-1)
    tmp1[0]= 1.
    x= numpy.cumprod(tmp1)*tmp2
    if _retz:
        return (x,z)
    else:
        return x

def Rn_to_simplex_jac(y,det=False,dlogdet=False):
    """
    NAME:
       Rn_to_simplex_jac
    PURPOSE:
       Compute the Jacobian of the transformation R^n --> simplex
    INPUT:
       y \in R^n
       det= (False) if True, also return the determinant of the Jacobian
       dlogdet= (False) if True, also return the derivative of the ln of the determinant
    OUTPUT:
       (Jacobian,determinant, d ln determinant/ d y)
    HISTORY:
       2017-02-21 - Written - Bovy (UofT/CCA)
    """
    w, x= Rn_to_simplex(y,_retz=True)
    out= numpy.zeros((len(y)+1,len(y)))
    indx= numpy.tril_indices(len(y)+1,k=0,m=len(y))
    out[indx[0],indx[1]]= w[indx[0]]*(1.-x[indx[1]])
    indx= numpy.diag_indices(len(y))
    out[indx[0],indx[1]]= -w[indx[0]]*x[indx[1]]
    out= (out,)
    if det:
        out= out+(numpy.prod(w[:-1]*x),)
    if dlogdet:
        out= out+((len(x)-numpy.arange(len(x)))*(1.-x)-x,)
    if len(out) == 1: return out[0]
    else: return out

def simplex_to_Rn_derivs(dx,jac):
    """
    NAME:
       simplex_to_Rn_derivs
    PURPOSE:
       transform derivatives wrt the simplex (d / d x) to derivatives with respect to the R^N-1 variables
    INPUT:
       dx - derivatives
       jac - pre-computed jacobian
    OUTPUT:
       derivatives wrt y
    HISTORY:
       2017-02-22 - Written - Bovy (UofT/CCA)
    """
    return numpy.dot(jac.T,dx)

def simplex_to_Rn_derivs_fast(y,dx):
    """
    NAME:
       simplex_to_Rn_derivs_fast
    PURPOSE:
       transform derivatives wrt the simplex (d / d x) to derivatives with respect to the R^N-1 variables in a fast manner
    INPUT:
       y \in R^n
       dx - derivatives
    OUTPUT:
       derivatives wrt y
    HISTORY:
       2017-02-23 - Written - Bovy (UofT/CCA)
    """
    w, x= Rn_to_simplex(y,_retz=True)
    out= numpy.cumsum((w*dx)[::-1])[::-1]
    out= out[1:]*(1.-x)-x*(w*dx)[:-1]
    return out
