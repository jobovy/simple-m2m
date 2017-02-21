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

def Rn_to_simplex(y):
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
    return x
