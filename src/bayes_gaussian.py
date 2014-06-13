import numpy as np
import statsmodels.api as sm
from numpy import sqrt, log, exp, pi, dot
from scipy.interpolate import interp1d
from scipy import ndimage
from scipy import stats


def gaussian(x,mu,sigma):
    """
    Returns a normalised gaussian with mean mu and half-width sigma, at x
    x can be a scalar or a numpy array
    """
    return (1/sqrt(2*pi*sigma*sigma))*exp(-(x-mu)*(x-mu)/(2*sigma*sigma))

def gaussian_dD(x,mu,sigma):
    """
    Returns the general multivariate normal probability density in d-dimensions.
    x is a 1-D numpy array of length d (a single point in d-space)
    """
    d = len(x)

    norm = 1 / ((2 * pi)**(d/2.0)*sqrt(np.linalg.det(sigma)))
    s_inverse = np.linalg.inv(sigma)
    
    e = dot( (x-mu).T, dot(s_inverse,(x-mu)) )

    p = norm * exp( -0.5 * e)

    return p

def gaussian_dD_ranges(xranges,mu,sigma) :
    """
    Given a list of xranges,  
    Returns an output array of the same shape of the input array xarray
    with the multivariate normal probability density
    """
    d=len(mu)
    xlens = [len(xranges[i]) for i in xrange(d)]
    xshape = tuple(xlens)
    npts=np.prod(xlens)
    
    p = np.empty(npts)
    for i in xrange(npts) : 
        indexes = np.unravel_index(i,xshape)
        x=np.array( [ xranges[j][indexes[j]] for j in xrange(d)])
        p[i] = gaussian_dD(x,mu,sigma)

    p_dd = np.reshape(p,xshape)
    return p_dd


def transform_data_to_uniform_1D(feature_vector):
    """
    Takes not-necessarily any distributed data and transforms it
    to a uniform distribution using the cumulative distribution function
    """
    import matplotlib.pyplot as plt
    x=feature_vector

    n=len(x)
    ecdf = sm.distributions.ECDF(x)
    new_x=ecdf(feature_vector)
        
    return new_x

def transform_data_to_gaussian_1D(feature_vector):
    """
    Takes not-necessarily any distributed data and transforms it
    to a gaussian distribution using the box-cox transform
    """
    import matplotlib.pyplot as plt
    x=feature_vector

    n=len(x)
    new_x,l=stats.boxcox(feature_vector)
        
    return new_x

def data_to_gaussian_1D(feature_vector):
    """
    Gets parameters for best fit gaussian to the distribution of values 
    in the feauture vector.
    Returns mu, sigma  for the data distribution.
    """

    # maximum likelihood estimator for the mean of a gaussian distribution
    # is simply the mean of the distribution 
    mu=np.mean(feature_vector)

    # maximum likelihood estimator for sigma of 1-D gaussian is the
    # square-root of the variance of the distribution
    sigma=sqrt(np.var(feature_vector))

    return mu, sigma

def data_to_gaussian_dD(feature_matrix):
    """
    feature_matrix has d columns (features) and n rows (samples)
    Returns maximum likelihood estimators for vector mu and variance/covariance matrix Sigma
    """
    x=feature_matrix
    (n_samples, n_features) = x.shape
    
    mu = np.mean(feature_matrix, axis=0)
    
    Sigma=dot((x-mu).T,(x-mu)) / n_samples

    return mu, Sigma

def sigma_to_corr(sigma):
    """
    Calculates a correlation matrix from a covariance matrix
    """
    (d,dd)=sigma.shape
    if dd != d :
        raise UserWarning('Covariance matrix is not square.')

    corr=np.zeros(sigma.shape)
    for i in range(d):
        for j in range(d):
            corr[i,j]=sigma[i,j]/(sqrt(sigma[i,i]*sigma[j,j]))

    return corr

def data_to_eigen_projection_dD(feature_matrix,M=None):
    """
    Reprojects data onto the eigenvectors of the variance-covariance matrix.
    Use only the eigenvectors corresponding to the M largest eigenvalues
    If M = None, then use all the eigenvectors
    """

    (n,d) = feature_matrix.shape
    if M == None :
        M = d

    # get mu and Sigma for original data
    mu, Sigma = data_to_gaussian_dD(feature_matrix)

    # get eigenvalues and eigenvectors
    w, v = np.linalg.eig(Sigma)

    # get eigenvalues in increascing order
    i_sorted=np.argsort(w)[::-1]
    w_sorted=w[i_sorted]
    v_sorted=v[:,i_sorted]

    # do projection
    new_matrix=np.empty((n,M))
    for i in xrange(M) :
        ev_index=i_sorted[i]
        new_matrix[:,i] = dot((feature_matrix-mu),v[:,ev_index])

    # return the sorted eigenvalues and the new feature matrix
    return w_sorted, v_sorted, new_matrix 



def verify_dict_prior(dict_prior):
    """
    Raises an exception if the values of dict_prior do not sum to unity.
    """
    p_sum=0.0
    for val in dict_prior.values():
        p_sum=p_sum+val
    if np.abs(p_sum-1.0) > 1e-5 :
        raise UserWarning('Priors do not sum to unity.')

def verify_dict_conditionnal(dict_c):
    """
    Raises an exception if length of probability density
    vectors for the different classes are not the same.
    """
    keys = dict_c.keys()
    # first length
    pd_shape = dict_c[keys[0]].shape
    for val in dict_c.values():
        if val.shape != pd_shape :
            raise UserWarning('Conditionnal probabilities of different shapes')


def unconditionnal(dict_c, dict_prior):
    """
    Returns unconditionnal (data) probability density, given
    conditional probability densities and priors : 
    p(x) = p(x|C1)*P(C1) + p(x|C2)*P(C2)
    """
    
    # Test priors are ok
    verify_dict_prior(dict_prior)

    # Test conditionnals are ok
    verify_dict_conditionnal(dict_c)

    keys=dict_prior.keys()
    try:
        p_shape=dict_c[keys[0]].shape
        x = np.zeros(p_shape)
        for key in keys :
            x = x + dict_c[key] * dict_prior[key]
    except KeyError:
        raise UserWarning('Class name not found : %s'%key)

    return x

def posterior(dict_c, dict_prior):
    """
    Calculates the posterior probabilities given conditionnal
    probabilites and priors. Normalises by the unconditionnal
    (data) probability.
    """

    post = {}

    uncond = unconditionnal(dict_c, dict_prior) + 1e-12

    try:
        for key in dict_prior.keys():
            post[key] = dict_c[key] * dict_prior[key] / uncond
    except KeyError:
        raise UserWarning('Class name not found : %s'%key)

    return post

def predict_1D(post,feature_x,feature_values):
    """
    Returns a dictionnary of probabilites for each
    class in the input dictionnary post.
    """

    prediction={}
    for key in post.keys():
        
        # set up the interpolator
        f = interp1d(feature_x, post[key], kind='cubic')
        # do interpolation for given values
        prediction[key] = f(feature_values)

    return prediction

def predict_dD(post,xranges,x_values):
    """
    f_values are column vectors (d columns of n samples)
    """
    # get number of samples and dimensions
    (n,d) = x_values.shape
    xlens = np.array([len(xranges[i]) for i in xrange(d)])
    mins  = np.array([np.min(xranges[i]) for i in xrange(d)])
    maxs  = np.array([np.max(xranges[i]) for i in xrange(d)])

    # map values to index-coordinates
    coords = np.array([ (x_values[j]-mins) * (xlens - 1) / (maxs - mins) for j in xrange(n)])
    # transpose for map_coordinates
    coords=coords.T
    # do prediction

    pred={}
    for key in post.keys():
        pred[key]=ndimage.map_coordinates(post[key],coords,order=3,mode='nearest')

    return pred

