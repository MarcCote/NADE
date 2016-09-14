# -*- coding: utf-8 -*-
"""Multivariate Normal and t distributions

Taken from statsmodel (github.com/statsmodels)

Created on Sat May 28 15:38:23 2011

@author: Josef Perktold


Examples
--------

Note, several parts of these examples are random and the numbers will not be
(exactly) the same.

>>> import numpy as np
>>> import statsmodels.sandbox.distributions.mv_normal as mvd
>>>
>>> from numpy.testing import assert_array_almost_equal
>>>
>>> cov3 = np.array([[ 1.  ,  0.5 ,  0.75],
...                    [ 0.5 ,  1.5 ,  0.6 ],
...                    [ 0.75,  0.6 ,  2.  ]])

>>> mu = np.array([-1, 0.0, 2.0])

multivariate normal distribution
--------------------------------

>>> mvn3 = mvd.MVNormal(mu, cov3)
>>> mvn3.rvs(size=3)
array([[-0.08559948, -1.0319881 ,  1.76073533],
       [ 0.30079522,  0.55859618,  4.16538667],
       [-1.36540091, -1.50152847,  3.87571161]])

>>> mvn3.std
array([ 1.        ,  1.22474487,  1.41421356])
>>> a = [0.0, 1.0, 1.5]
>>> mvn3.pdf(a)
0.013867410439318712
>>> mvn3.cdf(a)
0.31163181123730122
"""

import numpy as np

class MVElliptical(object):
    '''Base Class for multivariate elliptical distributions, normal and t

    contains common initialization, and some common methods
    subclass needs to implement at least rvs and logpdf methods

    '''
    #getting common things between normal and t distribution

    def __init__(self, mean, sigma, *args, **kwds):
        '''initialize instance

        Parameters
        ----------
        mean : array_like
            parameter mu (might be renamed), for symmetric distributions this
            is the mean
        sigma : array_like, 2d
            dispersion matrix, covariance matrix in normal distribution, but
            only proportional to covariance matrix in t distribution
        args : list
            distribution specific arguments, e.g. df for t distribution
        kwds : dict
            currently not used

        '''
        self.extra_args = []
        self.mean = np.asarray(mean)
        self.sigma = sigma = np.asarray(sigma)
        sigma = np.squeeze(sigma)
        self.nvars = nvars = len(mean)

        #in the following sigma is original, self.sigma is full matrix
        if sigma.shape == ():
            #iid
            self.sigma = np.eye(nvars) * sigma
            self.sigmainv = np.eye(nvars) / sigma
            self.cholsigmainv = np.eye(nvars) / np.sqrt(sigma)
        elif (sigma.ndim == 1) and (len(sigma) == nvars):
            #independent heteroscedastic
            self.sigma = np.diag(sigma)
            self.sigmainv = np.diag(1. / sigma)
            self.cholsigmainv = np.diag( 1. / np.sqrt(sigma))
        elif sigma.shape == (nvars, nvars): #python tuple comparison
            #general
            self.sigmainv = np.linalg.pinv(sigma)
            self.cholsigmainv = np.linalg.cholesky(self.sigmainv).T
        else:
            raise ValueError('sigma has invalid shape')

        #store logdetsigma for logpdf
        self.logdetsigma = np.log(np.linalg.det(self.sigma))

    def rvs(self, size=1):
        '''random variable

        Parameters
        ----------
        size : int or tuple
            the number and shape of random variables to draw.

        Returns
        -------
        rvs : ndarray
            the returned random variables with shape given by size and the
            dimension of the multivariate random vector as additional last
            dimension


        '''
        raise NotImplementedError

    def logpdf(self, x):
        '''logarithm of probability density function

        Parameters
        ----------
        x : array_like
            can be 1d or 2d, if 2d, then each row is taken as independent
            multivariate random vector

        Returns
        -------
        logpdf : float or array
            probability density value of each random vector


        this should be made to work with 2d x,
        with multivariate normal vector in each row and iid across rows
        doesn't work now because of dot in whiten

        '''


        raise NotImplementedError

    def cdf(self, x, **kwds):
        '''cumulative distribution function

        Parameters
        ----------
        x : array_like
            can be 1d or 2d, if 2d, then each row is taken as independent
            multivariate random vector
        kwds : dict
            contains options for the numerical calculation of the cdf

        Returns
        -------
        cdf : float or array
            probability density value of each random vector

        '''
        raise NotImplementedError


    def affine_transformed(self, shift, scale_matrix):
        '''affine transformation define in subclass because of distribution
        specific restrictions'''
        #implemented in subclass at least for now
        raise NotImplementedError

    def whiten(self, x):
        """
        whiten the data by linear transformation

        Parameters
        -----------
        x : array-like, 1d or 2d
            Data to be whitened, if 2d then each row contains an independent
            sample of the multivariate random vector

        Returns
        -------
        np.dot(x, self.cholsigmainv.T)

        Notes
        -----
        This only does rescaling, it doesn't subtract the mean, use standardize
        for this instead

        See Also
        --------
        standardize : subtract mean and rescale to standardized random variable.

        """
        x = np.asarray(x)
        return np.dot(x, self.cholsigmainv.T)

    def pdf(self, x):
        '''probability density function

        Parameters
        ----------
        x : array_like
            can be 1d or 2d, if 2d, then each row is taken as independent
            multivariate random vector

        Returns
        -------
        pdf : float or array
            probability density value of each random vector

        '''
        return np.exp(self.logpdf(x))

    def standardize(self, x):
        '''standardize the random variable, i.e. subtract mean and whiten

        Parameters
        -----------
        x : array-like, 1d or 2d
            Data to be whitened, if 2d then each row contains an independent
            sample of the multivariate random vector

        Returns
        -------
        np.dot(x - self.mean, self.cholsigmainv.T)

        Notes
        -----


        See Also
        --------
        whiten : rescale random variable, standardize without subtracting mean.


        '''
        return self.whiten(x - self.mean)

    def standardized(self):
        '''return new standardized MVNormal instance
        '''
        return self.affine_transformed(-self.mean, self.cholsigmainv)


    def normalize(self, x):
        '''normalize the random variable, i.e. subtract mean and rescale

        The distribution will have zero mean and sigma equal to correlation

        Parameters
        -----------
        x : array-like, 1d or 2d
            Data to be whitened, if 2d then each row contains an independent
            sample of the multivariate random vector

        Returns
        -------
        (x - self.mean)/std_sigma

        Notes
        -----


        See Also
        --------
        whiten : rescale random variable, standardize without subtracting mean.


        '''
        std_ = np.atleast_2d(self.std_sigma)
        return (x - self.mean)/std_ #/std_.T

    def normalized(self, demeaned=True):
        '''return a normalized distribution where sigma=corr

        if demeaned is True, then mean will be set to zero

        '''
        if demeaned:
            mean_new = np.zeros_like(self.mean)
        else:
            mean_new = self.mean / self.std_sigma
        sigma_new = self.corr
        args = [getattr(self, ea) for ea in self.extra_args]
        return self.__class__(mean_new, sigma_new, *args)

    def normalized2(self, demeaned=True):
        '''return a normalized distribution where sigma=corr



        second implementation for testing affine transformation
        '''
        if demeaned:
            shift = -self.mean
        else:
            shift = self.mean * (1. / self.std_sigma - 1.)
        return self.affine_transformed(shift, np.diag(1. / self.std_sigma))
        #the following "standardizes" cov instead
        #return self.affine_transformed(shift, self.cholsigmainv)

    @property
    def std(self):
        '''standard deviation, square root of diagonal elements of cov
        '''
        return np.sqrt(np.diag(self.cov))

    @property
    def std_sigma(self):
        '''standard deviation, square root of diagonal elements of sigma
        '''
        return np.sqrt(np.diag(self.sigma))

    @property
    def corr(self):
        '''correlation matrix'''
        return self.cov / np.outer(self.std, self.std)

    def marginal(self, indices):
        '''return marginal distribution for variables given by indices

        this should be correct for normal and t distribution

        Parameters
        ----------
        indices : array_like, int
            list of indices of variables in the marginal distribution

        Returns
        -------
        mvdist : instance
            new instance of the same multivariate distribution class that
            contains the marginal distribution of the variables given in
            indices

        '''
        indices = np.asarray(indices)
        mean_new = self.mean[indices]
        sigma_new = self.sigma[indices[:,None], indices]
        args = [getattr(self, ea) for ea in self.extra_args]
        return self.__class__(mean_new, sigma_new, *args)

class MVNormal(MVElliptical):
    '''Class for Multivariate Normal Distribution

    uses Cholesky decomposition of covariance matrix for the transformation
    of the data

    '''
    __name__ == 'Multivariate Normal Distribution'


    def rvs(self, size=1):
        '''random variable

        Parameters
        ----------
        size : int or tuple
            the number and shape of random variables to draw.

        Returns
        -------
        rvs : ndarray
            the returned random variables with shape given by size and the
            dimension of the multivariate random vector as additional last
            dimension

        Notes
        -----
        uses numpy.random.multivariate_normal directly

        '''
        return np.random.multivariate_normal(self.mean, self.sigma, size=size)

    def logpdf(self, x):
        '''logarithm of probability density function

        Parameters
        ----------
        x : array_like
            can be 1d or 2d, if 2d, then each row is taken as independent
            multivariate random vector

        Returns
        -------
        logpdf : float or array
            probability density value of each random vector


        this should be made to work with 2d x,
        with multivariate normal vector in each row and iid across rows
        doesn't work now because of dot in whiten

        '''
        x = np.asarray(x)
        x_whitened = self.whiten(x - self.mean)
        SSR = np.sum(x_whitened**2, -1)
        llf = -SSR
        llf -= self.nvars * np.log(2. * np.pi)
        llf -= self.logdetsigma
        llf *= 0.5
        return llf

#    def cdf(self, x, **kwds):
#        '''cumulative distribution function
#
#        Parameters
#        ----------
#        x : array_like
#            can be 1d or 2d, if 2d, then each row is taken as independent
#            multivariate random vector
#        kwds : dict
#            contains options for the numerical calculation of the cdf
#
#        Returns
#        -------
#        cdf : float or array
#            probability density value of each random vector
#
#        '''
#        #lower = -np.inf * np.ones_like(x)
#        #return mvstdnormcdf(lower, self.standardize(x), self.corr, **kwds)
#        return mvnormcdf(x, self.mean, self.cov, **kwds)

    @property
    def cov(self):
        '''covariance matrix'''
        return self.sigma

    def affine_transformed(self, shift, scale_matrix):
        '''return distribution of an affine transform

        for full rank scale_matrix only

        Parameters
        ----------
        shift : array_like
            shift of mean
        scale_matrix : array_like
            linear transformation matrix

        Returns
        -------
        mvt : instance of MVT
            instance of multivariate t distribution given by affine
            transformation


        Notes
        -----
        the affine transformation is defined by
        y = a + B x

        where a is shift,
        B is a scale matrix for the linear transformation

        Notes
        -----
        This should also work to select marginal distributions, but not
        tested for this case yet.

        currently only tested because it's called by standardized

        '''
        B = scale_matrix  #tmp variable
        mean_new = np.dot(B, self.mean) + shift
        sigma_new = np.dot(np.dot(B, self.sigma), B.T)
        return MVNormal(mean_new, sigma_new)

    def conditional(self, indices, values):
        '''return conditional distribution

        indices are the variables to keep, the complement is the conditioning
        set
        values are the values of the conditioning variables

        \bar{\mu} = \mu_1 + \Sigma_{12} \Sigma_{22}^{-1} \left( a - \mu_2 \right)

        and covariance matrix

        \overline{\Sigma} = \Sigma_{11} - \Sigma_{12} \Sigma_{22}^{-1} \Sigma_{21}.T

        Parameters
        ----------
        indices : array_like, int
            list of indices of variables in the marginal distribution
        given : array_like
            values of the conditioning variables

        Returns
        -------
        mvn : instance of MVNormal
            new instance of the MVNormal class that contains the conditional
            distribution of the variables given in indices for given
             values of the excluded variables.
        '''
        #indices need to be nd arrays for broadcasting
        keep = np.asarray(indices)
        given = np.asarray([i for i in range(self.nvars) if not i in keep])
        sigmakk = self.sigma[keep[:, None], keep]
        sigmagg = self.sigma[given[:, None], given]
        sigmakg = self.sigma[keep[:, None], given]
        sigmagk = self.sigma[given[:, None], keep]

        sigma_new = sigmakk - np.dot(sigmakg, np.linalg.solve(sigmagg, sigmagk))
        mean_new = self.mean[keep] + np.dot(sigmakg, np.linalg.solve(sigmagg, values-self.mean[given]))
        return MVNormal(mean_new, sigma_new)