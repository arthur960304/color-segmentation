import numpy as np

class SimpleGaussian(object):
    def __init__(self, mu, cov):
        self.mean = mu
        self.cov = cov
    
    def show_mean_cov(self):
        return [self.mean, self.cov]
    
    def predict(self, x):
        dim = x.shape[1]
        cov_inv = np.linalg.cholesky(np.linalg.inv(self.cov))
        exp_term = np.exp(-0.5*np.sum(np.square(np.dot(x-self.mean,cov_inv)), axis=1))
        coeff = np.sqrt(((2*np.pi)**dim)*(np.linalg.det(self.cov)))
        prob = exp_term / coeff

        return prob