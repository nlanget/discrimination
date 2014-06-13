import os, unittest
import numpy as np
from numpy.testing import assert_array_almost_equal
from bayes_gaussian import *
from PdF_io import *

def suite():
    suite = unittest.TestSuite()
    suite.addTest(BayesTests('test_read_data'))
    suite.addTest(BayesTests('test_priors'))
    suite.addTest(BayesTests('test_gaussian_normalisation'))
    suite.addTest(BayesTests('test_data_to_gaussian_1D'))
    suite.addTest(BayesTests('test_unconditionnal_1D'))
    suite.addTest(BayesTests('test_posterior_1D'))
    suite.addTest(BayesTests('test_predict_1D'))
    suite.addTest(BayesTests('test_data_to_gaussian_dD'))
    suite.addTest(BayesTests('test_gaussian_normalisation_dD'))
    suite.addTest(BayesTests('test_predict_dD'))
    suite.addTest(BayesTests('test_data_to_eigen_projection_dD'))
    
    return suite

class BayesTests(unittest.TestCase):

    def setUp(self):
        self.data=read_training_data()
        self.testdata=read_test_data()
        pass

    def test_read_data(self):
        data=self.data
        testdata=self.testdata
        self.assertEquals(len(data['KurtoEB']),len(data['RappMaxMeanEB']))
        self.assertEquals(len(data['KurtoVT']),len(data['RappMaxMeanVT']))

        self.assertEquals(len(testdata['KurtoVal']),len(testdata['RappMaxMeanVal']))
        self.assertEquals(len(testdata['KurtoVal']),len(testdata['OVPFID']))
        self.assertEquals(len(testdata['KurtoVal']),len(testdata['ValMetaHypo']))
        
    def test_priors(self):
        P_prior = get_EB_VT_priors()
        self.assertEquals(P_prior['EB']+P_prior['VT'],1)

    def test_gaussian_normalisation(self):
        from scipy.integrate import trapz
        mu=5
        sigma=2
        x=np.arange(-10,15,0.01)
        g=gaussian(x,mu,sigma)
        integral=trapz(g,x)
        self.assertAlmostEquals(integral,1.0,5)

    def test_data_to_gaussian_1D(self):
        from numpy.random import randn
        mu_test=5
        sigma_test=2
        x=sigma_test * randn(100000) + mu_test
        mu, sigma = data_to_gaussian_1D(x)
        self.assertAlmostEquals(mu,mu_test,1)
        self.assertAlmostEquals(sigma,sigma_test,1)

    def test_data_to_gaussian_dD(self):
        m=np.zeros((1000,3))+np.random.randn(1000,3)*0.1
        m[:,0]+=     np.random.randn(1000)*0.1
        m[:,1]+= 1 + np.random.randn(1000)*0.2
        m[:,2]+= 2 + np.random.randn(1000)*0.3

        mu,Sigma=data_to_gaussian_dD(m)
        self.assertAlmostEquals(mu[0],0,1)
        self.assertAlmostEquals(mu[1],1,1)
        self.assertAlmostEquals(mu[2],2,1)

        self.assertAlmostEquals(np.sqrt(Sigma[0,0]),0.1,1)
        self.assertAlmostEquals(np.sqrt(Sigma[1,1]),0.2,1)
        self.assertAlmostEquals(np.sqrt(Sigma[2,2]),0.3,1)

    def test_gaussian_normalisation_dD(self):
        #import matplotlib.pyplot as plt
        from scipy.integrate import trapz

        m=np.zeros((1000,2))
        m[:,0]+=     np.random.randn(1000)*0.1
        m[:,1]+= 1 + np.random.randn(1000)*0.2
        mu,Sigma=data_to_gaussian_dD(m)

        # make a full 2D grid of probability density function
        range_0=np.arange(-0.5,0.5,0.01)
        range_1=np.arange( 0.5,1.5,0.01)
        xranges=[range_0, range_1]
        p = gaussian_dD_ranges(xranges,mu,Sigma)

        # check that the integral of the probability density is one
        integral=trapz(trapz(p,dx=0.01,axis=0),dx=0.01,axis=0)
        self.assertAlmostEquals(integral, 1.0 ,1)

        #plt.contour(p)
        #plt.show()

    def test_data_to_eigen_projection_dD(self):
        #import matplotlib.pyplot as plt

        # set up a correlated dataset
        data=np.zeros((1000,2))
        data[:,0] = np.random.randn(1000)*3 + 1
        data[:,1] = 0.5*data[:,0] + 5 + np.random.randn(1000)*0.2

        # do projection
        e_values, e_vectors, new_data = data_to_eigen_projection_dD(data)

        # get mean and covariances of new datset
        mu, Sigma = data_to_gaussian_dD(new_data)

        # sanity checks
        self.assertAlmostEquals(mu[0],0.0)
        self.assertAlmostEquals(mu[1],0.0)
        self.assertAlmostEquals(Sigma[0,0],e_values[0])
        self.assertAlmostEquals(Sigma[1,1],e_values[1])

        #plt.scatter(data[:,0],data[:,1],color='b')
        #plt.scatter(new_data[:,0],new_data[:,1],color='r')
        #plt.show()


    def test_predict_dD(self):

        m=np.zeros((1000,2))
        m[:,0]+=     np.random.randn(1000)*0.1
        m[:,1]+= 1 + np.random.randn(1000)*0.2
        mu,Sigma=data_to_gaussian_dD(m)


        # make a full 2D grid of probability density function
        range_0=np.arange(-0.5,0.5,0.01)
        range_1=np.arange( 0.5,1.5,0.01)
        xranges=[range_0, range_1]
        p = gaussian_dD_ranges(xranges,mu,Sigma)
        post={'C1' : p}

        f_values=np.array([[0,1],[-0.1,1],[0,1.2]])
        pred = predict_dD(post,xranges,f_values)
        self.assertAlmostEquals(pred['C1'][0],p[50,50])
        self.assertAlmostEquals(pred['C1'][1],p[40,50])
        self.assertAlmostEquals(pred['C1'][2],p[50,70])
                

    def test_unconditionnal_1D(self):
        x=np.arange(-10, 15, 0.01)
        prior={'C1' : 0.40, 'C2' : 0.60}
        c_dict={'C1' : gaussian(x,0,4), 'C2' : gaussian(x,5,3)}
        u = unconditionnal(c_dict, prior)

        test_value = gaussian(4.5,0,4) * 0.40 + gaussian(4.5,5,3) * 0.60
        test_index = np.int((4.5+10)/0.01)

        self.assertAlmostEquals(u[test_index],test_value)

    def test_posterior_1D(self):
        x=np.arange(-10, 15, 0.01)
        prior={'C1' : 0.40, 'C2' : 0.60}

        # Two non-overlapping classes
        c_dict={'C1' : gaussian(x,-5,0.5), 'C2' : gaussian(x,5,0.5)}
        
        post = posterior(c_dict, prior)

        c1_index = np.int((-5+10)/0.01)
        c2_index = np.int((5+10)/0.01)
        self.assertAlmostEquals(post['C1'][c1_index],1.0)
        self.assertAlmostEquals(post['C2'][c2_index],1.0)

    def test_predict_1D(self):
        x=np.arange(-10, 15, 0.1)
        prior={'C1' : 0.40, 'C2' : 0.60}
        # Two non-overlapping classes
        c_dict={'C1' : gaussian(x,-5,0.5), 'C2' : gaussian(x,5,0.5)}
        post = posterior(c_dict, prior)
        prediction = predict_1D(post,x,np.array([-5,5]))

        self.assertAlmostEquals(prediction['C1'][0],1.0)
        self.assertAlmostEquals(prediction['C1'][1],0.0)
        self.assertAlmostEquals(prediction['C2'][0],0.0)
        self.assertAlmostEquals(prediction['C2'][1],1.0)

if __name__ == '__main__':

  import logging
  logging.basicConfig(level=logging.INFO, format='%(levelname)s : %(asctime)s : %(message)s')
 
  unittest.TextTestRunner(verbosity=2).run(suite())
 
