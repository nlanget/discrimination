import unittest
import numpy as np
import pandas as pd
from LR_functions import *
from plot_functions import *
from do_classification import generate_datasets

def suite():
  suite = unittest.TestSuite()
  #suite.addTest(unittest.makeSuite(BinaryClassTests_2f))
  #suite.addTest(unittest.makeSuite(BinaryClassTests_3f))
  #suite.addTest(unittest.makeSuite(MultiClassTests))
  #suite.addTest(unittest.makeSuite(Polynome))
  #suite.addTest(unittest.makeSuite(DictMatrix))
  #suite.addTest(unittest.makeSuite(Compare))
  suite.addTest(unittest.makeSuite(Synthetics))
  return suite


class BinaryClassTests_2f(unittest.TestCase):
  
  def setUp(self):

    self.verb = False

    self.x = {}
    self.x['x1'] = np.array([1,0.5,2,1.6,2.3,5,6,7,6.2,5.6])
    self.x['x2'] = np.array([4,2,3,1.5,3.2,5,4,3.5,6,4.5])
    self.x = pd.DataFrame(self.x)

    self.y = {}
    self.y[1] = np.array([0,0,0,0,0,1,1,1,1,1])
    self.y = pd.DataFrame(self.y)

    self.theta = {}
    #self.theta[1] = np.random.rand(self.x.shape[1]+1)
    self.theta[1] = np.zeros(self.x.shape[1]+1)

    self.lamb = 0

    self.theta = logistic_reg(self.x,self.y,self.theta,l=self.lamb,verbose=self.verb)

    self.x_new = {}
    self.x_new['x1'] = np.array([2,5])
    self.x_new['x2'] = np.array([2,4])
    self.x_new = pd.DataFrame(self.x_new)
    self.exp_class = np.array([0,1])

  def test_binary_classification(self):
    x_class = test_hyp(self.x_new,self.theta,verbose=self.verb)
    diff = x_class-self.exp_class
    print self.theta
    print x_class, self.exp_class
    self.assertFalse(diff.any())
    if self.verb:
      plt.show()

class BinaryClassTests_3f(unittest.TestCase):
  
  def setUp(self):

    self.verb = False

    self.x = {}
    self.x['x1'] = np.array([1,2,1.6,2.3,4.1,6,5.6,4.9])
    self.x['x2'] = np.array([1,0.5,0.8,1.5,5,3.2,4.6,4])
    self.x['x3'] = np.array([6,5.5,6.4,5.9,2.2,4.1,3.6,1.9])
    self.x = pd.DataFrame(self.x)

    self.y = {}
    self.y[1] = np.array([0,0,0,0,1,1,1,1])
    self.y = pd.DataFrame(self.y)

    self.theta = {}
    self.theta[1] = np.random.rand(self.x.shape[1]+1)

    self.lamb = 0

    self.theta = logistic_reg(self.x,self.y,self.theta,l=self.lamb,verbose=self.verb)

    self.x_new = {}
    self.x_new['x1'] = np.array([2,5])
    self.x_new['x2'] = np.array([1,6])
    self.x_new['x3'] = np.array([5,2])
    self.x_new = pd.DataFrame(self.x_new)

    self.exp_class = np.array([0,1])

  def test_binary_classification(self):
    x_class = test_hyp(self.x_new,self.theta,verbose=self.verb)
    diff = x_class-self.exp_class
    self.assertFalse(diff.any())


class MultiClassTests(unittest.TestCase):

  def setUp(self):

    self.verb = False

    self.x = {}
    self.x['x1'] = np.array([1,2.5,1.6,2.5,2,1.5,5.5,5,6])
    self.x['x2'] = np.array([4,3.5,4.5,1.2,2,1.5,3.5,4,5])
    self.x = pd.DataFrame(self.x) 
 
    self.y = {}
    self.y[1] = np.array([1,1,1,0,0,0,0,0,0])
    self.y[2] = np.array([0,0,0,1,1,1,0,0,0])
    self.y[3] = np.array([0,0,0,0,0,0,1,1,1])
    self.y = pd.DataFrame(self.y)

    self.theta = {}
    self.theta[1] = np.random.rand(self.x.shape[1]+1)
    self.theta[2] = np.random.rand(self.x.shape[1]+1)
    self.theta[3] = np.random.rand(self.x.shape[1]+1)


    self.theta = logistic_reg(self.x,self.y,self.theta,verbose=self.verb)

    self.x_new = {}
    self.x_new['x1'] = np.array([3,4.5,3])
    self.x_new['x2'] = np.array([2,3,4.5])
    self.x_new = pd.DataFrame(self.x_new)

    self.exp_class = np.array([1,2,0])


  def test_multiclass_classification(self):
    x_class = test_hyp(self.x_new,self.theta,verbose=self.verb)
    diff = x_class-self.exp_class
    self.assertFalse(diff.any())


class Polynome(unittest.TestCase):

  def setUp(self):

    self.x = {}
    self.x[1] = np.array([2,2,2,2,2])
    self.x[2] = np.array([1,3,1,3,1])
    self.x[3] = np.array([4,1,4,1,4])
    self.x[4] = np.array([5,5,5,6,6])
    self.x = pd.DataFrame(self.x)
    
  def test_nb_features(self):
    deg = 4
    n = self.x.shape[1]
    from math import factorial
    nb_feat_exp = factorial(n+deg)/(factorial(n)*factorial(deg))

    x_deg = poly(deg,self.x)
    self.assertEqual(x_deg.shape[1],nb_feat_exp-1)


class DictMatrix(unittest.TestCase):

  def setUp(self):

    self.dic = {(0,0):[0], (0,0.1):[4], (0,0.2):[8], (1,0):[1], (1,0.1):[5], (1,0.2):[9], (2,0):[2], (2,0.1):[6], (2,0.2):[10],(3,0):[3],(3,0.1):[7],(3,0.2):[11]}
    self.d = range(4)
    self.l = np.arange(0,0.3,0.1)

  def test_transform(self):

    mat = dic2mat(self.d,self.l,self.dic,1)
    exp_mat = np.array([[0,1,2,3],[4,5,6,7],[8,9,10,11]])
    diff = mat-exp_mat
    self.assertFalse(diff.any())


class Compare(unittest.TestCase):

  def setUp(self):

    self.y_actual = [0,0,0,0,0,1,1,1,1,1]
    self.y_predict = [0,0,0,1,0,0,0,1,1,1]

  def test_compare(self):

    false_pos,false_neg,true_pos = comparison(self.y_predict,self.y_actual)
    self.assertEqual(true_pos,3)
    self.assertEqual(false_pos,1)
    self.assertEqual(false_neg,2)


class Synthetics(unittest.TestCase):

  def setUp(self):

    from sklearn import datasets

    self.iris = datasets.load_iris()
    data = pd.DataFrame(self.iris['data'],columns=self.iris['feature_names'])
    data['target'] = self.iris['target']

    sizes = [300,800,50]
    new_data = data.copy()
    for i in np.unique(data.target):
      cl = data[data.target==i]
      nb = sizes[i]
      nb_tir = nb-len(cl)
      r = {}
      r['target'] = i*np.ones(nb_tir,dtype=int)
      for feat in self.iris['feature_names']:
        mini = np.min(cl['%s'%feat])
        maxi = np.max(cl['%s'%feat])
        r['%s'%feat] = np.random.rand(nb_tir)*(maxi-mini)+mini
      r = pd.DataFrame(r)
      self.new_data = new_data.append(r,ignore_index=True)

    self.data = data

  def test_original_dataset(self):

    print "\n-------------------Original data set-------------------"
    x_data = self.data.reindex(columns=self.iris['feature_names'])
    y_data = self.data.reindex(columns=['target'])
    y_data.columns = ['NumType']
    y_train, y_cv, y_test = generate_datasets((0.6,0.2,0.2),np.unique(y_data.values),y_data)
    do_all_logistic_regression(x_data,y_data,list(y_train.index),list(y_cv.index),list(y_test.index))

  def test_extended_dataset(self):

    print "\n-------------------Extended data set-------------------"
    x_data = self.new_data.reindex(columns=self.iris['feature_names'])
    y_data = self.new_data.reindex(columns=['target'])
    y_data.columns = ['NumType']
    y_train, y_cv, y_test = generate_datasets((0.6,0.2,0.2),np.unique(y_data.values),y_data)
    do_all_logistic_regression(x_data,y_data,list(y_train.index),list(y_cv.index),list(y_test.index))

  def plot_features(self):

    import matplotlib.pyplot as plt

    for i,feat1 in enumerate(self.iris['feature_names']):
      for feat2 in self.iris['feature_names'][i+1:]:
        fig = plt.figure()
        fig.set_facecolor('white')
        plt.scatter(self.data['%s'%feat1],self.data['%s'%feat2],c=self.data['target'],cmap=plt.cm.gray)
        plt.xlabel(feat1)
        plt.ylabel(feat2)

        fig = plt.figure()
        fig.set_facecolor('white')
        plt.scatter(self.new_data['%s'%feat1],self.new_data['%s'%feat2],c=self.new_data['target'],cmap=plt.cm.gray)
        plt.xlabel(feat1)
        plt.ylabel(feat2)
        plt.show()


if __name__ == '__main__':

  unittest.TextTestRunner(verbosity=2).run(suite())
