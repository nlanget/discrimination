#!/usr/bin/env python
# encoding: utf-8

import sys
import numpy as np
import matplotlib.pyplot as plt
from plot_functions import *
from polynome import *
import pandas as pd
# ---------------------------------------------------
def g(x):
  """
  Sigmoid function
  """
  return 1./(1+np.exp(-x))

# ---------------------------------------------------

def features_mat(x):
  """
  Add the bias term to the matrix
  """
  m = x.shape[0]  # size of training set
  x_all = np.ones(m)
  x_all = np.vstack((x_all,x.T.values))
  return x_all

# ---------------------------------------------------

def normalize(x,x_data):
  """
  Normalizes x and x_data
  x and x_data are pandas DataFrame
  """
  av = x.mean()
  r = x.max() - x.min()
  x = (x - av) / r
  x_data = (x_data - av) / r
  return x, x_data

# ---------------------------------------------------

class CostFunction():

  def __init__(self,x,y,l):
    self.x = x # dataset (DataFrame type)
    self.y = y # label (DataFrame type)
    self.l = l # regularization coefficient
    self.m = self.x.shape[0] # number of data set examples
    self.n = self.x.shape[1] # number of features
    self.x_mat = np.vstack((np.ones(self.m),self.x.T.values)) # np.array with the bias term

  def predict_y(self,theta):
    """
    Computes the hypothesis function
    """
    y_pred = g(np.dot(theta.reshape(1,len(theta)),self.x_mat))
    y_pred = y_pred.ravel()
    return y_pred

  def compute_cost(self,theta):
    """
    Computes the cost function
    """
    y_pred=self.predict_y(theta)

    jVal = 0
    for i in range(len(self.y)):
      jVal+=self.y[i]*np.log10(y_pred[i])+(1-self.y[i])*np.log10(1-y_pred[i])
    jVal = -1./self.m*jVal+1./(2*self.m)*self.l*np.sum(theta[1:]**2)

    return jVal

  def compute_gradient(self,theta):
    """
    Computes the gradient of the cost function
    """
    y_pred = self.predict_y(theta)
    gradient = 1./self.m*np.dot(self.x_mat,y_pred-self.y)
    return gradient

  def cost_and_gradient(self,theta):
    """
    Computes the cost function and its gradient
    """
    y_pred = self.predict_y(theta)

    jVal = 0
    for i in range(len(self.y)):
      jVal+=self.y[i]*np.log10(y_pred[i])+(1-self.y[i])*np.log10(1-y_pred[i])
    jVal = -1./self.m*jVal+1./(2*self.m)*self.l*np.sum(theta[1:]**2)

    gradient = 1./self.m*np.dot(self.x_mat,y_pred-self.y)

    return jVal, gradient

# ---------------------------------------------------

def gradient_descent(CF,theta,opt=1,verbose=False):
  """
  Gradient descent
  Options:
    opt = 0: learning rate alpha is a constant
    opt = 1: learning rate alpha is optimized at each iteration
    Default is opt = 0
  """
  eps = 10**-4
  alpha = 0.5
  prev_cost = 1000
  cost = 0
  diff = np.abs(prev_cost-cost)
  min_cost = []
  while diff > eps:
    cost, delta = CF.cost_and_gradient(theta)
    min_cost.append(cost)

    if opt == 0:
      theta[0]=theta[0]-alpha*delta[0]
      theta[1:]=(1-alpha*CF.l*1./CF.m)*theta[1:]-alpha*delta[1:]

    if opt == 1:
      # Update alpha at each iteration (convergence fastened) 
      A = np.empty((CF.n,CF.n))
      for i in range(CF.n):
        for j in range(i,CF.n):
          A[i,j] = np.dot(x[i,:],x[j,:])
          A[j,i] = A[i,j]
      A = 1./CF.m*A
      alpha = np.dot(delta,delta)/(np.dot(delta,np.dot(A,delta)))

    diff = np.abs(prev_cost-cost)
    prev_cost = cost

    if verbose:
      if CF.n == 2:
        plot_db(CF.x,CF.y,theta)
      if CF.n == 3:
        plot_db_3d(CF.x,CF.y,theta)
      plt.show()

  return theta, min_cost

# ---------------------------------------------------

def logistic_reg(x,y,theta,l=0,verbose=0,method='g'):
  """
  Determines theta vector for a given polynomial degree and lambda
  x is a panda DataFrame
  y is a panda DataFrame
  l = 0: regularization coefficient / default is no regularization
  Methods for cost function minimization (default is gradient descent):
    'g': gradient descent
    'cg': conjugate gradient
    'bfgs': BFGS (Broyden Fletcher Goldfarb Shanno)
  """
  # Number of features
  n = x.shape[1]
  # Number of training set examples
  m = x.shape[0]
  # Number of classes
  K = y.shape[1]

  if len(theta[1]) != n+1:
    print "In logistic_reg.py:\nproblem of dimension between number of features and number of parameters !!"
    print "Number of features:", n
    print "Length of theta vector:", len(theta[1])
    sys.exit()

  for k in range(1,K+1):
    theta[k] = np.array(theta[k],dtype=float)
    CF = CostFunction(x,y.values[:,k-1],l)

    if verbose:
      if n == 1:
        from PdF_log_reg import hypothesis_function
        syn, hyp = hypothesis_function(x.min(),x.max(),theta[k])
        plot_hyp_func(x,y[k],syn,hyp)
      if n == 2:
        plot_db(x,y[k],theta[k],lim=3,title='Initial decision boundary')
      if n == 3:
        plot_db_3d(x,y[k],theta[k],lim=3,title='Initial decision boundary')

    stop = 10**-3
    if method == 'cg':
      # Conjugate gradient
      from scipy.optimize import fmin_cg
      theta[k],allvecs = fmin_cg(CF.compute_cost,theta[k],fprime=CF.compute_gradient,gtol=stop,disp=verbose,retall=True)
    elif method == 'bfgs':
    # BFGS (Broyden Fletcher Goldfarb Shanno)
      from scipy.optimize import fmin_bfgs
      theta[k],allvecs = fmin_bfgs(CF.compute_cost,theta[k],fprime=CF.compute_gradient,gtol=stop,disp=verbose,retall=True)
    elif method == 'g':
      # Gradient descent
      theta[k],min_cost = gradient_descent(CF,theta[k],opt=0)
      allvecs = None
  
    if verbose:
      if allvecs: 
        min_cost = []
        for vec in allvecs:
          min_cost.append(CF.compute_cost(vec))
      nb_iter = len(min_cost)
      plot_cost_function(nb_iter,min_cost)
      plt.show()

  if verbose:
    if n == 1 and K == 1:
      from PdF_log_reg import hypothesis_function
      syn, hyp = hypothesis_function(x.min(),x.max(),theta[1])
      plot_hyp_func(x,y[1],syn,hyp)
    if n == 2:
      if K != 1:
        plot_multiclass_2d(x,theta)
      else:
        plot_db(x,y,theta[1],title='Decision boundary')
    if n == 3:
      if K != 1:
        plot_multiclass_3d(x,theta)
      else:
        plot_db_3d(x,y,theta[1],title='Decision boundary')
    plt.show()

  return theta

# ---------------------------------------------------

def degree_and_regularization(xtest,ytest,xcv,ycv,xtrain,ytrain,verbose=False):
  """
  Looks for the best polynomial degree (model selection) and lambda (regularization)
  xtest, ytest, xcv, ycv, xtrain, ytrain are pandas DataFrame
  """
  n = xtest.shape[1] # number of features
  K = ytest.shape[1] # number of classes
  mtest = xtest.shape[0] # size of test set
  mcv = xcv.shape[0] # size of cross-validation set
  mtraining = xtrain.shape[0] # size of training set

  # Polynomial degrees vector
  DEG_MAX = 1
  degrees = np.array(range(1,DEG_MAX+1),dtype=int)
  # Lambda vector (regulariztion coefficient)
  lambdas = [0]
  lambdas = list(10.0 ** np.arange(-2, 5))

  all_theta = {}

  list_j_cv,list_j_test,list_j_train = {},{},{}
  min_cv = np.zeros(K)+10**3
  best_dl = np.empty([K,2])

  # loop on polynomial degree
  for deg in degrees:
    xtest_deg = poly(deg,xtest)
    xcv_deg = poly(deg,xcv)
    xtrain_deg = poly(deg,xtrain)

    # loop on lambda
    for l in lambdas:
      theta = {}
      for k in range(1,K+1):
        theta[k] = np.random.rand(xtrain_deg.shape[1]+1)

      theta = logistic_reg(xtrain_deg,ytrain,theta,l=l,verbose=0)
      all_theta[deg,l] = theta
      if verbose:
        print deg,l,theta

      list_j_cv[deg,l],list_j_test[deg,l],list_j_train[deg,l] = [],[],[]
      for k in range(1,K+1):
        CF_cv = CostFunction(xcv_deg,ycv[k],l)
        j_cv = CF_cv.compute_cost(theta[k])
        CF_train = CostFunction(xtrain_deg,ytrain[k],l)
        j_train = CF_train.compute_cost(theta[k])
        CF_test = CostFunction(xtest_deg,ytest[k],l)
        j_test = CF_test.compute_cost(theta[k])

        list_j_cv[deg,l].append(j_cv)
        list_j_test[deg,l].append(j_test)
        list_j_train[deg,l].append(j_train)

        if j_cv < min_cv[k-1]:
          best_dl[k-1,:] = [deg,l]
          min_cv[k-1] = j_cv

  theta = {}
  for k in range(1,K+1):
    theta[k] = all_theta[best_dl[k-1][0],best_dl[k-1][1]][k]

  if verbose:
    print best_dl
    print theta
    if len(degrees) > 1 and len(lambdas) > 1:
      from mpl_toolkits.mplot3d import Axes3D
      for k in range(1,K+1):
        plot_deg_vs_lambda(k,degrees,lambdas,list_j_cv,best_dl[k-1],min_cv[k-1])
        plot_deg_vs_lambda(k,degrees,lambdas,list_j_train,best_dl[k-1],min_cv[k-1])
      plt.show()
    elif len(degrees) > 1 and len(lambdas) == 1:
      for k in range(1,K+1):
        plot_j_cv = [list_j_cv[(deg,l)][k-1] for deg,l in sorted(list_j_cv)]
        plot_j_train = [list_j_train[(deg,l)][k-1] for deg,l in sorted(list_j_train)]
        plot_learning_curves(k,degrees,lambdas[0],plot_j_cv,plot_j_train,'Polynomial degree')
      plt.show()
    elif len(degrees) == 1 and len(lambdas) > 1:
      for k in range(1,K+1):
        plot_j_cv = [list_j_cv[(deg,l)][k-1] for deg,l in sorted(list_j_cv)]
        plot_j_train = [list_j_train[(deg,l)][k-1] for deg,l in sorted(list_j_train)]
        plot_learning_curves(k,lambdas,degrees[0],plot_j_cv,plot_j_train,'Lambda')
      plt.show()

  return best_dl,theta

# ---------------------------------------------------

def misclassification_error(y,y_pred,t):
  err = 0
  for i in range(len(y)):
    if y_pred[i] >= t and y[i] == 0:
      err+=1
    elif y_pred[i] < t and y[i] == 1:
      err+=1
    else:
      err+=0
  return 1./len(y)*err

# ---------------------------------------------------

def test_hyp(xtest,theta,threshold=0.5,deg=None,verbose=0):
  """
  Returns a classification vector for a given test set after the hypothesis function was determined on the training set
  xtest is a pandas DataFrame
  """

  K = len(theta)
  if deg == None:
    deg = np.ones(K)

  hyp = []
  a = xtest.copy()
  for i in range(1,K+1):
    a = poly(int(deg[i-1]),xtest)
    a = features_mat(a)
    hyp.append(g(np.dot(theta[i].reshape(1,len(theta[i])),a))[0])

  if K >= 2:
    hyp = np.array(hyp).transpose()

  x_class = []
  for i in range(xtest.shape[0]):
    if K >= 2:
      x_class.append(np.argmax(hyp[i]))
    else:
      if hyp[0][i] < threshold:
        x_class.append(0)
      else:
        x_class.append(1)

  if verbose:
    if xtest.shape[1] == 2:
      plt.plot(xtest.values[:,0],xtest.values[:,1],'yo')
      plt.show()
    if xtest.shape[1] == 3:
      #ax.scatter(xtest.values[:,0],xtest.values[:,1],xtest.values[:,2],c='y')
      plt.show()

  return np.array(x_class)

# --------------------------------------------------

def comparison(y_predict,y_actual):
  """
  Compares the prediction with the true classification
  y_predict and y_actual are lists or np.arrays
  """
  if not type(y_predict) == np.ndarray:
    y_predict = np.array(y_predict)
  if not type(y_actual) == np.ndarray:
    y_actual = np.array(y_actual)  

  y_diff = y_predict-y_actual

  false_pos = len(np.where(y_diff==1)[0]) # EB misclassified as VT
  false_neg = len(np.where(y_diff==-1)[0]) # VT misclassified as EB

  index_vt_actual = np.where(y_actual==1)[0]
  index_vt_predict = np.where(y_predict==1)[0]
  true_pos = 0
  for ind in index_vt_actual:
    if ind in index_vt_predict:
      true_pos+=1 # well classified VTs

  return false_pos,false_neg,true_pos

# ---------------------------------------------------

def precision_and_recall(x,y,theta,verbose=False):
  """
  Precision and recall: try different prediction thresholds
  and choose the one which maximizes the F1 score
  x is a pandas DataFrame
  y is a np.array
  """
  thresholds = np.arange(0.05,1,0.05)
  P,R,f1_score = [],[],[]
  if not type(theta) is dict:
    theta = {1:theta}
  for t in thresholds:
    predict = test_hyp(x,theta,threshold=t,verbose=0)

    false_pos,false_neg,true_pos = comparison(predict,y)

    if true_pos:
      precision = np.float(true_pos)/(true_pos+false_pos) # amongst all predicted VTs, proportion of VTs which actually are VT
      recall = np.float(true_pos)/(true_pos+false_neg) # amongst all real VTs, proportion of well classified VTs
      P.append(precision)  
      R.append(recall)
      f1_score.append(2*precision*recall/(precision+recall))

  if f1_score:
    best_t = thresholds[np.argmax(f1_score)]
  else:
    best_t = .5

  if verbose:
    print "threshold", best_t
    # Plot recall vs precision
    fig = plt.figure()
    fig.set_facecolor('white')
    plt.plot(R,P,'r')
    plt.xlim((-0.01,1.01))
    plt.ylim((-0.01,1.01))
    plt.xlabel('Recall')
    plt.ylabel('Precision')

  return best_t

# ---------------------------------------------------

def data_sets(x,y,wtr=None,verbose=False):
  """
  Randomly generates training/cross-validation/test sets
  x, y are pandas DataFrame
  """
  m = x.shape[0] # size of the training set
  n = x.shape[1] # number of features
  K = y.shape[1] # number of classes

  mtraining = int(0.6*m)
  mtest = int(0.2*m)
  mcv = int(0.2*m)

  if not wtr:
    wtr = list(np.random.permutation(m))

  xtrain = x.reindex(index=wtr[:mtraining])
  ytrain = y.reindex(index=wtr[:mtraining])
  xtrain.index = range(xtrain.shape[0])
  ytrain.index = range(ytrain.shape[0])

  xcv = x.reindex(index=wtr[mtraining:mtraining+mcv])
  ycv = y.reindex(index=wtr[mtraining:mtraining+mcv])
  xcv.index = range(xcv.shape[0])
  ycv.index = range(ycv.shape[0])

  xtest = x.reindex(index=wtr[mtraining+mcv:])
  ytest = y.reindex(index=wtr[mtraining+mcv:])
  xtest.index = range(xtest.shape[0])
  ytest.index = range(ytest.shape[0])

  if verbose:
    if n > 1:
      fig = plt.figure()
      fig.set_facecolor('white')
      plt.plot(xtrain.values[:,0],xtrain.values[:,1],'ko')
      plt.plot(xcv.values[:,0],xcv.values[:,1],'yo')
      plt.plot(xtest.values[:,0],xtest.values[:,1],'ro')
      plt.xlabel(x.columns[0])
      plt.ylabel(x.columns[1])
      plt.legend(('Training set','CV set','Test set'),numpoints=1)#,'upper left')
      plt.show()

  retlist = xcv,ycv,xtest,ytest,xtrain,ytrain,wtr
  return retlist

# ---------------------------------------------------

def evaluation(x,y,wtr=np.array([]),learn=False,verbose=False):
  """
  Returns the best theta vector as well as the polynomial degree, lambda and the prediction threshold
  x, y are pandas DataFrame
  If learn = True, then learning curves are computed and displayed
  """ 
  m = x.shape[0] # size of the training set
  n = x.shape[1] # number of features
  K = y.shape[1] # number of classes

  # Separation of the training set in: training/CV/test sets for learning curves calculation
  if list(wtr):
    xcv,ycv,xtest,ytest,xtrain,ytrain,wtr = data_sets(x,y,wtr=wtr,verbose=False)
  else:
    xcv,ycv,xtest,ytest,xtrain,ytrain,wtr = data_sets(x,y,verbose=False)

  mcv = xcv.shape[0]
  mtest = xtest.shape[0]
  mtraining = xtrain.shape[0]

  # Determination of the best hypothesis function
  best_dl,theta = degree_and_regularization(xtest,ytest,xcv,ycv,xtrain,ytrain,verbose=verbose)
 
  # Misclassification error on test set
  dic_xtest,dic_xcv,dic_xtrain,dic_x = {},{},{},{}
  err,best_threshold = {},{}
  for k in range(1,K+1):
    degree = int(best_dl[k-1][0])
    l = best_dl[k-1][1]

    dic_x[k] = poly(degree,x)
    dic_xtest[k] = poly(degree,xtest)
    dic_xcv[k] = poly(degree,xcv)
    dic_xtrain[k] = poly(degree,xtrain)

    # Determination of the best prediction threshold
    best_threshold[k] = precision_and_recall(dic_xcv[k],ycv[k].values,theta[k])

    CF_test = CostFunction(dic_xtest[k],ytest[k],l)
    y_test_pred = CF_test.predict_y(theta[k])
    err[k] = misclassification_error(ytest[k].values,y_test_pred,best_threshold[k])
  print "MISCLASSIFICATION TEST ERROR", err

  # Learning curves
  if learn:
    print "Computing learning curves......."
    verbose = True
    list_j_cv,list_j_train={},{}
    c = {}
    for k in range(1,K+1):
      list_j_cv[k],list_j_train[k] = [],[]
      l = best_dl[k-1][1]

      t = {1:theta[k]}

      for i in range(1,mcv):
        a = dic_xcv[k].reindex(index=range(i))
        b = dic_xtrain[k].reindex(index=range(i))
        c = pd.DataFrame({1:ytrain[k][:i]})
        t = logistic_reg(b,c,t,l,verbose=False)

        CF_cv = CostFunction(a,ycv[k][:i],l)
        j_cv = CF_cv.compute_cost(theta[k])
        CF_train = CostFunction(b,ytrain[k][:i],l)
        j_train = CF_train.compute_cost(theta[k])
        list_j_cv[k].append(j_cv)
        list_j_train[k].append(j_train)

      if verbose:
        fig = plt.figure()
        fig.set_facecolor('white')
        plt.plot(range(1,mcv),list_j_cv[k],'b')
        plt.plot(range(1,mcv),list_j_train[k],'r')
        plt.xlabel('Size of training set')
        plt.ylabel('Cost function')
        plt.legend(('Jcv','Jtrain'),'upper left')
        plt.title('Class %d'%k)
        plt.show()

        if len(dic_x[k]) == 2:
          plot_db(dic_x[k],y[k].values,theta[k])
        if len(dic_x[k]) == 3:
          plot_db_3d(dic_x[k],y[k].values,theta[k])

  retlist = theta,best_dl,best_threshold
  if list(wtr):
    retlist = retlist + (wtr,)

  return retlist
# ---------------------------------------------------
def bad_class(x_test,list_i):
  """
  Returns a DataFrame containing misclassified points only
  x_test is a pandas DataFrame
  """
  x_bad = x_test.copy()
  x_bad = x_bad.reindex(index=list_i)
  return x_bad
# ---------------------------------------------------
def do_all_logistic_regression(x,y_all,x_testset,y_testset=None,norm=True,verbose=False,output=False,perc=False,wtr=np.array([])):
  """
  Implements the whole logistic regression process
  1) datasets normalization
  2) determination of the best theta vector (associated with the best degree/lambda/threshold) for each class
  3) prediction and classification of the test set
  y_testset: if available, the function will compare it to the results obtained by logistic regression 
  If norm = True: data from the training and test sets are normalized. Default is True. Must be deactivated if data are already normalized (e.g. after PCA...)
  If output = True: returns prediction for both training and test sets and theta vector
  If perc = True: returns the percentage rates of recovery of the training and test sets
  If wtr: indexes of the events in the following order : "true" training set (60%), CV set (20%) and test set (20%)
  """
  x.index = range(len(x.index))
  y_all.index = range(len(y_all.index))
  x_testset.index = range(len(x_testset.index))
  if y_testset:
    y_testset.index = range(len(y_testset.index))

  x_unnorm = x.copy()
  x_test_unnorm = x_testset.copy()

  # For multiclass: separation of values in y for "one-vs-all" strategy
  K = len(np.unique(y_all.values)) # Number of classes
  print "Number of events - training set = %d"%y_all.shape[0]
  print "Number of events - test set = %d"%x_testset.shape[0]
  print "Number of features = %d"%x.shape[1]
  print "Number of classes = %d"%K
  if K > 2:
    y = y_all.copy().reindex(columns=[])
    for i in range(K):
      a = y_all.copy()
      a[y_all!=i] = 0
      a[y_all==i] = 1
      y[i] = a.values[:,0]
  else:
    y = y_all.copy()

  y.columns = range(1,y.shape[1]+1) # Class numbering begins at 1 (instead of 0)
  y_all.columns = [1]


  # Normalization
  if norm:
    x,x_testset = normalize(x,x_testset)

  if list(wtr):
    theta,deg_and_lambda,thres,wtr = evaluation(x,y,wtr=wtr,learn=False,verbose=False)
  else:
    theta,deg_and_lambda,thres,wtr = evaluation(x,y,learn=False,verbose=False)

  for k in range(1,len(theta)+1):
    print "Class %d: degree = %d - lambda = %.1f - threshold = %.2f"%(k,int(deg_and_lambda[k-1,0]),deg_and_lambda[k-1,1],thres[k])

  deg=deg_and_lambda[:,0]
  # Test hypothesis function on training set
  y_pred_train = test_hyp(x,theta,deg=deg,threshold=thres[1],verbose=False)

  print "\nTRAINING SET"
  p_tr = np.float(len(np.where(y_pred_train-y_all[1]==0)[0]))*100/y.shape[0]
  print "Correct classification: %.2f %%"%p_tr

  # Test hypothesis function on test set
  y_pred = test_hyp(x_testset,theta,deg=deg,threshold=thres[1],verbose=False)
  if y_testset:
    x_bad = bad_class(x_test_unnorm,np.where(y_pred-y_testset.values[:,0]!=0)[0])
    print "TEST SET"
    p_test = 100-np.float(x_bad.shape[0]*100)/y_testset.shape[0]
    print "Correct classification: %.2f %%"%p_test

  #verbose=True
  if verbose and x.shape[1] < 5:
    list_key = x_unnorm.columns
    for i,key1 in enumerate(list_key):
      for key2 in list_key[i+1:]:
        fig = plt.figure()
        fig.set_facecolor('white')
        plt.scatter(x_unnorm[key1],x_unnorm[key2],c=y.values[:,0],cmap=plt.cm.gray)
        plt.xlabel(key1)
        plt.ylabel(key2)
        plt.title('Training set')

        fig = plt.figure()
        fig.set_facecolor('white')
        plt.scatter(x_test_unnorm[key1],x_test_unnorm[key2],c=y_pred,cmap=plt.cm.gray)
        if y_testset:
          plt.plot(x_bad[key1],x_bad[key2],'ro')
          plt.legend(('Bad class',),numpoints=1)
        plt.xlabel(key1)
        plt.ylabel(key2)
        plt.title('Test set')
    plt.show()

  if output:
    retlist = y_pred_train,theta,y_pred
  else:
    retlist = ()
  if perc:
    if not y_testset:
      retlist = retlist + ((p_tr,),)
    else:
      retlist = retlist + ((p_tr,p_test),)
  if list(wtr):
    retlist = retlist + (wtr,)
  return retlist
