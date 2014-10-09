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
def plot_cost_function(thetas):
  """
  Plot of the cost function.
  """
  n = len(thetas)-1 # number of features
  pas = .01
  x_ini = np.arange(-1,1,pas)

  if n == 1:
    x = np.vstack((np.ones(len(x_ini)),x_ini))
    thx = np.dot(thetas.T,x)
  else:
    x = np.vstack((np.ones(len(x_ini)),x_ini))
    i = 1
    while i < n:
      x = np.vstack((x,x_ini))
      i = i+1
    thx = np.dot(thetas.T,x)

  h = g(thx)

  fig = plt.figure(figsize=(10,4.5))
  fig.set_facecolor('white')

  ax1 = fig.add_subplot(121)
  plt.plot(h,-np.log(h),'b',lw=3.)
  plt.xlabel(r'$h_\theta(x)$')
  plt.ylabel('Cost function')
  ax1.text(.75,.9,'y = 1',size='xx-large',transform=ax1.transAxes)

  ax2 = fig.add_subplot(122)
  plt.plot(h,-np.log(1-h),'b',lw=3.)
  plt.xlabel(r'$h_\theta(x)$')
  plt.ylabel('Cost function')
  ax2.text(.75,.9,'y = 0',size='xx-large',transform=ax2.transAxes)

  plt.show()

# ---------------------------------------------------

class CostFunction():


  def __init__(self,x,y,l):
    self.x = x # dataset (type : pandas.core.series.Series)
    self.y = y # label (type : pandas.core.series.Series)
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
    y_pred = self.predict_y(theta)

    j = self.y*np.log10(y_pred)+(1-self.y)*np.log10(1-y_pred)
    jVal = -1./self.m*np.sum(j) + 1./(2*self.m)*self.l*np.sum(theta[1:]**2)

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

    j = self.y*np.log10(y_pred)+(1-self.y)*np.log10(1-y_pred)
    jVal = -1./self.m*np.sum(j) + 1./(2*self.m)*self.l*np.sum(theta[1:]**2)

    gradient = 1./self.m*np.dot(self.x_mat,y_pred-self.y)

    return jVal, gradient


  def compute_hessian(self,theta):
    """
    Computes the Hessian matrix (second-order partial derivatives)
    """
    y_pred = self.predict_y(theta)
    y_pred_multi = list(y_pred)*self.n

    A = np.empty((self.n+1,self.n+1))
    for i in range(self.n+1):
      for j in range(i,self.n+1):
        A[i,j] = np.dot(self.x_mat[i,:],self.x_mat[j,:])
        print A[i,j].shape
        A[j,i] = A[i,j]

    hessian = 1./self.m*A*y_pred*(1-y_pred)

    return hessian

# ---------------------------------------------------

def gradient_descent(CF,theta,eps,opt=1,verbose=False):
  """
  Gradient descent
  Options:
    opt = 0: learning rate alpha is a constant
    opt = 1: learning rate alpha is optimized at each iteration
    Default is opt = 0
  """
  alpha = 0.05#*10
  prev_cost = 1000
  cost = 0
  diff = np.abs(prev_cost-cost)
  min_cost = []
  while diff > eps:
    cost, delta = CF.cost_and_gradient(theta)
    min_cost.append(cost)

    theta[0]=theta[0]-alpha*delta[0]
    theta[1:]=(1-alpha*CF.l*1./CF.m)*theta[1:]-alpha*delta[1:]

    if opt == 1:
      # Update alpha at each iteration (convergence fastened) 
      hess = CF.compute_hessian(theta)
      #hess = 1./CF.m*hess
      #delta = np.matrix(delta).T
      #alpha = np.dot(delta.T,delta)/(np.dot(delta.T,np.dot(hess,delta)))
      #alpha = alpha[0,0]

      A = np.matrix(1./CF.m*A)
      print type(delta[1:]), type(A)
      alpha = np.dot(delta[1:].T,delta[1:])/(np.dot(delta[1:].T,np.dot(A,delta[1:])))
      alpha = alpha[0,0]

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
  K = len(y.columns)

  if len(theta[1]) != n+1:
    print "In logistic_reg.py:\nproblem of dimension between number of features and number of parameters !!"
    print "Number of features:", n
    print "Length of theta vector:", len(theta[1])
    sys.exit()

  for k in range(1,K+1):
    theta[k] = np.array(theta[k],dtype=float)
    CF = CostFunction(x,y.values[:,k-1],l)

    verbose = False
    if verbose:
      if n == 1:
        from plot_functions import plot_hyp_func_1f, plot_sep_1f
        syn, hyp = hypothesis(x.min(),x.max(),theta[k])
        plot_hyp_func_1f(x,y[k],syn,hyp,threshold=.5)
      if n == 2:
        plot_db(x,y[k],theta[k],lim=3,title='Initial decision boundary')
      if n == 3:
        plot_db_3d(x,y[k],theta[k],lim=3,title='Initial decision boundary')

    method = 'bfgs'
    stop = 10**-5
    if method == 'cg':
      # Conjugate gradient
      from scipy.optimize import fmin_cg
      theta[k],allvecs = fmin_cg(CF.compute_cost,theta[k],fprime=CF.compute_gradient,gtol=stop,disp=False,retall=True)#,maxiter=1000)
    elif method == 'bfgs':
    # BFGS (Broyden Fletcher Goldfarb Shanno)
      from scipy.optimize import fmin_bfgs
      theta[k],allvecs = fmin_bfgs(CF.compute_cost,theta[k],fprime=CF.compute_gradient,gtol=stop,disp=False,retall=True)
    elif method == 'g':
      # Gradient descent
      theta[k],min_cost = gradient_descent(CF,theta[k],stop,opt=0)
      allvecs = None

    verbose = False 
    if verbose:
      if allvecs: 
        min_cost = []
        for vec in allvecs:
          min_cost.append(CF.compute_cost(vec))
      nb_iter = len(min_cost)
      #plot_cost_function_iter(nb_iter,min_cost)
      #plt.show()

  verbose = False
  if verbose:
    if n == 1 and K == 1:
      from plot_functions import plot_hyp_func_1f
      syn, hyp = hypothesis(x.min(),x.max(),theta[1])
      plot_hyp_func_1f(x,y[1],syn,hyp,threshold=.5)
    if n == 2:
      if K != 1:
        from plot_functions import plot_multiclass_2d
        plot_multiclass_2d(x,theta)
      else:
        from plot_functions import plot_db
        plot_db(x,y,theta[1],title='Decision boundary')
    if n == 3:
      if K != 1:
        from plot_functions import plot_multiclass_3d
        plot_multiclass_3d(x,theta)
      else:
        from plot_functions import plot_db_3d
        plot_db_3d(x,y,theta[1],title='Decision boundary')
    plt.show()

  return theta

# ---------------------------------------------------

def degree_and_regularization(xcv,ycv,xtrain,ytrain,verbose=False):
  """
  Looks for the best polynomial degree (model selection) and lambda (regularization)
  xcv, ycv, xtrain, ytrain are pandas DataFrame
  """
  n = xtrain.shape[1] # number of features
  K = ytrain.shape[1] # number of classes
  mcv = xcv.shape[0] # size of cross-validation set
  mtraining = xtrain.shape[0] # size of training set
  
  # Polynomial degrees vector
  DEG_MAX = 1
  degrees = np.array(range(1,DEG_MAX+1),dtype=int)
  # Lambda vector (regularization coefficient)
  lambdas = list(10.0 ** np.arange(-2, 5))

  all_theta = {}

  list_j_cv,list_j_train = {},{}
  min_cv = np.zeros(K)+10**3
  best_dl = np.empty([K,2])

  # loop on polynomial degree
  for deg in degrees:
    xcv_deg = poly(deg,xcv)
    xtrain_deg = poly(deg,xtrain)

    # loop on lambda
    for l in lambdas:

      #boot = 10
      #for b in range(boot):
      # Tirage aléatoire des valeurs initiales de theta dans l'intervalle [-1,1]
      theta = {}
      for k in range(1,K+1):
        #theta[k] = np.random.rand(xtrain_deg.shape[1]+1)*2-1
        theta[k] = np.zeros(xtrain_deg.shape[1]+1)

      theta = logistic_reg(xtrain_deg,ytrain,theta,l=l,verbose=0)

      all_theta[deg,l] = theta
      if verbose:
        print deg,l,theta

      list_j_cv[deg,l],list_j_train[deg,l] = [],[]
      for k in range(1,K+1):
        CF_cv = CostFunction(xcv_deg,ycv[k],l)
        j_cv = CF_cv.compute_cost(theta[k])
        CF_train = CostFunction(xtrain_deg,ytrain[k],l)
        j_train = CF_train.compute_cost(theta[k])

        list_j_cv[deg,l].append(j_cv)
        list_j_train[deg,l].append(j_train)

        if j_cv < min_cv[k-1]:
          best_dl[k-1,:] = [deg,l]
          min_cv[k-1] = j_cv


  theta = {}
  for k in range(1,K+1):
    theta[k] = all_theta[best_dl[k-1][0],best_dl[k-1][1]][k]

  verbose = False
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

def precision_and_recall(x,y,theta):
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

  return best_t
# ---------------------------------------------------
def plot_precision_recall(x_train,y_train,x_test,y_test,theta):
  """
  Plots the precision and recall curves.
  Tests different thresholds.
  """
  precision, rappel = [],[]
  precision_tr, rappel_tr = [],[]
  thress = np.arange(0,1.05,.05)
  for t in thress:
    y_pred = test_hyp(x_test,theta,threshold=t,verbose=False)
    false_pos,false_neg,true_pos = comparison(y_pred,y_test)

    if true_pos != 0:
      precision.append(true_pos*1./(true_pos+false_pos))
      rappel.append(true_pos*1./(true_pos+false_neg))
    else:
      precision.append(1)
      rappel.append(0)

    y_pred_tr = test_hyp(x_train,theta,threshold=t,verbose=False)
    false_pos,false_neg,true_pos = comparison(y_pred_tr,y_train)

    if true_pos != 0:
      precision_tr.append(true_pos*1./(true_pos+false_pos))
      rappel_tr.append(true_pos*1./(true_pos+false_neg))
    else:
      precision_tr.append(1)
      rappel_tr.append(0)

  fig = plt.figure(figsize=(10,4))
  fig.set_facecolor('white')
  ax = fig.add_subplot(121)
  plt.plot(thress[:-1],precision[:-1],'b',lw=2,label='Precision')
  plt.plot(thress,rappel,'g',lw=2,label='Recall')
  plt.plot(thress[:-1],precision_tr[:-1],'b--')
  plt.plot(thress,rappel_tr,'g--')
  plt.legend(loc=3)
  plt.figtext(0.07,0.88,'(a)')
  plt.xlabel('Threshold')

  ax = fig.add_subplot(122)
  plt.plot(rappel[:-1],precision[:-1],'k',lw=2)
  plt.plot(rappel_tr[:-1],precision_tr[:-1],'k--')
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.xlim([0.3,1.1])
  plt.ylim([0.3,1.1])
  plt.figtext(0.49,0.88,'(b)')
  plt.show()

# ---------------------------------------------------

def plot_learning_curves(xtrain,xcv,ytrain,ycv,best):
  """
  Compute and plot learning curves to diagnose 
  bias vs variance in a discrimination problem.
  """
  K = len(sorted(xtrain)) # number of classes
  # For a class k
  for k in range(1,K+1):
    mcv = xcv[k].shape[0] # number of samples in the CV set
    mtrain = xtrain[k].shape[0] # number of samples in the training set
    n_feat = xtrain[k].shape[1] # number of features
    min = mcv

    list_jcv, list_jtrain = [],[]
    l = best[k-1][1] # lambda
    for i in range(min,mtrain):
      int_xtrain = xtrain[k].reindex(index=range(i))
      int_ytrain = ytrain[k].reindex(index=range(i))

      thetaL = logistic_reg(int_xtrain,pd.DataFrame(int_ytrain),{1:np.zeros(n_feat+1)},l=l,method='g',verbose=False)
      CF_train = CostFunction(int_xtrain,int_ytrain,l)
      j_train = CF_train.compute_cost(thetaL[1])

      CF_cv = CostFunction(xcv[k],ycv[k],l)
      j_cv = CF_cv.compute_cost(thetaL[1])

      list_jcv.append(j_cv)
      list_jtrain.append(j_train)

    fig = plt.figure()
    fig.set_facecolor('white')
    plt.plot(range(min,mtrain),list_jcv,'b',label=r'$J_{CV}$')
    plt.plot(range(min,mtrain),list_jtrain,'r',label=r'$J_{train}$')
    plt.xlabel('Size of training set')
    plt.ylabel('Cost function')
    plt.title('Class %d'%k)
    plt.legend(loc=1)
    plt.show()

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
def do_all_logistic_regression(x_all,y_all,i_train,i_cv,i_test,norm=True):
  """
  Implements the whole logistic regression process
  1) decomposition of the whole set in training, CV and test sets
  2) datasets normalization
  3) determination of the best theta vector (associated with the best degree/lambda/threshold) for each class
  4) prediction and classification of the test set
  If norm = True: data from the training and test sets are normalized. Default is True.
            Must be deactivated if data are already normalized (e.g. after PCA...)
  """
  x_all.index = range(len(x_all.index))
  y_all.index = range(len(y_all.index))

  # For multiclass: separation of values in y for "one-vs-all" strategy
  K = len(np.unique(y_all.NumType.values)) # Number of classes
  m = x_all.shape[0] # size of the whole set
  n = x_all.shape[1] # number of features
  print "Number of features = %d"%n
  print "Number of classes = %d"%K
  if K > 2:
    y = y_all.reindex(columns=[])
    for i in range(K):
      a = y_all.reindex(columns=['NumType'])
      a[y_all.NumType!=i] = 0
      a[y_all.NumType==i] = 1
      y[i] = a.values[:,0]
  else:
    y = y_all.reindex(columns=['NumType'])

  y.columns = range(1,y.shape[1]+1) # Class numbering begins at 1 (instead of 0)

  ### Separation of the whole set in: training/CV/test sets ###
  x = x_all.copy()

  xtrain = x.reindex(index=i_train)
  ytrain = y.reindex(index=i_train)
  xcv = x.reindex(index=i_cv)
  ycv = y.reindex(index=i_cv)
  xtest = x.reindex(index=i_test)
  ytest = y.reindex(index=i_test)

  # Normalization
  xtrain_unnorm = xtrain.copy()
  xtrain, xcv = normalize(xtrain_unnorm,xcv)
  xtrain, xtest = normalize(xtrain_unnorm,xtest)

  mtraining = xtrain.shape[0] # size of the training set
  mcv = xcv.shape[0] # size of the cross-validation set
  mtest = xtest.shape[0] # size of the test set

  # Determination of the best hypothesis function
  best_dl,theta = degree_and_regularization(xcv,ycv,xtrain,ytrain,verbose=False)

  ### MISCLASSIFICATION ERROR COMPUTED ON THE TEST SET ###
  dic_xtest, dic_xcv, dic_xtrain, dic_x = {},{},{},{}
  err, best_threshold = {},{}
  K = y.shape[1]
  for k in range(1,K+1):
    degree = int(best_dl[k-1][0])
    l = best_dl[k-1][1]

    dic_xtest[k] = poly(degree,xtest)
    dic_xcv[k] = poly(degree,xcv)
    dic_xtrain[k] = poly(degree,xtrain)

    # Determination of the best prediction threshold
    best_threshold[k] = precision_and_recall(dic_xcv[k],ycv[k].values,theta[k])

    CF_test = CostFunction(dic_xtest[k],ytest[k],l)
    y_test_pred = CF_test.predict_y(theta[k])
    err[k] = misclassification_error(ytest[k].values,y_test_pred,best_threshold[k])*100
  print "MISCLASSIFICATION TEST ERROR ",err

  # Learning curves
  learn = False
  if learn:
    print "Computing learning curves......."
    plot_learning_curves(dic_xtrain,dic_xcv,ytrain,ycv,best_dl)

  deg_and_lambda = best_dl
  thres = best_threshold

  for k in range(1,len(theta)+1):
    print "Class %d: degree = %d - lambda = %.1f - threshold = %.2f"%(k,int(deg_and_lambda[k-1,0]),deg_and_lambda[k-1,1],thres[k])

  deg = deg_and_lambda[:,0]
  # Test hypothesis function on training set
  y_pred_train = test_hyp(xtrain,theta,deg=deg,threshold=thres[1],verbose=False)

  # Classify the test set
  y_pred = test_hyp(xtest,theta,deg=deg,threshold=thres[1],verbose=False)

  output = {}
  output['label_test'] = y_pred
  output['label_train'] = y_pred_train
  output['thetas'] = theta
  output['threshold'] = thres
  return output
