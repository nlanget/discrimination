#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# ---------------------------------------------------
# Plotting function for 1 feature
def plot_hyp_func_1f(x,y,str_t,syn,hyp,x_ok=None,x_bad=None,text=None):
  """
  Plots the hypothesis function for one feature.
  Superimposed with histograms (training and test sets)
  x is a pandas DataFrame
  y is a pandas DataFrame
  """
  num_t = np.unique(y.Type.values)

  fig = plt.figure()
  fig.set_facecolor('white')

  x1 = x[y.Type==num_t[0]].values[:,0]
  x2 = x[y.Type==num_t[1]].values[:,0]

  nn,b,p = plt.hist([x1,x2],25,histtype='stepfilled',alpha=.2,color=('b','g'),label=str_t)
  norm = np.mean([np.max(nn[0]),np.max(nn[1])])
  plt.plot(syn,norm*hyp,'y-',lw=2,label='hypothesis')

  if x_ok and x_bad:
    nn, b, p = plt.hist([x_ok,x_bad],25,normed=True,color=('k','r'),histtype='step',fill=False,ls='dashed',lw=2,label=['Test Set'])

  plt.legend()
  if text:
    plt.figtext(0.15,0.85,"%.2f %% %s"%(text[0],str_t[0]),color='b')
    plt.figtext(0.15,0.8,"%.2f %% %s"%(text[1],str_t[1]),color='g')
    plt.figtext(0.15,0.75,"%.2f %% test set"%text[2])
    plt.figtext(0.15,0.7,"%.2f %% test set"%text[3],color='r')

  plt.xlabel(x.columns[0])
  plt.title(x.columns[0])
  plt.show()
# ---------------------------------------------------
def plot_sep_2f(x_train,y_train,str_t,x_all,y_all,x_bad,theta=[],text=None):
  """
  Plots the decision boundary for two feature.
  Superimposed with scatter plots (training and test sets)
  x_train, y_train, x_all, y_all are pandas DataFrame
  """
  n = x_train.shape[1]
  for ikey,key in enumerate(x_train.columns):
    for k in x_train.columns[ikey+1:]:
      fig = plt.figure()
      fig.set_facecolor('white')
      plt.scatter(x_all[key],x_all[k],c=list(y_all.values),cmap=plt.cm.gray,alpha=0.2)
      plt.scatter(x_train[key],x_train[k],c=list(y_train.values),cmap=plt.cm.winter,alpha=0.5)
      if key in x_bad.columns:
        plt.scatter(x_bad[key],x_bad[k],c='r',alpha=0.2)
      if list(theta):
        lim = .1
        x1 = x_all.values[:,0]
        x2 = x_all.values[:,1]
        xplot = np.arange(np.min(x1)-lim,np.max(x1)+lim,0.1)
        plt.plot(xplot,1./theta[2]*(-theta[0]-theta[1]*xplot),'y',lw=3.)
        plt.xlim((np.min(x1)-lim,np.max(x1)+lim))
        plt.ylim((np.min(x2)-lim,np.max(x2)+lim))
      if text:
        plt.figtext(0.7,0.85,"%.2f %% %s"%(text[0],str_t[0]),color='b')
        plt.figtext(0.7,0.8,"%.2f %% %s"%(text[1],str_t[1]),color='g')
        plt.figtext(0.7,0.75,"%.2f %% test set"%text[2])
        plt.figtext(0.7,0.7,"%.2f %% test set"%text[3],color='r')
      plt.xlabel(key)
      plt.ylabel(k)
      plt.ylim([-1,1])
      plt.title('%s and %s'%(x_all.columns[0],x_all.columns[1]))
      plt.show()
# ---------------------------------------------------
# Plotting function for 2 features and degree = 2
def plot_db(x,y,theta,lim=.1,title=None):
  """
  Plots the decision boundary 
  x is a pandas DataFrame
  y is a np.array
  """
  x1 = x.values[:,0]
  x2 = x.values[:,1]

  fig = plt.figure()
  fig.set_facecolor('white')
  xplot = np.arange(np.min(x1)-lim,np.max(x1)+lim,0.1)
  plt.scatter(x1,x2,c=y,cmap=plt.cm.gray)
  plt.plot(xplot,1./theta[2]*(-theta[0]-theta[1]*xplot),'r')
  plt.xlim((np.min(x1)-lim,np.max(x1)+lim))
  plt.ylim((np.min(x2)-lim,np.max(x2)+lim))
  plt.xlabel(x.columns[0])
  plt.ylabel(x.columns[1])
  if title:
    plt.title(title)
# ---------------------------------------------------
def plot_db_test(x,y,theta,lim=.1,title=None):
  """
  Plots the decision boundary 
  x is a pandas DataFrame
  y is a np.array
  """
  x1 = x.values[:,0]
  x2 = x.values[:,1]

  fig = plt.figure()
  fig.set_facecolor('white')
  ax = fig.add_subplot(111)
  xplot = np.arange(np.min(x1)-lim,np.max(x1)+lim,0.1)
  plt.scatter(x1,x2,c=y,cmap=plt.cm.gray,s=50)
  plt.plot(xplot,1./theta[2]*(-theta[0]-theta[1]*xplot),'r',lw=2.)
  plt.text(4.7,5.5,"LR",color='r',size='xx-large')
  plt.plot([1.5,6.5],[6,1.5],'g',lw=2.)
  plt.text(2.5,5.5,"SVM",color='g',size='xx-large')
  plt.plot([.5,4.5],[5,1.5],'g--')
  plt.plot([3.5,7],[6,3],'g--')
  ax.annotate("",xy=(4.5,1.6),xycoords='data',xytext=(6.5,3.4),textcoords='data',arrowprops=dict(arrowstyle="<->",facecolor='b',edgecolor='b'))
  #ax.arrow(4.5,1.6,6.5,3.5,fc='b',ec='b')
  ax.text(6,2.5,"Margin",color='b')
  plt.xlim((np.min(x1)-lim,np.max(x1)+lim))
  plt.ylim((np.min(x2)-lim,np.max(x2)+lim))
  plt.xlabel(x.columns[0])
  plt.ylabel(x.columns[1])
  if title:
    plt.title(title)
# ---------------------------------------------------
# Plotting function for 2 features and degree = 3
def plot_db_3d(x,y,theta,lim=.1,title=None):
  from mpl_toolkits.mplot3d import Axes3D

  x1 = x.values[:,0]
  x2 = x.values[:,1]
  x3 = x.values[:,2]

  fig=plt.figure()
  fig.set_facecolor('white')
  ax = fig.add_subplot(111, projection='3d')
  xplot = np.arange(np.min(x1)-lim,np.max(x1)+lim,0.1)
  yplot = np.arange(np.min(x2)-lim,np.max(x2)+lim,0.1)
  xplot,yplot = np.meshgrid(xplot,yplot)
  ax.scatter(x1,x2,x3,c=list(y.values),cmap=plt.cm.gray)

  zplot = -1./theta[3]*(theta[0]+theta[1]*xplot+theta[2]*yplot)
  ax.plot_surface(xplot,yplot,zplot,color=(0.7,0.7,0.7),cstride=1,rstride=1,alpha=.2)
  ax.set_xbound(np.min(x1)-lim,np.max(x1)+lim)
  ax.set_ybound(np.min(x2)-lim,np.max(x2)+lim)
  ax.set_zbound(np.min(x3)-lim,np.max(x3)+lim)
  ax.set_xlabel(x.columns[0])
  ax.set_ylabel(x.columns[1])
  ax.set_zlabel(x.columns[2])
  if title:
    ax.set_title(title)
  plt.show()
# ---------------------------------------------------
# Plot for multiclass
def plot_multiclass_2d(x,theta,title=None):
  """
  x is a pandas DataFrame
  """
  x1 = x.values[:,0]
  x2 = x.values[:,1]

  fig = plt.figure()
  fig.set_facecolor('white')
  xplot = np.arange(np.min(x1)-.1,np.max(x1)+.1,0.1)
  plt.plot(x1,x2,'ko')
  for i in range(1,len(theta)+1):
    plt.plot(xplot,1./theta[i][2]*(-theta[i][0]-theta[i][1]*xplot),'r')
  plt.xlim((np.min(x1)-.1,np.max(x1)+.1))
  plt.ylim((np.min(x2)-.1,np.max(x2)+.1))
  plt.xlabel(x.columns[0])
  plt.ylabel(x.columns[1])
  if title:
    plt.title(title)
# ---------------------------------------------------
def plot_multiclass_3d(x,theta,title=None):
  from mpl_toolkits.mplot3d import Axes3D

  x1 = x.values[:,0]
  x2 = x.values[:,1]
  x3 = x.values[:,2]
 
  fig=plt.figure()
  fig.set_facecolor('white')
  ax = fig.add_subplot(111, projection='3d')
  xplot = np.arange(np.min(x1)-.1,np.max(x1)+.1,0.1)
  yplot = np.arange(np.min(x2)-.1,np.max(x2)+.1,0.1)
  xplot,yplot = np.meshgrid(xplot,yplot)
  ax.scatter(x1,x2,x3,c='k')

  for i in range(1,len(theta)+1):
    zplot = -1./theta[i][3]*(theta[i][0]+theta[i][1]*xplot+theta[i][2]*yplot)
    ax.plot_surface(xplot,yplot,zplot,color=(0.7,0.7,0.7),cstride=1,rstride=1,alpha=.2)
  ax.set_xbound(np.min(x1)-.1,np.max(x1)+.1)
  ax.set_ybound(np.min(x2)-.1,np.max(x2)+.1)
  ax.set_zbound(np.min(x3)-.1,np.max(x3)+.1)
  ax.set_xlabel(x.columns[0])
  ax.set_ylabel(x.columns[1])
  ax.set_zlabel(x.columns[2])
  if title:
    ax.set_title(title)
# ---------------------------------------------------
def plot_cost_function(iter,cost):
  """
  Plots the minimum value of the cost function at each iteration
  """
  fig = plt.figure()
  fig.set_facecolor('white')
  plt.plot(range(iter),cost)
  plt.xlabel('Number of iterations')
  plt.ylabel('Cost function')
# ---------------------------------------------------
def plot_deg_vs_lambda(k,degrees,lambdas,Jlist,best_dl,min_cv):
  """
  Plot for diagnosing bias vs variance
  """
  fig = plt.figure()
  fig.set_facecolor('white')
  ax = fig.add_subplot(111, title="Class %d"%k, projection='3d')
  xplot,yplot = np.meshgrid(degrees,lambdas)
  zplot = dic2mat(degrees,lambdas,Jlist,k)
  ax.plot_wireframe(xplot,yplot,zplot,rstride=1,cstride=1)
  cset = ax.contour(xplot,yplot,zplot,zdir='x',offset=0)
  cset = ax.contour(xplot,yplot,zplot,zdir='y',offset=0)
  cset = ax.contour(xplot,yplot,zplot,zdir='z',offset=min_cv-0.1)
  ax.scatter(best_dl[0],best_dl[1],min_cv)
  ax.set_xlabel('Polynomial degree')
  ax.set_xlim(0,degrees[-1])
  ax.set_ylabel('Lambda')
  ax.set_ylim(0,lambdas[-1])
  ax.set_zlabel('Jcv')
  ax.set_zlim(min_cv-0.1,np.max(zplot))
# ---------------------------------------------------
def plot_learning_curves(k,vec,val,c1,c2,xlab):
  """
  Plots learning curves.
  If vec = degrees of polynomial --> diagnoses bias vs variance
  If vec = lambdas (regularization) --> diagnoses bias vs variance
  """
  fig = plt.figure()
  fig.set_facecolor('white')
  plt.plot(range(len(c1)),c1,'b',label='Jcv')
  plt.plot(range(len(c2)),c2,'r',label='Jtrain')
  plt.legend()
  plt.xlabel(xlab)
  plt.ylabel('Error')
  #plt.xticks(vec)
  plt.title('Class %d'%k)
# ---------------------------------------------------
def dic2mat(degrees,lambdas,jlist,i_class):
  """
  Converts a dictionnary into a matrix.
  """
  mat=np.empty([len(lambdas),len(degrees)])
  if not type(degrees) == np.ndarray:
    degrees=np.array(degrees)
  if not type(lambdas) == np.ndarray:
    lambdas=np.array(lambdas)
  for key_d, key_l in tuple(jlist.keys()):
    id=np.where(degrees==key_d)[0]
    il=np.where(lambdas==key_l)[0]
    mat[il,id]=jlist[key_d,key_l][i_class-1]
  return mat
