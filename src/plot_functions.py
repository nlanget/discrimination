#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# ---------------------------------------------------
# Plotting function for 1 feature
def plot_hyp_func(x,y,syn,hyp):
  """
  Plots the hypothesis function
  """
  fig=plt.figure()
  fig.set_facecolor('white')
  x1 = x[y==0].values[:,0]
  x2 = x[y==1].values[:,0]
  nn, b, p = plt.hist([x1,x2],25,normed=True,histtype='stepfilled',alpha=.2,color=('b','g'),label=['Class 1','Class 2'])
  norm=np.mean([np.max(nn[0]),np.max(nn[1])])
  plt.plot(syn,norm*hyp,'y-',lw=2,label='hypothesis')
  plt.legend()
  plt.xlabel(x.columns[0])
  plt.title('Logistic regression - 1 feature')
# ---------------------------------------------------
# Plotting function for 2 features
def plot_db(x,y,theta,lim=.1,title=None):
  """
  Plots the decision boundary 
  x is of DataFrame type
  y is a np.array
  """
  x1 = x.values[:,0]
  x2 = x.values[:,1]

  fig=plt.figure()
  fig.set_facecolor('white')
  xplot=np.arange(np.min(x1)-lim,np.max(x1)+lim,0.1)
  plt.scatter(x1,x2,c=y,cmap=plt.cm.gray)
  plt.plot(xplot,1./theta[2]*(-theta[0]-theta[1]*xplot),'r')
  plt.xlim((np.min(x1)-lim,np.max(x1)+lim))
  plt.ylim((np.min(x2)-lim,np.max(x2)+lim))
  plt.xlabel(x.columns[0])
  plt.ylabel(x.columns[1])
  if title:
    plt.title(title)
# ---------------------------------------------------
# Plotting function for 3 features
def plot_db_3d(x,y,theta,lim=.1,title=None):
  from mpl_toolkits.mplot3d import Axes3D

  x1 = x.values[:,0]
  x2 = x.values[:,1]
  x3 = x.values[:,2]

  fig=plt.figure()
  fig.set_facecolor('white')
  ax = fig.add_subplot(111, projection='3d')
  xplot=np.arange(np.min(x1)-lim,np.max(x1)+lim,0.1)
  yplot=np.arange(np.min(x2)-lim,np.max(x2)+lim,0.1)
  xplot,yplot=np.meshgrid(xplot,yplot)
  ax.scatter(x1,x2,x3,c=y,cmap=plt.cm.gray)

  zplot=-1./theta[3]*(theta[0]+theta[1]*xplot+theta[2]*yplot)
  ax.plot_surface(xplot,yplot,zplot,color=(0.7,0.7,0.7),cstride=1,rstride=1,alpha=.2)
  ax.set_xbound(np.min(x1)-lim,np.max(x1)+lim)
  ax.set_ybound(np.min(x2)-lim,np.max(x2)+lim)
  ax.set_zbound(np.min(x3)-lim,np.max(x3)+lim)
  ax.set_xlabel(x.columns[0])
  ax.set_ylabel(x.columns[1])
  ax.set_zlabel(x.columns[2])
  if title:
    ax.set_title(title)
# ---------------------------------------------------
# Plot for multiclass
def plot_multiclass_2d(x,theta,title=None):
  """
  x if of DataFrame type
  """
  x1=x.values[:,0]
  x2=x.values[:,1]

  fig=plt.figure()
  fig.set_facecolor('white')
  xplot=np.arange(np.min(x1)-.1,np.max(x1)+.1,0.1)
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
  xplot=np.arange(np.min(x1)-.1,np.max(x1)+.1,0.1)
  yplot=np.arange(np.min(x2)-.1,np.max(x2)+.1,0.1)
  xplot,yplot=np.meshgrid(xplot,yplot)
  ax.scatter(x1,x2,x3,c='k')

  for i in range(1,len(theta)+1):
    zplot=-1./theta[i][3]*(theta[i][0]+theta[i][1]*xplot+theta[i][2]*yplot)
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
  from logistic_reg import dic2mat
  fig = plt.figure()
  fig.set_facecolor('white')
  ax = fig.add_subplot(111, title="Class %d"%k, projection='3d')
  xplot,yplot=np.meshgrid(degrees,lambdas)
  zplot=dic2mat(degrees,lambdas,Jlist,k)
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

