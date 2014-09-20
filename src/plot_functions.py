#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# ---------------------------------------------------
def plot_hyp_func_1f(x,y,theta,method,threshold=None,x_ok=None,x_bad=None,th_comp=None,cmat_test=[],cmat_svm=[],cmat_train=[]):

  num_t = np.unique(y.NumType.values.ravel())
  num_t = map(int,list(num_t))
  str_t = np.unique(y.Type.values.ravel())
  str_t = map(str,list(str_t))

  fig = plt.figure(figsize=(12,7))
  fig.set_facecolor('white')

  left, bottom = .1, .1
  width, height = .5, .8
  width_h, height_h = .25, .35
  left_h, bottom_h = left+width+.05, bottom+height_h+.1
  rect_1 = [left, bottom, width, height]
  rect_2 = [left_h, bottom_h, width_h, height_h]
  rect_3 = [left_h, bottom, width_h, height_h]

  ax1 = plt.axes(rect_1)
  ax2 = plt.axes(rect_2)
  ax3 = plt.axes(rect_3)

  x1 = x[y.NumType.values.ravel()==num_t[0]].values[:,0]
  x2 = x[y.NumType.values.ravel()==num_t[1]].values[:,0]

  nn,b,p = ax1.hist([x1,x2],25,normed=True,histtype='stepfilled',alpha=.2,color=('b','g'),label=str_t)

  from LR_functions import g
  syn = np.arange(-1,1,0.01)
  hyp = g(theta[1][0]+theta[1][1]*syn)
  norm = np.mean([np.max(nn[0]),np.max(nn[1])])
  ax1.plot(syn,norm*hyp,'y-',lw=2,label='hypothesis')

  if threshold:
    thres = np.ones(len(hyp))*threshold[1]
    imin = np.argmin(np.abs(thres-hyp))
    t = syn[imin]
    ax1.plot([t,t],[0,np.max(nn)],'orange',lw=3.,label='decision')

  if th_comp:
    hyp_svm = g(th_comp[1][0]+th_comp[1][1]*syn)
    ax1.plot(syn,norm*hyp_svm,'magenta',lw=2.)

    thres = np.ones(len(hyp_svm))*.5
    imin = np.argmin(np.abs(thres-hyp_svm))
    t = syn[imin]
    ax1.plot([t,t],[0,np.max(nn)],'purple',lw=3.)

  if x_ok and x_bad:
    nn, b, p = ax1.hist([x_ok,x_bad],25,normed=True,color=('k','r'),histtype='step',fill=False,ls='dashed',lw=2,label=['Test Set'])

  ax1.set_xlim([-1,1])
  ax1.legend(prop={'size':12})
  ax1.set_xlabel(x.columns[0])
  ax1.set_title(x.columns[0])

  from do_classification import plot_confusion_mat, dic_percent
  s = 12
  x_pos = .05
  y_pos = .95
  pas = .04
  if list(cmat_test):
    plot_confusion_mat(cmat_test,str_t,'Test',method,ax=ax2)
    p_test = dic_percent(cmat_test,str_t)
    ax1.text(x_pos,y_pos,method.upper(),size=s,transform=ax1.transAxes,color='orange')
    ax1.text(x_pos,y_pos-pas,"Test : %.2f%%"%p_test['global'],size=s,transform=ax1.transAxes)
    ax1.text(x_pos+.2,y_pos-pas,"(%.2f%%)"%(100-p_test['global']),size=s,color='red',transform=ax1.transAxes)
    
  if list(cmat_train) and not list(cmat_svm):
    plot_confusion_mat(cmat_train,str_t,'Training',method,ax=ax3)
    p_train = dic_percent(cmat_train,str_t)
    ax1.text(x_pos,y_pos-2*pas,"Training : %.2f%%"%p_train['global'],size=s,transform=ax1.transAxes)

  elif list(cmat_train) and list(cmat_svm):
    p_train = dic_percent(cmat_train,str_t)
    ax1.text(x_pos,y_pos-2*pas,"Training : ",size=s,transform=ax1.transAxes)
    ax1.text(x_pos+.15,y_pos-2*pas,"%s %s%%"%(str_t[0],p_train[(str_t[0],0)]),color='b',size=s,transform=ax1.transAxes)
    ax1.text(x_pos+.15,y_pos-3*pas,"%s %s%%"%(str_t[1],p_train[(str_t[1],1)]),color='g',size=s,transform=ax1.transAxes)

    plot_confusion_mat(cmat_svm,str_t,'Test','SVM',ax=ax3)
    p_svm = dic_percent(cmat_svm,str_t)
    ax1.text(x_pos,y_pos-4*pas,"SVM",size=s,transform=ax1.transAxes,color='purple')
    ax1.text(x_pos,y_pos-5*pas,"Test : %.2f%%"%p_svm['global'],size=s,transform=ax1.transAxes)

  elif not list(cmat_train) and list(cmat_svm):
    plot_confusion_mat(cmat_svm,str_t,'Test','SVM',ax=ax3)
    p_svm = dic_percent(cmat_svm,str_t)
    ax1.text(x_pos,y_pos-3*pas,"SVM",size=s,transform=ax1.transAxes,color='purple')
    ax1.text(x_pos,y_pos-4*pas,"Test : %.2f%%"%p_svm['global'],size=s,transform=ax1.transAxes)

  plt.figtext(.1,.93,'(a)')
  plt.figtext(.63,.93,'(b)')
  plt.figtext(.63,.48,'(c)')

# ---------------------------------------------------
def histo_pdfs(x_test,y_test,x_train=None,y_train=None):
  """
  Plot the histograms of the training and test set superimposed with PDFs.
  """
  from scipy.stats.kde import gaussian_kde

  num_t = np.unique(y_test.NumType.values.ravel())
  num_t = map(int,list(num_t))
  str_t = np.unique(y_test.Type.values.ravel())
  str_t = map(str,list(str_t))

  fig = plt.figure()
  fig.set_facecolor('white')

  colors = ('k','w')
  for feat in x_test.columns:
    hist = []
    g = {}

    if y_train:
      c_train = ('b','g')
      hist_train, g_train = [],{}
      lab_tr = []
      mini = np.min([x_test.min()[feat],x_train.min()[feat]])
      maxi = np.max(x_test.max()[feat],x_train.max()[feat])
    else:
      mini = x_test.min()[feat]
      maxi = x_test.max()[feat]

    print mini, maxi
    bins_hist = np.linspace(mini,maxi,25)
    bins = np.linspace(mini,maxi,200)

    for i in num_t:
      index = y_test[y_test.NumType.values==i].index
      x_plot = x_test.reindex(columns=[feat],index=index).values
      hist.append(x_plot)
      kde = gaussian_kde(x_plot.ravel())
      g[i] = kde(bins)
      if y_train:
        lab_tr.append('%s (train)'%str_t[i])
        index = y_train[y_train.NumType.values==i].index
        x_plot = x_train.reindex(columns=[feat],index=index).values
        hist_train.append(x_plot)
        kde = gaussian_kde(x_plot.ravel())
        g_train[i] = kde(bins)
    
    plt.hist(hist,bins=bins_hist,color=colors,normed=1,histtype='stepfilled',alpha=.2,label=str_t)
    if y_train:
      plt.hist(hist_train,bins=bins_hist,color=c_train,normed=1,histtype='stepfilled',alpha=.2,label=lab_tr)

    colors = ('k','y')
    for key in sorted(g):
      plt.plot(bins,g[key],color=colors[key],lw=2.)
      if y_train:
        plt.plot(bins,g_train[key],color=c_train[key],lw=1.,ls='--')
 
    plt.legend(loc=2)
    plt.xlabel(feat)
 
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
def plot_cost_function_iter(iter,cost):
  """
  Plot the minimum value of the cost function at each iteration
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


