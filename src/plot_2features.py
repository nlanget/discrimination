import sys, os, glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import matplotlib.mlab as mlab

# *******************************************************************************

def class_2c_2f(thetas,threshold):
  """
  Computes the classification map for a discrimination problem 
  with 2 classes and 2 features.
  thetas is a dictionary of length = 1 (2 classes) 
  and each independent theta vector is of length = 3 (2 features).
  """
  from LR_functions import g
  pas = .01
  x1 = np.arange(-1,1,pas)
  x2 = np.arange(-1,1,pas)
  x1, x2 = np.meshgrid(x1,x2)

  probas = g(thetas[1][0]+thetas[1][1]*x1+thetas[1][2]*x2)
  classi = probas.copy()
  classi[classi>=threshold[1]] = 1
  classi[classi!=1] = 0

  return x1, x2, probas, classi

# *******************************************************************************

def class_multi_2f(thetas):
  """
  Computes the classification map for a discrimination problem 
  with more than 2 classes and only 2 features.
  thetas is a dictionary of length = 3 (3 classes) 
  and each independent theta vector is of length = 3 (2 features).
  """
  from LR_functions import g
  pas = .01
  x1 = np.arange(-1,1,pas)
  x2 = np.arange(-1,1,pas)
  x1, x2 = np.meshgrid(x1,x2)

  probas = []
  for i in sorted(thetas):
    probas.append(g(thetas[i][0]+thetas[i][1]*x1+thetas[i][2]*x2))
  probas = np.array(probas)

  return x1, x2, np.max(probas,axis=0), np.argmax(probas,axis=0)

# *******************************************************************************

def plot_2f(theta,rate,t,method,x_train,x_test,y_test,NB_test):
    """
    Plots decision boundaries for a discrimination problem with 
    2 features.
    """

    if len(theta) > 2:
      NB_class = len(theta)
      x_vec, y_vec, proba, map = class_multi_2f(theta)

    elif len(theta) == 1:
      NB_class = 2
      x_vec, y_vec, proba, map = class_2c_2f(theta,t)

    #### PLOT ####
    nullfmt = NullFormatter()

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left+width+0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    # start with a rectangular Figure
    fig = plt.figure(1, figsize=(8,8))
    fig.set_facecolor('white')

    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    # No labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    # Scatter plot:
    from LR_functions import normalize
    x_train, x_test = normalize(x_train,x_test)
    x = x_test['x1']
    y = x_test['x2']
    axScatter.scatter(x,y,c=list(y_test.TypeNum.values),cmap=plt.cm.gray_r)

    # Determine nice limits by hand
    binwidth = 0.025
    xymax = np.max( [np.max(np.fabs(x)), np.max(np.fabs(y))] )
    lim_plot = ( int(xymax/binwidth) + 1) * binwidth
    bins = np.arange(-lim_plot, lim_plot + binwidth, binwidth)

    # Plot decision boundaries
    for i in sorted(theta):
      db = -1./theta[i][2]*(theta[i][0]+np.log((1-t[i])/t[i])+theta[i][1]*x_vec[0])
      axScatter.plot(x_vec[0],db,lw=2.,c='pink')
      axScatter.contourf(x_vec, y_vec, map, cmap=plt.cm.gray, alpha=0.2)
    axScatter.legend(['%s (%.1f%%)'%(method,rate[1])],loc=4)

    axScatter.set_xlim((-lim_plot, lim_plot))
    axScatter.set_ylim((-lim_plot, lim_plot))

    # Plot histograms and PDFs
    lim = 0
    x_hist, y_hist = [],[]
    g_x, g_y = {}, {}

    if NB_class > 2:
      colors = ('w','gray','k')
    elif NB_class == 2:
      colors = ('w','k')
    for i in range(NB_class):
      x_hist.append(x[lim:lim+NB_test[i]])
      y_hist.append(y[lim:lim+NB_test[i]])
      g_x[i] = mlab.normpdf(bins, np.mean(x[lim:lim+NB_test[i]]), np.std(x[lim:lim+NB_test[i]]))
      g_y[i] = mlab.normpdf(bins, np.mean(y[lim:lim+NB_test[i]]), np.std(y[lim:lim+NB_test[i]]))
      axHisty.hist(y[lim:lim+NB_test[i]],bins=bins,color=colors[i],normed=1,orientation='horizontal',histtype='stepfilled',alpha=.5)
      lim = lim + NB_test[i]
    axHistx.hist(x_hist,bins=bins,color=colors,normed=1,histtype='stepfilled',alpha=.5)

    if NB_class > 2:
      colors = ('r','orange','y')
    elif NB_class == 2:
      colors = ('r','y')
    for key in sorted(g_x):
      axHistx.plot(bins,g_x[key],color=colors[key],lw=2.)
      axHisty.plot(g_y[key],bins,color=colors[key],lw=2.)

    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())

    axScatter.set_xlabel(x_test.columns[0])
    axScatter.set_ylabel(x_test.columns[1])

# *******************************************************************************

def plot_2f_variability(theta,rate,t,method,x_train,x_test,y_test,NB_test):
    """
    Plots decision boundaries for a discrimination problem with 
    2 classes and 2 features.
    """

    if len(theta[0]) > 2:
      NB_class = len(theta[0])
      x_vec, y_vec, proba, map = class_multi_2f(theta[0])

    elif len(theta[0]) == 1:
      NB_class = 2
      x_vec, y_vec, proba, map = class_2c_2f(theta[0],t[0])


    #### PLOT ####
    nullfmt = NullFormatter()

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left+width+0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    # start with a rectangular Figure
    fig = plt.figure(1, figsize=(8,8))
    fig.set_facecolor('white')

    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    # No labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    # Scatter plot:
    from LR_functions import normalize
    x_train, x_test = normalize(x_train,x_test)
    x = x_test['x1']
    y = x_test['x2']
    axScatter.scatter(x,y,c=list(y_test.TypeNum.values),cmap=plt.cm.gray_r)

    # Determine nice limits by hand
    binwidth = 0.025
    xymax = np.max( [np.max(np.fabs(x)), np.max(np.fabs(y))] )
    lim_plot = ( int(xymax/binwidth) + 1) * binwidth
    bins = np.arange(-lim_plot, lim_plot + binwidth, binwidth)

    # Plot decision boundaries
    # VARIABILITY OF THE LR DECISION BOUNDARY
    if len(theta) > 1:
      rates = []
      for i in range(len(theta)):
        db = -1./theta[i][1][2]*(theta[i][1][0]+np.log((1-t[i][1])/t[i][1])+theta[i][1][1]*x_vec[0])
        axScatter.plot(x_vec[0],db,lw=1.,c=(0,0.1*i,1))
        rates.append(rate[i][1])

      imax = np.argmax(rates)
      db = -1./theta[imax][1][2]*(theta[imax][1][0]+np.log((1-t[imax][1])/t[imax][1])+theta[imax][1][1]*x_vec[0])
      axScatter.plot(x_vec[0],db,lw=3.,c='midnightblue')
      x_vec, y_vec, proba, map = class_2c_2f(theta[imax],t[imax])
      axScatter.contourf(x_vec,y_vec,map,cmap=plt.cm.gray,alpha=.2)

      axScatter.text(0.6*lim_plot,-0.9*lim_plot,r'%.1f$\pm$%.1f%%'%(np.mean(rates),np.std(rates)))

    # VARIABILITY WITH THE THRESHOLD
    else:
      #for thres in np.arange(0,1,.1):
      #  db = -1./theta[0][1][2]*(theta[0][1][0]+np.log((1-thres)/thres)+theta[0][1][1]*x_vec[0])
      #  axScatter.plot(x_vec[0],db,lw=1.,c=(0,thres,1))
      from LR_functions import g
      blue_scale = []
      for i in range(10):
        blue_scale.append((0,i*0.1,1))
      CS = axScatter.contour(x_vec,y_vec,proba,10,colors=blue_scale)
      axScatter.clabel(CS, inline=1, fontsize=10)

      if NB_class == 2:
        db = -1./theta[0][1][2]*(theta[0][1][0]+np.log((1-t[0][1])/t[0][1])+theta[0][1][1]*x_vec[0])
        axScatter.plot(x_vec[0],db,lw=3.,c='midnightblue')
      axScatter.contourf(x_vec,y_vec,map,cmap=plt.cm.gray,alpha=.2)

      axScatter.text(0.6*lim_plot,-0.9*lim_plot,'LR (%.1f%%)'%rate[0][1])
      axScatter.text(0.6*lim_plot,-0.8*lim_plot,'t = %.1f'%t[0][1])

    axScatter.set_xlim((-lim_plot, lim_plot))
    axScatter.set_ylim((-lim_plot, lim_plot))

    # Plot histograms and PDFs
    lim = 0
    x_hist, y_hist = [],[]
    g_x, g_y = {}, {}

    if NB_class > 2:
      colors = ('w','gray','k')
    elif NB_class == 2:
      colors = ('w','k')
    for i in range(NB_class):
      x_hist.append(x[lim:lim+NB_test[i]])
      y_hist.append(y[lim:lim+NB_test[i]])
      g_x[i] = mlab.normpdf(bins, np.mean(x[lim:lim+NB_test[i]]), np.std(x[lim:lim+NB_test[i]]))
      g_y[i] = mlab.normpdf(bins, np.mean(y[lim:lim+NB_test[i]]), np.std(y[lim:lim+NB_test[i]]))
      axHisty.hist(y[lim:lim+NB_test[i]],bins=bins,color=colors[i],normed=1,orientation='horizontal',histtype='stepfilled',alpha=.5)
      lim = lim + NB_test[i]
    axHistx.hist(x_hist,bins=bins,color=colors,normed=1,histtype='stepfilled',alpha=.5)

    if NB_class > 2:
      colors = ('r','orange','y')
    elif NB_class == 2:
      colors = ('r','y')
    for key in sorted(g_x):
      axHistx.plot(bins,g_x[key],color=colors[key],lw=2.)
      axHisty.plot(g_y[key],bins,color=colors[key],lw=2.)

    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())

    axScatter.set_xlabel(x_test.columns[0])
    axScatter.set_ylabel(x_test.columns[1])

# *******************************************************************************

def plot_2f_superimposed(theta_lr,rate_lr,t_lr,theta_svm,rate_svm,t_svm,x_train,x_test,y_test,NB_test):
    """
    Compares decision boundaries of the SVM and LR.
    Map the LR results.
    """

    if len(theta_lr) > 2:
      NB_class = len(theta_lr)
      x_vec, y_vec, proba, map = class_multi_2f(theta_lr)

    elif len(theta_lr) == 1:
      NB_class = 2
      x_vec, y_vec, proba, map = class_2c_2f(theta_lr,t_lr)

    #### PLOT ####
    nullfmt = NullFormatter()

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left+width+0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    # start with a rectangular Figure
    fig = plt.figure(1, figsize=(8,8))
    fig.set_facecolor('white')

    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    # No labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    # Scatter plot:
    from LR_functions import normalize
    x_train, x_test = normalize(x_train,x_test)
    x = x_test['x1']
    y = x_test['x2']
    axScatter.scatter(x,y,c=list(y_test.TypeNum.values),cmap=plt.cm.gray_r)

    # Determine nice limits by hand
    binwidth = 0.025
    xymax = np.max( [np.max(np.fabs(x)), np.max(np.fabs(y))] )
    lim_plot = ( int(xymax/binwidth) + 1) * binwidth
    bins = np.arange(-lim_plot, lim_plot + binwidth, binwidth)

    # Plot decision boundaries
    for i in sorted(theta_lr):
      db_lr = -1./theta_lr[i][2]*(theta_lr[i][0]+np.log((1-t_lr[i])/t_lr[i])+theta_lr[i][1]*x_vec[0])
      axScatter.plot(x_vec[0],db_lr,lw=2.,c='b')

      db_svm = -1./theta_svm[i][2]*(theta_svm[i][0]+np.log((1-t_svm[i])/t_svm[i])+theta_svm[i][1]*x_vec[0])
      axScatter.plot(x_vec[0],db_svm,lw=2.,c='c')

      axScatter.contourf(x_vec, y_vec, map, cmap=plt.cm.gray, alpha=0.2)
    axScatter.legend(['LR (%.1f%%)'%rate_lr[1],'SVM (%.1f%%)'%rate_svm[1]],loc=4)

    axScatter.set_xlim((-lim_plot, lim_plot))
    axScatter.set_ylim((-lim_plot, lim_plot))

    # Plot histograms and PDFs
    lim = 0
    x_hist, y_hist = [],[]
    g_x, g_y = {}, {}

    if NB_class > 2:
      colors = ('w','gray','k')
    elif NB_class == 2:
      colors = ('w','k')
    for i in range(NB_class):
      x_hist.append(x[lim:lim+NB_test[i]])
      y_hist.append(y[lim:lim+NB_test[i]])
      g_x[i] = mlab.normpdf(bins, np.mean(x[lim:lim+NB_test[i]]), np.std(x[lim:lim+NB_test[i]]))
      g_y[i] = mlab.normpdf(bins, np.mean(y[lim:lim+NB_test[i]]), np.std(y[lim:lim+NB_test[i]]))
      axHisty.hist(y[lim:lim+NB_test[i]],bins=bins,color=colors[i],normed=1,orientation='horizontal',histtype='stepfilled',alpha=.5)
      lim = lim + NB_test[i]
    axHistx.hist(x_hist,bins=bins,color=colors,normed=1,histtype='stepfilled',alpha=.5)

    if NB_class > 2:
      colors = ('r','orange','y')
    elif NB_class == 2:
      colors = ('r','y')
    for key in sorted(g_x):
      axHistx.plot(bins,g_x[key],color=colors[key],lw=2.)
      axHisty.plot(g_y[key],bins,color=colors[key],lw=2.)

    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())

    axScatter.set_xlabel(x_test.columns[0])
    axScatter.set_ylabel(x_test.columns[1])

