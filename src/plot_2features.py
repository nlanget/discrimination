import sys, os, glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import matplotlib.mlab as mlab
from scipy.stats.kde import gaussian_kde

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

def plot_histos_and_pdfs_gauss(axHistx,axHisty,bins_x,bins_y,x_test,y_test,x_train=None,y_train=None):
    """
    Histograms and gaussian PDFs
    """
    x_hist, y_hist = [],[]
    g_x, g_y = {}, {}
    if y_train:
      g_x_train, g_y_train = {}, {}

    feat_1 = x_test.columns[0]
    feat_2 = x_test.columns[1] 
    NB_class = len(np.unique(y_test.Type.values))

    if NB_class > 2:
      colors = ('k','gray','w')
    elif NB_class == 2:
      colors = ('k','w')
    for i in range(NB_class):
      index = y_test[y_test.NumType.values==i].index
      x1 = x_test.reindex(columns=[feat_1],index=index).values
      x2 = x_test.reindex(columns=[feat_2],index=index).values
      x_hist.append(x1)
      y_hist.append(x2)
      g_x[i] = mlab.normpdf(bins_x, np.mean(x1), np.std(x1))
      g_y[i] = mlab.normpdf(bins_y, np.mean(x2), np.std(x2))
      axHisty.hist(x2,bins=bins_y,color=colors[i],normed=1,orientation='horizontal',histtype='stepfilled',alpha=.5)
      if y_train:
        index = y_train[y_train.NumType.values==i].index
        x1 = x_train.reindex(columns=[feat_1],index=index).values
        x2 = x_train.reindex(columns=[feat_2],index=index).values
        g_x_train[i] = mlab.normpdf(bins_x, np.mean(x1), np.std(x1))
        g_y_train[i] = mlab.normpdf(bins_y, np.mean(x2), np.std(x2))

    axHistx.hist(x_hist,bins=bins_x,color=colors,normed=1,histtype='stepfilled',alpha=.5)

    if NB_class > 2:
      colors = ('y','orange','r')
    elif NB_class == 2:
      colors = ('y','r')
    for key in sorted(g_x):
      axHistx.plot(bins_x,g_x[key],color=colors[key],lw=2.)
      axHisty.plot(g_y[key],bins_y,color=colors[key],lw=2.)
      if y_train:
        axHistx.plot(bins_x,g_x_train[key],color=colors[key],lw=1.,ls='--')
        axHisty.plot(g_y_train[key],bins_y,color=colors[key],lw=1.,ls='--')

# *******************************************************************************

def plot_histos_and_pdfs_kde(axHistx,axHisty,bins_x,bins_y,x_test,y_test,x_train=None,y_train=None):
    """
    Histograms and KDE PDFs
    """
    x_hist, y_hist = [],[]
    g_x, g_y = {}, {}
    if y_train:
      g_x_train, g_y_train = {}, {}

    feat_1 = x_test.columns[0]
    feat_2 = x_test.columns[1] 
    NB_class = len(np.unique(y_test.Type.values))

    if NB_class > 2:
      colors = ('k','gray','w')
    elif NB_class == 2:
      colors = ('k','w')
    for i in range(NB_class):
      index = y_test[y_test.NumType.values==i].index
      x1 = x_test.reindex(columns=[feat_1],index=index).values
      x2 = x_test.reindex(columns=[feat_2],index=index).values
      x_hist.append(x1)
      y_hist.append(x2)
      kde = gaussian_kde(x1.ravel())
      g_x[i] = kde(bins_x)
      kde = gaussian_kde(x2.ravel())
      g_y[i] = kde(bins_y)
      axHisty.hist(x2,bins=bins_y,color=colors[i],normed=1,orientation='horizontal',histtype='stepfilled',alpha=.5)
      if y_train:
        index = y_train[y_train.NumType.values==i].index
        x1 = x_train.reindex(columns=[feat_1],index=index).values
        x2 = x_train.reindex(columns=[feat_2],index=index).values
        kde = gaussian_kde(x1.ravel())
        g_x_train[i] = kde(bins_x)
        kde = gaussian_kde(x2.ravel())
        g_y_train[i] = kde(bins_y)
    axHistx.hist(x_hist,bins=bins_x,color=colors,normed=1,histtype='stepfilled',alpha=.5)

    if NB_class > 2:
      colors = ('y','orange','r')
    elif NB_class == 2:
      colors = ('y','r')
    for key in sorted(g_x):
      axHistx.plot(bins_x,g_x[key],color=colors[key],lw=2.)
      axHisty.plot(g_y[key],bins_y,color=colors[key],lw=2.)
      if y_train:
        axHistx.plot(bins_x,g_x_train[key],color=colors[key],lw=1.,ls='--')
        axHisty.plot(g_y_train[key],bins_y,color=colors[key],lw=1.,ls='--')

# *******************************************************************************

def plot_2f(theta,rate,t,method,x_train,x_test,y_test,th_comp=None,t_comp=None,p=None):
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

    feat_1 = x_test.columns[0]
    feat_2 = x_test.columns[1]
    x = x_test[feat_1]
    y = x_test[feat_2]
    axScatter.scatter(x,y,c=list(y_test.NumType.values),cmap=plt.cm.gray)

    # Determine nice limits by hand
    binwidth = 0.025
    maxi_1 = np.max([np.max(x_test[feat_1]),np.max(x_train[feat_1])])
    mini_1 = np.min([np.min(x_test[feat_1]),np.min(x_train[feat_1])])
    lim_plot_sup_1 = (int(maxi_1/binwidth)+2)*binwidth
    lim_plot_inf_1 = (int(mini_1/binwidth)-2)*binwidth
    if lim_plot_sup_1 > 1: 
      lim_plot_sup_1 = 1
    if lim_plot_inf_1 < -1:
      lim_plot_inf_1 = -1
    bins_1 = np.arange(lim_plot_inf_1, lim_plot_sup_1 + binwidth, binwidth)

    maxi_2 = np.max([np.max(x_test[feat_2]),np.max(x_train[feat_2])])
    mini_2 = np.min([np.min(x_test[feat_2]),np.min(x_train[feat_2])])
    lim_plot_sup_2 = (int(maxi_2/binwidth)+2)*binwidth
    lim_plot_inf_2 = (int(mini_2/binwidth)-2)*binwidth
    if lim_plot_sup_2 > 1: 
      lim_plot_sup_2 = 1
    if lim_plot_inf_2 < -1:
      lim_plot_inf_2 = -1
    bins_2 = np.arange(lim_plot_inf_2, lim_plot_sup_2 + binwidth, binwidth)

    # Plot decision boundaries
    if th_comp and t_comp:
      colors = ['b','c']
    else:
      colors = ['pink']
    for i in sorted(theta):
      db = -1./theta[i][2]*(theta[i][0]+np.log((1-t[i])/t[i])+theta[i][1]*x_vec[0])
      axScatter.plot(x_vec[0],db,lw=2.,c=colors[0])
      if th_comp and t_comp:
        db = -1./th_comp[i][2]*(th_comp[i][0]+np.log((1-t_comp[i])/t_comp[i])+th_comp[i][1]*x_vec[0])
        axScatter.plot(x_vec[0],db,lw=3.,c=colors[1])

    axScatter.contourf(x_vec, y_vec, map, cmap=plt.cm.gray, alpha=0.2)

    label = ['%s (%.2f%%)'%(method.upper(),rate['global'])]
    if th_comp and t_comp:
      label.append('SVM (%.2f%%)'%p[1])
    axScatter.legend(label,loc=4,prop={'size':14})

    axScatter.set_xlim((lim_plot_inf_1, lim_plot_sup_1))
    axScatter.set_ylim((lim_plot_inf_2, lim_plot_sup_2))

    axScatter.set_xlabel(feat_1)
    axScatter.set_ylabel(feat_2)

    # Plot histograms and PDFs
    plot_histos_and_pdfs_kde(axHistx,axHisty,bins_1,bins_2,x_test,y_test)
    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())

    # Display success rates
    pos_y_ini = .95
    pas = .025
    plt.figtext(.78,pos_y_ini,'Test set %s'%method.upper())
    for key in sorted(rate):
      if key != 'global':
        cl, icl = key[0], key[1]
        plt.figtext(.78,pos_y_ini-(icl+1)*pas,'%s (%d) : %s%%'%(cl,icl,rate[(cl,icl)]))
    if th_comp and t_comp:
      pos_y = pos_y_ini-.04
      plt.figtext(.78,pos_y-NB_class*pas,'Test set SVM')
      for key in sorted(p):
        if key != 'global':
          cl, icl = key[0], key[1]
          plt.figtext(.78,pos_y-NB_class*pas-(icl+1)*pas,'%s (%d) : %s%%'%(cl,icl,p[(cl,icl)]))


# *******************************************************************************

def plot_2f_synthetics(out,x_train,x_test,y_test,y_train=None,out_comp=None,map_nl=None):
    """
    For synthetic tests.
    """
    theta = out['thetas']
    rate = out['rate_test']
    t = out['threshold']
    method = out['method']

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

    feat_1 = x_test.columns[0]
    feat_2 = x_test.columns[1]
    x = x_test[feat_1]
    y = x_test[feat_2]
    axScatter.scatter(x,y,c=list(y_test.NumType.values),cmap=plt.cm.gray)
    if y_train:
      axScatter.scatter(x_train[feat_1],x_train[feat_2],c=list(y_train.NumType.values),cmap=plt.cm.YlOrRd)

    # Determine nice limits by hand
    binwidth = 0.025
    xymax = np.max( [np.max(np.fabs(x)), np.max(np.fabs(y))] )
    lim_plot = ( int(xymax/binwidth) + 1) * binwidth
    bins = np.arange(-lim_plot, lim_plot + binwidth, binwidth)

    # Plot decision boundaries
    if out_comp:
      colors = ['b','c']
    else:
      colors = ['pink']
    for i in sorted(theta):
      db = -1./theta[i][2]*(theta[i][0]+np.log((1-t[i])/t[i])+theta[i][1]*x_vec[0])
      axScatter.plot(x_vec[0],db,lw=2.,c=colors[0])
      if out_comp:
        th_comp = out_comp['thetas']
        t_comp = out_comp['threshold']
        db = -1./th_comp[i][2]*(th_comp[i][0]+np.log((1-t_comp[i])/t_comp[i])+th_comp[i][1]*x_vec[0])
        axScatter.plot(x_vec[0],db,lw=3.,c=colors[1])

    if map_nl:
      axScatter.contourf(x_vec, y_vec, map_nl['map'], cmap=plt.cm.gray, alpha=0.3)
    else:
      axScatter.contourf(x_vec, y_vec, map, cmap=plt.cm.gray, alpha=0.3)

    label = ['%s (%.2f%%)'%(method.upper(),rate['global'])]
    if out_comp:
      label.append('%s (%.2f%%)'%(out_comp['method'].upper(),out_comp['rate_test']['global']))
    if map_nl:
      label.append('%s (%.2f%%)'%(map_nl['method'].upper(),map_nl['rate_test']['global']))
    if len(label) == 1:
      s = 14
    else:
      s = 11
    axScatter.legend(label,loc=4,prop={'size':s})

    axScatter.set_xlim((-lim_plot, lim_plot))
    axScatter.set_ylim((-lim_plot, lim_plot))
    axScatter.set_xlabel(feat_1)
    axScatter.set_ylabel(feat_2)

    # Plot histograms and PDFs
    plot_histos_and_pdfs_gauss(axHistx,axHisty,bins,bins,x_test,y_test,x_train=x_train,y_train=y_train)
    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())

    # Display success rates
    if not (out_comp and map_nl):
      pos_y_ini = .95
      pas = .025
      plt.figtext(.78,pos_y_ini,'Test set %s'%method.upper())
      for key in sorted(rate):
        if key != 'global':
          cl, icl = key[0], key[1]
          plt.figtext(.78,pos_y_ini-(icl+1)*pas,'%s (%d) : %s%%'%(cl,icl,rate[(cl,icl)]))
      if out_comp or map_nl:
        if out_comp:
          sup = out_comp
        elif map_nl:
          sup = map_nl
        pos_y = pos_y_ini-.04
        plt.figtext(.78,pos_y-NB_class*pas,'Test set %s'%sup['method'].upper())
        p = sup['rate_test']
        for key in sorted(p):
          if key != 'global':
            cl, icl = key[0], key[1]
            plt.figtext(.78,pos_y-NB_class*pas-(icl+1)*pas,'%s (%d) : %s%%'%(cl,icl,p[(cl,icl)]))

    if out_comp and map_nl:
      if NB_class == 2:
        pos_y_ini = .95
        pas = .02
      else:
        pos_y_ini = .96
        pas = .015
      s = 10
      plt.figtext(.78,pos_y_ini,'Test set %s'%method.upper(),size=s+1)
      for key in sorted(rate):
        if key != 'global':
          cl, icl = key[0], key[1]
          plt.figtext(.78,pos_y_ini-(icl+1)*pas,'%s (%d) : %s%%'%(cl,icl,rate[(cl,icl)]),size=s)

      pos_y = pos_y_ini-.03
      pos_y_ini = pos_y-NB_class*pas
      plt.figtext(.78,pos_y_ini,'Test set %s'%out_comp['method'].upper(),size=s+1)
      p = out_comp['rate_test']
      for key in sorted(p):
        if key != 'global':
          cl, icl = key[0], key[1]
          plt.figtext(.78,pos_y_ini-(icl+1)*pas,'%s (%d) : %s%%'%(cl,icl,p[(cl,icl)]),size=s)

      pos_y = pos_y_ini-.03
      pos_y_ini = pos_y-NB_class*pas
      plt.figtext(.78,pos_y_ini,'Test set %s'%map_nl['method'].upper(),size=s+1)
      p = map_nl['rate_test']
      for key in sorted(p):
        if key != 'global':
          cl, icl = key[0], key[1]
          plt.figtext(.78,pos_y_ini-(icl+1)*pas,'%s (%d) : %s%%'%(cl,icl,p[(cl,icl)]),size=s)


# *******************************************************************************

def plot_2f_synthetics_nonlinear(out,x_train,x_test,y_test,y_train=None):
    """
    For synthetic tests. Non linear decision boundary.
    """
    NB_class = len(np.unique(y_test.Type.values))
    map = out['map']
    rate = out['rate_test']
    method = out['method']

    pas = .01
    x_vec = np.arange(-1,1,pas)
    y_vec = np.arange(-1,1,pas)
    x_vec, y_vec = np.meshgrid(x_vec,y_vec)

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

    feat_1 = x_test.columns[0]
    feat_2 = x_test.columns[1]
    x = x_test[feat_1]
    y = x_test[feat_2]
    axScatter.scatter(x,y,c=list(y_test.NumType.values),cmap=plt.cm.gray)
    if y_train:
      axScatter.scatter(x_train[feat_1],x_train[feat_2],c=list(y_train.NumType.values),cmap=plt.cm.YlOrRd)

    # Determine nice limits by hand
    binwidth = 0.025
    xymax = np.max( [np.max(np.fabs(x)), np.max(np.fabs(y))] )
    lim_plot = ( int(xymax/binwidth) + 1) * binwidth
    bins = np.arange(-lim_plot, lim_plot + binwidth, binwidth)

    # Plot decision boundaries
    axScatter.contourf(x_vec, y_vec, map, cmap=plt.cm.gray, alpha=0.3)
    label = ['%s (%.2f%%)'%(method.upper(),rate['global'])]
    axScatter.legend(label,loc=4,prop={'size':14})
    axScatter.set_xlim((-lim_plot, lim_plot))
    axScatter.set_ylim((-lim_plot, lim_plot))

    axScatter.set_xlabel(feat_1)
    axScatter.set_ylabel(feat_2)

    # Plot histograms and PDFs
    plot_histos_and_pdfs_gauss(axHistx,axHisty,bins,bins,x_test,y_test,x_train=x_train,y_train=y_train)
    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())

    pos_y_ini = .95
    pas = .025
    plt.figtext(.78,pos_y_ini,'Test set %s'%method.upper())
    for key in sorted(rate):
      if key != 'global':
        cl, icl = key[0], key[1]
        plt.figtext(.78,pos_y_ini-(icl+1)*pas,'%s (%d) : %s%%'%(cl,icl,rate[(cl,icl)]))

# *******************************************************************************

def plot_2f_synth_var(out,x_train,x_test,y_test):
    """
    Plots decision boundaries for a discrimination problem with 
    2 classes and 2 features.
    """
    theta_first = out[0]['thetas']
    rate_first = out[0]['rate_test']
    t_first = out[0]['threshold']
    method = out[0]['method']

    if len(theta_first) > 2:
      NB_class = len(theta_first)
      x_vec, y_vec, proba, map = class_multi_2f(theta_first)

    elif len(theta_first) == 1:
      NB_class = 2
      x_vec, y_vec, proba, map = class_2c_2f(theta_first,t_first)


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

    feat_1 = x_test.columns[0]
    feat_2 = x_test.columns[1]
    x = x_test[feat_1]
    y = x_test[feat_2]
    axScatter.scatter(x,y,c=list(y_test.NumType.values),cmap=plt.cm.gray)

    # Determine nice limits by hand
    binwidth = 0.025
    xymax = np.max( [np.max(np.fabs(x)), np.max(np.fabs(y))] )
    lim_plot = ( int(xymax/binwidth) + 1) * binwidth
    bins = np.arange(-lim_plot, lim_plot + binwidth, binwidth)

    # Plot decision boundaries
    # VARIABILITY OF THE LR DECISION BOUNDARY
    if len(out) > 1:
      rates = []
      for i in sorted(out):
        theta = out[i]['thetas']
        t = out[i]['threshold']
        db = -1./theta[1][2]*(theta[1][0]+np.log((1-t[1])/t[1])+theta[1][1]*x_vec[0])
        axScatter.plot(x_vec[0],db,lw=1.,c=(0,0.1*i,1))
        rates.append(out[i]['rate_test']['global'])

      imax = np.argmax(rates)
      theta = out[imax]['thetas']
      t = out[imax]['threshold']
      db = -1./theta[1][2]*(theta[1][0]+np.log((1-t[1])/t[1])+theta[1][1]*x_vec[0])
      axScatter.plot(x_vec[0],db,lw=3.,c='midnightblue')
      x_vec, y_vec, proba, map = class_2c_2f(theta,t)
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

      theta = out[0]['thetas']
      t = out[0]['threshold']
      rate = out[0]['rate_test']
      if NB_class == 2:
        db = -1./theta[1][2]*(theta[1][0]+np.log((1-t[1])/t[1])+theta[1][1]*x_vec[0])
        axScatter.plot(x_vec[0],db,lw=3.,c='midnightblue')
      axScatter.contourf(x_vec,y_vec,map,cmap=plt.cm.gray,alpha=.2)

      axScatter.text(0.6*lim_plot,-0.9*lim_plot,'LR (%.1f%%)'%rate['global'])
      axScatter.text(0.6*lim_plot,-0.8*lim_plot,'t = %.1f'%t[1])

    axScatter.set_xlim((-lim_plot, lim_plot))
    axScatter.set_ylim((-lim_plot, lim_plot))

    axScatter.set_xlabel(feat_1)
    axScatter.set_ylabel(feat_2)

    # Plot histograms and PDFs
    plot_histos_and_pdfs_gauss(axHistx,axHisty,bins,bins,x_test,y_test,x_train=x_train,y_train=y_train)
    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())

    pos_y_ini = .95
    pas = .025
    plt.figtext(.78,pos_y_ini,'Test set %s'%method.upper())
    for key in sorted(rate):
      if key != 'global':
        cl, icl = key[0], key[1]
        plt.figtext(.78,pos_y_ini-(icl+1)*pas,'%s (%d) : %s%%'%(cl,icl,rate[(cl,icl)]))

# *******************************************************************************

def plot_2f_all(theta,t,rate,method,x_train,y_train,x_test,y_test,x_bad,str_t,p_train=None,th_comp=None,t_comp=None,p=None):
    """
    Plots decision boundaries for a discrimination problem with 
    2 features.
    Superimposed with scatter plots of both training and test sets.
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
    feat_1 = x_test.columns[0]
    feat_2 = x_test.columns[1]
    axScatter.scatter(x_test[feat_1],x_test[feat_2],c=list(y_test.NumType.values),cmap=plt.cm.gray,alpha=.2)
    axScatter.scatter(x_train[feat_1],x_train[feat_2],c=list(y_train.NumType.values),cmap=plt.cm.winter,alpha=.5)
    axScatter.scatter(x_bad[feat_1],x_bad[feat_2],c='r',alpha=.2)

    # Determine nice limits by hand
    binwidth = 0.025
    maxi_1 = np.max([np.max(x_test[feat_1]),np.max(x_train[feat_1])])
    mini_1 = np.min([np.min(x_test[feat_1]),np.min(x_train[feat_1])])
    lim_plot_sup_1 = (int(maxi_1/binwidth)+2)*binwidth
    lim_plot_inf_1 = (int(mini_1/binwidth)-2)*binwidth
    if lim_plot_sup_1 > 1: 
      lim_plot_sup_1 = 1
    if lim_plot_inf_1 < -1:
      lim_plot_inf_1 = -1
    bins_1 = np.arange(lim_plot_inf_1, lim_plot_sup_1 + binwidth, binwidth)

    maxi_2 = np.max([np.max(x_test[feat_2]),np.max(x_train[feat_2])])
    mini_2 = np.min([np.min(x_test[feat_2]),np.min(x_train[feat_2])])
    lim_plot_sup_2 = (int(maxi_2/binwidth)+2)*binwidth
    lim_plot_inf_2 = (int(mini_2/binwidth)-2)*binwidth
    if lim_plot_sup_2 > 1: 
      lim_plot_sup_2 = 1
    if lim_plot_inf_2 < -1:
      lim_plot_inf_2 = -1
    bins_2 = np.arange(lim_plot_inf_2, lim_plot_sup_2 + binwidth, binwidth)

    # Plot decision boundaries
    for i in sorted(theta):
      db = -1./theta[i][2]*(theta[i][0]+np.log((1-t[i])/t[i])+theta[i][1]*x_vec[0])
      axScatter.plot(x_vec[0],db,lw=3.,c='orange')
      if th_comp and t_comp:
        db = -1./th_comp[i][2]*(th_comp[i][0]+np.log((1-t_comp[i])/t_comp[i])+th_comp[i][1]*x_vec[0])
        axScatter.plot(x_vec[0],db,lw=3.,c='purple')

    axScatter.contourf(x_vec, y_vec, map, cmap=plt.cm.gray, alpha=0.2)

    label = ['%s (%.2f%%)'%(method.upper(),rate['global'])]
    if th_comp and t_comp:
      label.append('SVM (%.2f%%)'%p['global'])
    axScatter.legend(label,loc=2,prop={'size':10})

    if p_train:
       x_pos = .7
       y_pos = .95
       pas = .05
       axScatter.text(x_pos,y_pos,"%s %% %s"%(p_train[(str_t[0],0)],str_t[0]),color='b',transform=axScatter.transAxes)
       axScatter.text(x_pos,y_pos-pas,"%s %% %s"%(p_train[(str_t[1],1)],str_t[1]),color='g',transform=axScatter.transAxes)
       axScatter.text(x_pos,y_pos-2*pas,"%.2f %% test set"%rate['global'],transform=axScatter.transAxes)
       axScatter.text(x_pos,y_pos-3*pas,"%.2f %% test set"%(100-rate['global']),color='r',transform=axScatter.transAxes)

    axScatter.set_xlim((lim_plot_inf_1, lim_plot_sup_1))
    axScatter.set_ylim((lim_plot_inf_2, lim_plot_sup_2))

    axScatter.set_xlabel(x_test.columns[0])
    axScatter.set_ylabel(x_test.columns[1])

    # Plot histograms and PDFs
    plot_histos_and_pdfs_kde(axHistx,axHisty,bins_1,bins_2,x_test,y_test,x_train=x_train,y_train=y_train) 
    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())

    # Display success rates
    pos_y_ini = .95
    pas = .025
    plt.figtext(.78,pos_y_ini,'Test set %s'%method.upper())
    for key in sorted(rate):
      if key != 'global':
        cl, icl = key[0], key[1]
        plt.figtext(.78,pos_y_ini-(icl+1)*pas,'%s (%d) : %s%%'%(cl,icl,rate[(cl,icl)]))
    if th_comp and t_comp:
      pos_y = pos_y_ini-.04
      plt.figtext(.78,pos_y-NB_class*pas,'Test set SVM')
      for key in sorted(p):
        if key != 'global':
          cl, icl = key[0], key[1]
          plt.figtext(.78,pos_y-NB_class*pas-(icl+1)*pas,'%s (%d) : %s%%'%(cl,icl,p[(cl,icl)]))

