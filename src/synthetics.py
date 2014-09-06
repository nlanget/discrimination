#!/usr/bin/env python
# encoding: utf-8

import sys, os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import matplotlib.mlab as mlab



def create_synthetics(npts,sig_x,sig_y,theta,mean):
  """
  Crée un jeu de données synthétiques
  Distributions gaussiennes à 2 variables
  npts : nombre d'échantillons
  sig_x : variance en x
  sig_y : variance en y
  theta : inclinaison (rotation) en radians
  mean : moyenne
  """
  a = np.cos(theta)**2/(2*sig_x**2)+np.sin(theta)**2/(2*sig_y**2)
  b = -np.sin(2*theta)/(4*sig_x**2)+np.sin(2*theta)/(4*sig_y**2)
  c = np.sin(theta)**2/(2*sig_x**2)+np.cos(theta)**2/(2*sig_y**2)
  cov = [[a,b],[b,c]]
  return np.random.multivariate_normal(mean,cov,npts)


from options import MultiOptions
class Synthetics(MultiOptions):

  def __init__(self):

    self.opdict = {}
    self.opdict['option'] = 'norm'
    self.opdict['dir'] = 'Test'
    self.opdict['stations'] = ['STA']
    self.opdict['channels'] = ['Z']
    self.opdict['types'] = ['A','B']

    self.opdict['libdir'] = os.path.join('../lib',self.opdict['dir'])
    self.opdict['outdir'] = os.path.join('../results',self.opdict['dir'])
    self.opdict['fig_path'] = '%s/figures'%self.opdict['outdir']

    self.sep = 'well'
    self.opdict['feat_train'] = '%s_%dc_train.csv'%(self.sep,len(self.opdict['types']))
    self.opdict['feat_test'] = '%s_%dc_test.csv'%(self.sep,len(self.opdict['types']))
    self.opdict['label_train'] = '%s_%dc_train.csv'%(self.sep,len(self.opdict['types']))
    self.opdict['label_test'] = '%s_%dc_test.csv'%(self.sep,len(self.opdict['types']))
    self.opdict['learn_file'] = os.path.join(self.opdict['libdir'],'learning_set')

    self.opdict['feat_list'] = ['x1','x2']

    self.opdict['method'] = 'lr'
    self.opdict['boot'] = 1
    self.opdict['plot_pdf'] = False # display the pdfs of the features
    self.opdict['save_pdf'] = False
    self.opdict['plot_confusion'] = False # display the confusion matrices
    self.opdict['save_confusion'] = False
    self.opdict['plot_sep'] = False # plot decision boundary
    self.opdict['save_sep'] = False
    self.opdict['plot_prec_rec'] = False # plot precision and recall

    self.opdict['feat_filepath'] = '%s/features/%s'%(self.opdict['outdir'],self.opdict['feat_test'])
    self.opdict['label_filename'] = '%s/%s'%(self.opdict['libdir'],self.opdict['label_test'])
    self.opdict['result_file'] = 'well_sep'
    self.opdict['result_path'] = '%s/%s/%s'%(self.opdict['outdir'],self.opdict['method'].upper(),self.opdict['result_file'])


  def set_params(self):
    # Random data
    self.NB_class = 2
    NB_test_all = 1200
    NB_train_all = int(0.4*NB_test_all)
    #prop = (1./3,1./3,1./3)
    prop = (.1,.9)

    if len(prop) != self.NB_class:
      print "Warning ! Check number of classes and proportions"
      sys.exit()
    if np.sum(prop) != 1:
      print "Warning ! Set correct proportions of the classes"
      sys.exit()


    self.NB_train, self.NB_test = {},{}
    for i in range(self.NB_class):
      self.NB_train[i] = int(NB_train_all*prop[i])
      self.NB_test[i] = int(NB_test_all*prop[i])
    print "Training set:", self.NB_train
    print "Test set:", self.NB_test

    if np.sum(self.NB_train.values()) != NB_train_all:
      print "Warning ! Check the total number of training set samples and their repartition into classes"
      sys.exit()
    if np.sum(self.NB_test.values()) != NB_test_all:
      print "Warning ! Check the total number of test set samples and their repartition into classes"
      sys.exit()


  def fill_matrices(self):

    self.set_params()
    NB_train_all = np.sum(self.NB_train.values())
    NB_test_all = np.sum(self.NB_test.values())

    from math import pi
    # Basis
    b_sig_x = 4
    b_sig_y = 2
    b_theta = -pi/6
    b_mean = [0,0]

    if self.sep == 'well':
      # Well separated
      sig_x = 2
      sig_y = 7
      theta = 0
      mean = [.85,-.5]
    elif self.sep == 'bad':
      # Badly separated
      sig_x = 4
      sig_y = 2
      theta = 0
      mean = [0.25,-.25]

    if self.NB_class == 3:
      sig_x_3c = 4
      sig_y_3c = 5
      theta_3c = pi/4
      mean_3c = [.75,0]


    s = {}
    s[0] = create_synthetics(self.NB_train[0],b_sig_x,b_sig_y,b_theta,b_mean)
    s[1] = create_synthetics(self.NB_train[1],sig_x,sig_y,theta,mean)
    s_val = np.concatenate([s[0],s[1]])
    if self.NB_class == 3:
      s[2] = create_synthetics(self.NB_train[2],sig_x_3c,sig_y_3c,theta_3c,mean_3c)
      s_val = np.concatenate([s_val,s[2]])

    x_train = {}
    x_train['x1'] = s_val[:,0]
    x_train['x2'] = s_val[:,1]
    x_train = pd.DataFrame(x_train)
    x_train.index = [(i,'STA','Z') for i in range(len(x_train))]

    s = {}
    s[0] = create_synthetics(self.NB_test[0],b_sig_x,b_sig_y,b_theta,b_mean)
    s[1] = create_synthetics(self.NB_test[1],sig_x,sig_y,theta,mean)
    s_val = np.concatenate([s[0],s[1]])
    if self.NB_class == 3:
      s[2] = create_synthetics(self.NB_test[2],sig_x_3c,sig_y_3c,theta_3c,mean_3c)
      s_val = np.concatenate([s_val,s[2]])

    x_test = {}
    x_test['x1'] = s_val[:,0]
    x_test['x2'] = s_val[:,1]
    x_test = pd.DataFrame(x_test)
    x_test.index = [(i,'STA','Z') for i in range(len(x_test))]

    types = ['A','B','C']
    y_train = pd.DataFrame(index=range(NB_train_all),columns=['Type','Date','TypeNum'],dtype=str)
    lim = 0
    for i in range(self.NB_class):
      y_train['Type'].values[lim:lim+self.NB_train[i]] = [types[i]]*self.NB_train[i]
      y_train['TypeNum'].values[lim:lim+self.NB_train[i]] = ['%d'%i]*self.NB_train[i]
      lim = lim + self.NB_train[i]
    y_train['Date'] = y_train.index

    y_test = pd.DataFrame(index=range(NB_test_all),columns=['Type','Date','TypeNum'],dtype=str)
    lim = 0
    for i in range(self.NB_class):
      y_test['Type'].values[lim:lim+self.NB_test[i]] = [types[i]]*self.NB_test[i]
      y_test['TypeNum'].values[lim:lim+self.NB_test[i]] = ['%d'%i]*self.NB_test[i]
      lim = lim + self.NB_test[i]
    y_test['Date'] = y_test.index

    x_train.to_csv('%s/features/%s'%(self.opdict['outdir'],self.opdict['feat_train']))
    x_test.to_csv(self.opdict['feat_filepath'])
    y_train.to_csv('%s/%s'%(self.opdict['libdir'],self.opdict['label_train']),index=False)
    y_test.to_csv(self.opdict['label_filename'],index=False)



def plot_sep(opt):

    opt.set_params()

    from do_classification import classifier
    opt.opdict['method'] = 'svm'
    classifier(opt)
    theta_svm = opt.theta
    rate_svm = opt.success

    opt.opdict['method'] = 'lr'

    theta_lr,rate_lr = {}, {}
    for b in range(1):
      opt.opdict['learn_file'] = os.path.join(opt.opdict['libdir'],'LR_%d'%b)
      classifier(opt)
      theta_lr[b] = opt.theta
      rate_lr[b] = opt.success

    print "LR", theta_lr
    print "SVM", theta_svm

    x_train = pd.read_csv('%s/features/%s'%(opt.opdict['outdir'],opt.opdict['feat_train']),index_col=False)
    x_test = pd.read_csv(opt.opdict['feat_filepath'],index_col=False)
    y_train = pd.read_csv('%s/%s'%(opt.opdict['libdir'],opt.opdict['label_train']))
    y_test = pd.read_csv(opt.opdict['label_filename'])

    NB_class = len(opt.opdict['types'])

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
    if len(theta_lr) < 2:
      if NB_class == 2:
        axScatter.plot(x,1./theta_lr[0][1][2]*(-theta_lr[0][1][0]-theta_lr[0][1][1]*x),lw=2.,c='b',label='LR (%.1f%%)'%rate_lr[0][1])
        axScatter.plot(x,1./theta_svm[1][2]*(-theta_svm[1][0]-theta_svm[1][1]*x),lw=2.,c='c',label='SVM (%.1f%%)'%rate_svm[1])
      elif NB_class > 2:
        for i in range(NB_class):
          axScatter.plot(x,1./theta_lr[0][i+1][2]*(-theta_lr[0][i+1][0]-theta_lr[0][i+1][1]*x),lw=1.,c='b')
          axScatter.plot(x,1./theta_svm[i+1][2]*(-theta_svm[i+1][0]-theta_svm[i+1][1]*x),lw=1.,c='c')
      axScatter.legend(loc=4)

    else:
      rates = []
      for i in range(len(theta_lr)):
        axScatter.plot(x,1./theta_lr[i][1][2]*(-theta_lr[i][1][0]-theta_lr[i][1][1]*x),lw=2.,c=(0,0.1*i,1))
        rates.append(rate_lr[i][1])
      print np.mean(rates),np.std(rates)
      axScatter.text(0.6*lim_plot,-0.9*lim_plot,r'%.1f$\pm$%.1f%%'%(np.mean(rates),np.std(rates)))

    axScatter.set_xlim((-lim_plot, lim_plot))
    axScatter.set_ylim((-lim_plot, lim_plot))


    # Plot histograms and PDFs
    lim = 0
    x_hist, y_hist = [],[]
    g_x, g_y = {}, {}

    if NB_class == 2:
      colors = ('w','k')
    elif NB_class == 3:
      colors = ('w','gray','k')

    for i in range(NB_class):
      x_hist.append(x[lim:lim+opt.NB_test[i]])
      y_hist.append(y[lim:lim+opt.NB_test[i]])
      g_x[i] = mlab.normpdf(bins, np.mean(x[lim:lim+opt.NB_test[i]]), np.std(x[lim:lim+opt.NB_test[i]]))
      g_y[i] = mlab.normpdf(bins, np.mean(y[lim:lim+opt.NB_test[i]]), np.std(y[lim:lim+opt.NB_test[i]]))
      axHisty.hist(y[lim:lim+opt.NB_test[i]],bins=bins,color=colors[i],normed=1,orientation='horizontal',histtype='stepfilled',alpha=.5)
      lim = lim + opt.NB_test[i]
    axHistx.hist(x_hist,bins=bins,color=colors,normed=1,histtype='stepfilled',alpha=.5)

    if NB_class == 2:
      colors = ('r','y')
    elif NB_class == 3:
      colors = ('r','orange','y')

    for key in sorted(g_x):
      axHistx.plot(bins,g_x[key],color=colors[key],lw=2.)
      axHisty.plot(g_y[key],bins,color=colors[key],lw=2.)

    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())

    axScatter.set_xlabel('x1')
    axScatter.set_ylabel('x2')

    plt.savefig('%s/Test_2c_ineq.png'%opt.opdict['fig_path'])
    plt.show()


def run_synthetics():
  synth = Synthetics()
  synth.fill_matrices()
  plot_sep(synth)

if __name__ == '__main__':
  run_synthetics()
  
