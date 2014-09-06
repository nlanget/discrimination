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
    self.opdict['types'] = ['0','1']

    self.opdict['libdir'] = os.path.join('../lib',self.opdict['dir'])
    self.opdict['outdir'] = os.path.join('../results',self.opdict['dir'])
    self.opdict['fig_path'] = '%s/figures'%self.opdict['outdir']

    self.sep = 'well'
    self.opdict['feat_train'] = '%s_sep_train.csv'%self.sep
    self.opdict['feat_test'] = '%s_sep_test.csv'%self.sep
    self.opdict['label_train'] = '%s_sep_train.csv'%self.sep
    self.opdict['label_test'] = '%s_sep_test.csv'%self.sep
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


  def fill_matrices(self):
    from math import pi

    # Random data
    NB_class = 3
    NB_test_all = 1200
    NB_train_all = int(0.4*NB_test_all)
    prop = (1./3,1./3,1./3)
    #prop = (.5,.5)

    if len(prop) != NB_class:
      print "Warning ! Check number of classes and proportions"
      sys.exit()
    if np.sum(prop) != 1:
      print "Warning ! Set correct proportions of the classes"
      sys.exit()


    self.NB_train, self.NB_test = {},{}
    for i in range(NB_class):
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
      mean = [.25,-.25]

    if NB_class == 3:
      sig_x_3c = 4
      sig_y_3c = 5
      theta_3c = pi/4
      mean_3c = [.75,0]


    s = {}
    s[0] = create_synthetics(self.NB_train[0],b_sig_x,b_sig_y,b_theta,b_mean)
    s[1] = create_synthetics(self.NB_train[1],sig_x,sig_y,theta,mean)
    s_val = np.concatenate([s[0],s[1]])
    if NB_class == 3:
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
    if NB_class == 3:
      s[2] = create_synthetics(self.NB_test[2],sig_x_3c,sig_y_3c,theta_3c,mean_3c)
      s_val = np.concatenate([s_val,s[2]])

    x_test = {}
    x_test['x1'] = s_val[:,0]
    x_test['x2'] = s_val[:,1]
    x_test = pd.DataFrame(x_test)
    x_test.index = [(i,'STA','Z') for i in range(len(x_test))]

    y_train = pd.DataFrame(index=range(NB_train_all),columns=['Type','Date'])
    y_train.Type = ['0']*NB_train_all
    lim = self.NB_train[0]
    for i in range(1,NB_class):
      y_train['Type'].values[lim:lim+self.NB_train[i]] = ['%d'%i]*self.NB_train[i]
      lim = lim + self.NB_train[i]
    y_train['Date'] = y_train.index

    y_test = pd.DataFrame(index=range(NB_test_all),columns=['Type','Date'])
    y_test['Type'] = ['0']*NB_test_all
    lim = self.NB_test[0]
    for i in range(1,NB_class):
      y_test['Type'].values[lim:lim+self.NB_test[i]] = ['%d'%i]*self.NB_test[i]
      lim = lim + self.NB_test[i]
    y_test['Date'] = y_test.index

    x_train.to_csv('%s/features/%s_sep_train.csv'%(self.opdict['outdir'],self.sep))
    x_test.to_csv('%s/features/%s_sep_test.csv'%(self.opdict['outdir'],self.sep))
    y_train.to_csv('%s/%s_sep_train.csv'%(self.opdict['libdir'],self.sep),index=False)
    y_test.to_csv('%s/%s_sep_test.csv'%(self.opdict['libdir'],self.sep),index=False)


def plot_sep(opt):

    from do_classification import classifier
    opt.opdict['method'] = 'svm'
    classifier(opt)
    theta_svm = opt.theta
    rate_svm = opt.success

    opt.opdict['method'] = 'lr'
    classifier(opt)
    theta_lr = opt.theta
    rate_lr = opt.success

    print "LR", theta_lr
    print "SVM", theta_svm

    x_train = pd.read_csv('%s/features/%s_sep_train.csv'%(opt.opdict['outdir'],opt.sep),index_col=False)
    x_test = pd.read_csv('%s/features/%s_sep_test.csv'%(opt.opdict['outdir'],opt.sep),index_col=False)
    y_train = pd.read_csv('%s/%s_sep_train.csv'%(opt.opdict['libdir'],opt.sep))
    y_test = pd.read_csv('%s/%s_sep_test.csv'%(opt.opdict['libdir'],opt.sep))


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

    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    # the scatter plot:
    from LR_functions import normalize
    x_train, x_test = normalize(x_train,x_test)
    x = x_test['x1']
    y = x_test['x2']
    axScatter.scatter(x,y,c=list(y_test.Type.values),cmap=plt.cm.gray_r)

    axScatter.plot(x,1./theta_lr[1][2]*(-theta_lr[1][0]-theta_lr[1][1]*x),lw=2.,c='b',label='LR (%.1f%%)'%rate_lr[1])
    axScatter.plot(x,1./theta_svm[1][2]*(-theta_svm[1][0]-theta_svm[1][1]*x),lw=2.,c='c',label='SVM (%.1f%%)'%rate_svm[1])
    for i in range(1,len(opt.opdict['types'])):
      axScatter.plot(x,1./theta_lr[i+1][2]*(-theta_lr[i+1][0]-theta_lr[i+1][1]*x),lw=2.,c='b')
      axScatter.plot(x,1./theta_svm[i+1][2]*(-theta_svm[i+1][0]-theta_svm[i+1][1]*x),lw=2.,c='c')
    axScatter.legend(loc=4)

    # now determine nice limits by hand:
    binwidth = 0.025
    xymax = np.max( [np.max(np.fabs(x)), np.max(np.fabs(y))] )
    lim = ( int(xymax/binwidth) + 1) * binwidth

    axScatter.set_xlim((-lim, lim))
    axScatter.set_ylim((-lim, lim))

    bins = np.arange(-lim, lim + binwidth, binwidth)

    lim = 0
    x_hist, y_hist = [],[]
    g_x, g_y = {}, {}
    colors = ('w','gray','k')
    for i in range(len(opt.opdict['types'])):
      x_hist.append(x[lim:lim+opt.NB_test[i]])
      y_hist.append(y[lim:lim+opt.NB_test[i]])
      g_x[i] = mlab.normpdf(bins, np.mean(x[lim:lim+opt.NB_test[i]]), np.std(x[lim:lim+opt.NB_test[i]]))
      g_y[i] = mlab.normpdf(bins, np.mean(y[lim:lim+opt.NB_test[i]]), np.std(y[lim:lim+opt.NB_test[i]]))
      axHisty.hist(y[lim:lim+opt.NB_test[i]],bins=bins,color=colors[i],normed=1,orientation='horizontal',histtype='stepfilled',alpha=.5)
      lim = lim + opt.NB_test[i]
    axHistx.hist(x_hist,bins=bins,color=colors,normed=1,histtype='stepfilled',alpha=.5)
    #axHisty.hist(y_hist,bins=bins,color=colors,normed=1,orientation='horizontal',histtype='stepfilled')

    colors = ('r','orange','y')
    for key in sorted(g_x):
      axHistx.plot(bins,g_x[key],color=colors[key],lw=2.)
      axHisty.plot(g_y[key],bins,color=colors[key],lw=2.)

    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())

    axScatter.set_xlabel('x1')
    axScatter.set_ylabel('x2')
    plt.show()


def run_synthetics():
  synth = Synthetics()
  synth.fill_matrices()
  plot_sep(synth)

if __name__ == '__main__':
  run_synthetics()
  
