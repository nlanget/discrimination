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
  a=np.cos(theta)**2/(2*sig_x**2)+np.sin(theta)**2/(2*sig_y**2)
  b=-np.sin(2*theta)/(4*sig_x**2)+np.sin(2*theta)/(4*sig_y**2)
  c=np.sin(theta)**2/(2*sig_x**2)+np.cos(theta)**2/(2*sig_y**2)
  cov=[[a,b],[b,c]]
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
    self.opdict['plot_sep'] = True # plot decision boundary
    self.opdict['save_sep'] = False
    self.opdict['plot_prec_rec'] = False # plot precision and recall

    self.opdict['feat_filepath'] = '%s/features/%s'%(self.opdict['outdir'],self.opdict['feat_test'])
    self.opdict['label_filename'] = '%s/%s'%(self.opdict['libdir'],self.opdict['label_test'])
    self.opdict['result_file'] = 'well_sep'
    self.opdict['result_path'] = '%s/%s/%s'%(self.opdict['outdir'],self.opdict['method'].upper(),self.opdict['result_file'])


  def fill_matrices(self):
    from math import pi
    # Random data
    NB_train = 400
    NB_test = 1000
    prop = (.3,.7)

    NB_train_0 = int(NB_train*prop[0])
    NB_train_1 = int(NB_train*prop[1])
    NB_test_0 = int(NB_test*prop[0])
    NB_test_1 = int(NB_test*prop[1])
    print "Training set:", NB_train_0, NB_train_1
    print "Test set:", NB_test_0, NB_test_1

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

    s1 = create_synthetics(NB_train_0,b_sig_x,b_sig_y,b_theta,b_mean)
    s2 = create_synthetics(NB_train_1,sig_x,sig_y,theta,mean)
    x_train = {}
    x_train['x1'] = np.concatenate((s1[:,0],s2[:,0]))
    x_train['x2'] = np.concatenate((s1[:,1],s2[:,1]))
    x_train = pd.DataFrame(x_train)
    x_train.index = [(i,'STA','Z') for i in range(len(x_train))]

    s1 = create_synthetics(NB_test_0,b_sig_x,b_sig_y,b_theta,b_mean)
    s2 = create_synthetics(NB_test_1,sig_x,sig_y,theta,mean)
    x_test = {}
    x_test['x1'] = np.concatenate((s1[:,0],s2[:,0]))
    x_test['x2'] = np.concatenate((s1[:,1],s2[:,1]))
    x_test = pd.DataFrame(x_test)
    x_test.index = [(i,'STA','Z') for i in range(len(x_test))]

    y_train = pd.DataFrame(index=range(NB_train),columns=['Type','Date'])
    y_train.Type = ['0']*NB_train
    y_train.Type.values[NB_train_0:] = ['1']*NB_train_1
    y_train['Date'] = y_train.index

    y_test = pd.DataFrame(index=range(NB_test),columns=['Type','Date'])
    y_test['Type'] = ['0']*NB_test
    y_test['Type'].values[NB_test_0:] = ['1']*NB_test_1
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

    opt.opdict['method'] = 'lr'
    classifier(opt)
    theta_lr = opt.theta

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
    axScatter.scatter(x,y,c=list(y_test.Type.values),cmap=plt.cm.gray)
    axScatter.plot(x,1./theta_lr[1][2]*(-theta_lr[1][0]-theta_lr[1][1]*x),lw=2.,c='b',label='LR')
    axScatter.plot(x,1./theta_svm[1][2]*(-theta_svm[1][0]-theta_svm[1][1]*x),lw=2.,c='c',label='SVM')
    axScatter.legend(loc=4)

    # now determine nice limits by hand:
    binwidth = 0.05
    xymax = np.max( [np.max(np.fabs(x)), np.max(np.fabs(y))] )
    lim = ( int(xymax/binwidth) + 1) * binwidth

    axScatter.set_xlim((-lim, lim))
    axScatter.set_ylim((-lim, lim))

    bins = np.arange(-lim, lim + binwidth, binwidth)
    axHistx.hist([x[:len(x)/2],x[len(x)/2:]], bins=bins,color=('w','k'),normed=1,histtype='stepfilled')
    g_x_0 = mlab.normpdf(bins, np.mean(x[:len(x)/2]), np.std(x[:len(x)/2]))
    axHistx.plot(bins,g_x_0,'r',lw=2.)
    g_x_1 = mlab.normpdf(bins, np.mean(x[len(x)/2:]), np.std(x[len(x)/2:]))
    axHistx.plot(bins,g_x_1,'y',lw=2.)

    axHisty.hist(y[:len(y)/2],bins=bins,color=('w'),normed=1,orientation='horizontal',histtype='stepfilled')
    axHisty.hist(y[len(y)/2:],bins=bins,color=('k'),normed=1,orientation='horizontal')
    g_y_0 = mlab.normpdf(bins, np.mean(y[:len(y)/2]), np.std(y[:len(y)/2]))
    axHisty.plot(g_y_0,bins,'r',lw=2.)
    g_y_1 = mlab.normpdf(bins, np.mean(y[len(y)/2:]), np.std(y[len(y)/2:]))
    axHisty.plot(g_y_1,bins,'y',lw=2.)

    axHistx.set_xlim( axScatter.get_xlim() )
    axHisty.set_ylim( axScatter.get_ylim() )

    axScatter.set_xlabel('x1')
    axScatter.set_ylabel('x2')
    plt.show()


def run_synthetics():
  synth = Synthetics()
  synth.fill_matrices()
  plot_sep(synth)

if __name__ == '__main__':
  run_synthetics()
  
