#!/usr/bin/env python
# encoding: utf-8

import sys, os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plot_2features import plot_2f_synthetics, plot_2f_synth_var, plot_2f_synthetics_nonlinear

def create_synthetics(npts,tup):
  """
  Crée un jeu de données synthétiques
  Distributions gaussiennes à 2 variables
  npts : nombre d'échantillons
  sig_x : variance en x
  sig_y : variance en y
    ==> extension
  theta : inclinaison (rotation) en radians
  mean : moyenne (position)
  """
  sig_x = tup[0]
  sig_y = tup[1]
  theta = tup[2]
  mean = tup[3]
  a = np.cos(theta)**2/(2*sig_x**2)+np.sin(theta)**2/(2*sig_y**2)
  b = -np.sin(2*theta)/(4*sig_x**2)+np.sin(2*theta)/(4*sig_y**2)
  c = np.sin(theta)**2/(2*sig_x**2)+np.cos(theta)**2/(2*sig_y**2)
  cov = [[a,b],[b,c]]
  return np.random.multivariate_normal(mean,cov,npts)


from options import MultiOptions
class Synthetics(MultiOptions):

  def __init__(self):

    self.synthetics()
    self.fill_opdict()
    self.opdict['feat_list'] = self.opdict['feat_all']

  def set_params(self):
    # Random data
    self.NB_class = len(self.opdict['types'])
    NB_test_all = 1200
    NB_train_all = int(0.4*NB_test_all)

    # Choose the proportions in the test set
    if self.NB_class == 2:
      prop_test = (.45,.55)
      #prop_test = (.25,.75)

    elif self.NB_class == 3:
      prop_test = (1./3,1./3,1./3)
      #prop_test = (0.3,0.1,0.6)

    # Choose the proportions in the training set
    prop_train = prop_test
    #prop_train = (.35,.65)

    if len(prop_test) != self.NB_class or len(prop_train) != self.NB_class:
      print "Warning ! Check number of classes and proportions"
      sys.exit()

    self.NB_train, self.NB_test = {},{}
    for i in range(self.NB_class):
      self.NB_train[i] = int(NB_train_all*prop_train[i])
      self.NB_test[i] = int(NB_test_all*prop_test[i])
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
    ### Choose the repartition of the features values ###
    ### May be different for the test and training sets...

    ### CLASS 0 ###
    class_0_test = (4, 2, -pi/6, [0,0])
    class_0_train = class_0_test
    #class_0_train = (2, 2, -pi/6, [0.1,0.1])

    ### CLASS 1 ###
    if self.sep == 'well':
      # Well separated
      class_1_test = (2, 7, 0, [.85,-.5])
      class_1_train = class_1_test
      #class_1_train = (3, 4, 0, [.70,-.3])

    elif self.sep == 'very_well':
      class_1_test = (4, 3, pi/4, [1.5,.9])
      class_1_train = class_1_test

    elif self.sep == 'bad':
      # Badly separated
      class_1_test = (4, 2, 0, [0.25,-.25])
      class_1_train = class_1_test

    ### CLASS 3 ###
    if self.NB_class == 3:
      class_2_test = (4, 3, pi/4, [.5,-.1])
      class_2_train = class_2_test


    ### Fill the training set ###
    s = {}
    s[0] = create_synthetics(self.NB_train[0],class_0_train)
    s[1] = create_synthetics(self.NB_train[1],class_1_train)
    s_val = np.concatenate([s[0],s[1]])
    if self.NB_class == 3:
      s[2] = create_synthetics(self.NB_train[2],class_2_train)
      s_val = np.concatenate([s_val,s[2]])

    x_train = {}
    x_train['x1'] = s_val[:,0]
    x_train['x2'] = s_val[:,1]
    x_train = pd.DataFrame(x_train)
    x_train.index = [(i,'STA','Z') for i in range(len(x_train))]

    ### Fill the test set ###
    s = {}
    s[0] = create_synthetics(self.NB_test[0],class_0_test)
    s[1] = create_synthetics(self.NB_test[1],class_1_test)
    s_val = np.concatenate([s[0],s[1]])
    if self.NB_class == 3:
      s[2] = create_synthetics(self.NB_test[2],class_2_test)
      s_val = np.concatenate([s_val,s[2]])

    x_test = {}
    x_test['x1'] = s_val[:,0]
    x_test['x2'] = s_val[:,1]
    x_test = pd.DataFrame(x_test)
    x_test.index = [(i,'STA','Z') for i in range(len(x_test))]

    ### Fill the labels ###
    types = self.opdict['types']
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
    x_test.to_csv(self.opdict['feat_filename'])
    y_train.to_csv('%s/%s'%(self.opdict['libdir'],self.opdict['label_train']),index=False)
    y_test.to_csv(self.opdict['label_filename'],index=False)


  def plot_PDFs(self):
    """
    Plot probability density functions
    """
    import matplotlib.mlab as mlab
    NB_class = len(self.opdict['types'])
    feat_1 = self.x.columns[0]
    feat_2 = self.x.columns[1]

    binwidth = .05
    lim_sup_1 = (int(np.max(self.x[feat_1])/binwidth)+2)*binwidth
    lim_inf_1 = (int(np.min(self.x[feat_1])/binwidth)-2)*binwidth
    bins_1 = np.arange(lim_inf_1, lim_sup_1 + binwidth, binwidth)
    lim_sup_2 = (int(np.max(self.x[feat_2])/binwidth)+2)*binwidth
    lim_inf_2 = (int(np.min(self.x[feat_2])/binwidth)-2)*binwidth
    bins_2 = np.arange(lim_inf_2, lim_sup_2 + binwidth, binwidth)

    x_hist, y_hist = [],[]
    g_x, g_y = {}, {}
    for i in range(NB_class):
      index = self.y[self.y.NumType.values==i].index
      x1 = self.x.reindex(columns=[feat_1],index=index).values
      x2 = self.x.reindex(columns=[feat_2],index=index).values
      g_x[i] = mlab.normpdf(bins_1, np.mean(x1), np.std(x1))
      g_y[i] = mlab.normpdf(bins_2, np.mean(x2), np.std(x2))
      x_hist.append(x1)
      y_hist.append(x2)

    if NB_class > 2:
      colors_g = ('y','orange','r')
      colors_h = ('k','gray','w')
    elif NB_class == 2:
      colors_g = ('y','r')
      colors_h = ('k','w')

    fig = plt.figure()
    fig.set_facecolor('white')
    plt.hist(x_hist,bins=bins_1,color=colors_h,normed=1,histtype='stepfilled',alpha=.5)
    for key in sorted(g_x):
      plt.plot(bins_1,g_x[key],color=colors_g[key],lw=2.,label='Class %s'%self.opdict['types'][key])
    plt.xlabel(feat_1)
    plt.legend(loc=2)
    plt.savefig('%s/histo_%s.png'%(self.opdict['fig_path'],feat_1))

    fig = plt.figure()
    fig.set_facecolor('white')
    plt.hist(y_hist,bins=bins_2,color=colors_h,normed=1,histtype='stepfilled',alpha=.5)
    for key in sorted(g_y):
      plt.plot(bins_2,g_y[key],color=colors_g[key],lw=2.,label='Class %s'%self.opdict['types'][key])
    plt.plot([.5,.5],[0,2],'g--',lw=2.)
    plt.figtext(.52,.7,'?',color='g',size=20)
    plt.xlabel(feat_2)
    plt.legend(loc=2)
    plt.savefig('%s/histo_%s.png'%(self.opdict['fig_path'],feat_2))
    plt.show()


def plot_sep(opt):

    opt.set_params()
    opt.opdict['fig_path'] = os.path.join(opt.opdict['outdir'],'figures')

    from do_classification import classifier
    ### LINEAR SVM ###
    opt.opdict['method'] = 'svm'
    classifier(opt)
    #opt.plot_PDFs()
    out_svm = opt.out
    print "SVM", out_svm['thetas']

    x_train = opt.train_x
    x_test = opt.x
    y_train = opt.train_y
    y_test = opt.y

    # *** Plot ***
    plot_2f_synthetics(out_svm,x_train,x_test,y_test,y_train=y_train)
    plt.savefig('%s/Test_%dc_%s_SVM.png'%(opt.opdict['fig_path'],len(opt.types),opt.sep))
    plt.show()

    ### LOGISTIC REGRESSION ###
    opt.opdict['method'] = 'lr'
    out_lr = {}
    for b in range(1):
      opt.opdict['learn_file'] = os.path.join(opt.opdict['libdir'],'LR_%d'%b)
      #os.remove(opt.opdict['learn_file'])
      classifier(opt)
      out_lr[b] = opt.out

    print "LR", out_lr[0]['thetas']

    # *** Plot ***
    if b == 0:
      plot_2f_synthetics(out_lr[0],x_train,x_test,y_test,y_train=y_train)
      plt.savefig('%s/Test_%dc_%s_LR.png'%(opt.opdict['fig_path'],len(opt.types),opt.sep))
      plt.show()

    else:
      plot_2f_synth_var(out_lr,x_train,x_test,y_test,opt.NB_test)
      plt.savefig('%s/Test_%dc_LR_%s.png'%(opt.opdict['fig_path'],len(opt.types),opt.sep))
      plt.show()

    if opt.opdict['plot_prec_rec']:
      plot_2f_synth_var(out_lr,x_train,x_test,y_test,opt.NB_test)
      plt.savefig('%s/Test_%dc_bad_threshold.png'%(opt.opdict['fig_path'],len(opt.types)))
      plt.show()


    ### NON LINEAR SVM ###
    opt.opdict['method'] = 'svm_nl'
    classifier(opt)
    out_svm_nl = opt.out
    plot_2f_synthetics_nonlinear(out_svm_nl,x_train,x_test,y_test,y_train=y_train)
    plt.savefig('%s/Test_%dc_SVM_NL_%s.png'%(opt.opdict['fig_path'],len(opt.types),opt.sep))
    plt.show()

    ### COMPARE ALL 3 METHODS ON THE SAME PLOT ###
    plot_2f_synthetics(out_lr[0],x_train,x_test,y_test,out_comp=out_svm,y_train=y_train,map_nl=out_svm_nl)
    plt.savefig('%s/Test_%dc_%s.png'%(opt.opdict['fig_path'],len(opt.types),opt.sep))
    plt.show()


def run_synthetics():
  synth = Synthetics()
  synth.fill_matrices()
  plot_sep(synth)

if __name__ == '__main__':
  run_synthetics()
  
