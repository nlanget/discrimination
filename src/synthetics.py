#!/usr/bin/env python
# encoding: utf-8

import sys, os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plot_2features import plot_2f_synthetics, plot_2f_synth_var

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

    self.sep = 'very_well'
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
    self.opdict['compare'] = False

    self.opdict['feat_filepath'] = '%s/features/%s'%(self.opdict['outdir'],self.opdict['feat_test'])
    self.opdict['label_filename'] = '%s/%s'%(self.opdict['libdir'],self.opdict['label_test'])
    self.opdict['result_file'] = 'well_sep'
    self.opdict['result_path'] = '%s/%s/%s'%(self.opdict['outdir'],self.opdict['method'].upper(),self.opdict['result_file'])


  def set_params(self):
    # Random data
    self.NB_class = len(self.opdict['types'])
    NB_test_all = 1200
    NB_train_all = int(0.4*NB_test_all)
    if self.NB_class == 2:
      prop = (.5,.5)
    elif self.NB_class == 3:
      prop = (1./3,1./3,1./3)
      #prop = (0.25,0.5,0.25)

    if len(prop) != self.NB_class:
      print "Warning ! Check number of classes and proportions"
      sys.exit()
    #if (1-np.sum(prop)) < 10**-6:
    #  print "Warning ! Set correct proportions of the classes"
    #  sys.exit()


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
    elif self.sep == 'very_well':
      sig_x = 4
      sig_y = 3
      theta = pi/4
      mean = [1.5,.9]
    elif self.sep == 'bad':
      # Badly separated
      sig_x = 4
      sig_y = 2
      theta = 0
      mean = [0.25,-.25]

    if self.NB_class == 3:
      sig_x_3c = 4
      sig_y_3c = 3
      theta_3c = pi/4
      mean_3c = [.5,-.1]

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
    ### SVM ###
    opt.opdict['method'] = 'svm'
    classifier(opt)
    theta_svm = opt.theta
    rate_svm = opt.success
    t_svm = opt.threshold

    ### LOGISTIC REGRESSION ###
    opt.opdict['method'] = 'lr'
    theta_lr,rate_lr,t_lr = {}, {}, {}
    for b in range(1):
      opt.opdict['learn_file'] = os.path.join(opt.opdict['libdir'],'LR_%d'%b)
      #os.remove(opt.opdict['learn_file'])
      classifier(opt)
      theta_lr[b] = opt.theta
      rate_lr[b] = opt.success
      t_lr[b] = opt.threshold

    print "LR", theta_lr
    print "SVM", theta_svm

    x_train = opt.train_x
    x_test = opt.x
    y_train = opt.train_y
    y_test = opt.y

    # PLOTS
    plot_2f_synthetics(theta_svm,rate_svm,t_svm,'SVM',x_train,x_test,y_test,y_train=y_train)
    plt.savefig('%s/Test_3c_%s_SVM_ineq.png'%(opt.opdict['fig_path'],opt.sep))
    plt.show()

    if len(theta_lr) == 1:
      plot_2f_synthetics(theta_lr[0],rate_lr[0],t_lr[0],'LR',x_train,x_test,y_test,y_train=y_train)
      plt.savefig('%s/Test_3c_%s_LR_ineq.png'%(opt.opdict['fig_path'],opt.sep))
      plt.show()
      plot_2f_synthetics(theta_lr[0],rate_lr[0],t_lr[0],'LR',x_train,x_test,y_test,th_comp=theta_svm,p=rate_svm,t_comp=t_svm,y_train=y_train)
      plt.savefig('%s/Test_3c_%s_ineq.png'%(opt.opdict['fig_path'],opt.sep))
      plt.show()

    elif len(theta_lr) > 1:
      plot_2f_variability(theta_lr,rate_lr,t_lr,'LR',x_train,x_test,y_test,opt.NB_test)
      plt.savefig('%s/Test_2c_LR_%s.png'%(opt.opdict['fig_path'],opt.sep))
      plt.show()

    if opt.opdict['plot_prec_rec']:
      plot_2f_synth_var(theta_lr,rate_lr,t_lr,'LR',x_train,x_test,y_test,opt.NB_test)
      plt.savefig('%s/Test_2c_bad_ineq_threshold.png'%opt.opdict['fig_path'])
      plt.show()


def run_synthetics():
  synth = Synthetics()
  synth.fill_matrices()
  plot_sep(synth)

if __name__ == '__main__':
  run_synthetics()
  
