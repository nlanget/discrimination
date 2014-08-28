#!/usr/bin/env python
# encoding: utf-8

import os, glob, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
Some functions for Piton de la Fournaise dataset only.
"""


def compare_clement():
  """
  Comparaison des attributs de Clément avec ceux que j'ai recalculés.
  """

  from options import MultiOptions
  opt = MultiOptions()
  opt.opdict['channels'] = ['Z']

  # Mes calculs
  opt.opdict['feat_list'] = ['Dur','AsDec','RappMaxMean','Kurto','Ene']
  opt.opdict['feat_log'] = ['AsDec','RappMaxMean','Kurto','Ene']
  opt.do_tri()
  opt.x = opt.xs[0]
  opt.y = opt.ys[0]
  opt.opdict['feat_list'] = ['Dur','AsDec','RappMaxMean','Kurto','Ene']
  opt.x.columns = opt.opdict['feat_list']
  opt.compute_pdfs()
  my_gauss = opt.gaussians

  fig = plt.figure()
  fig.set_facecolor('white')
  plt.plot(np.log(opt.x.Kurto),np.log(opt.x.RappMaxMean),'ko')
  plt.xlabel('Kurto')
  plt.ylabel('RappMaxMean')
  plt.show()

  # Les calculs de Clément
  opt.opdict['feat_list'] = ['Dur','AsDec','RappMaxMean','Kurto','Ene']
  opt.opdict['feat_log'] = []
  opt.opdict['feat_train'] = 'clement_train.csv'
  opt.opdict['feat_test'] = 'clement_test.csv'
  opt.do_tri()
  opt.x = opt.xs[0]
  opt.y = opt.ys[0]
  opt.compute_pdfs()

  # Trait plein --> Clément
  # Trait tireté --> moi
  opt.plot_superposed_pdfs(my_gauss)

# ================================================================
def compare_ponsets(set='test'):
  """
  Compare the Ponsets determined with the frequency stack of the spectrogram 
  in function of the spectrogram computation parameters...
  """
  from scipy.io.matlab import mio
  from features_extraction_piton import SeismicTraces
  from options import MultiOptions
  opt = MultiOptions()

  if set == 'test':
    datafiles = glob.glob(os.path.join(opt.opdict['datadir'],'TestSet/SigEve_*'))
    datafiles.sort()
    liste = [os.path.basename(datafiles[i]).split('_')[1].split('.mat')[0] for i in range(len(datafiles))]
    liste = map(int,liste) # sort the list of file following the event number
    liste.sort()

    df_norm = pd.read_csv('%s/features/Piton_testset.csv'%opt.opdict['outdir'],index_col=False)
    df_norm = df_norm.reindex(columns=['Ponset_freq','Dur'])

    df_clement = pd.read_csv('%s/features/clement_test.csv'%opt.opdict['outdir'],index_col=False)
    df_clement = df_clement.reindex(columns=['Dur'])

    df_hash_64 = pd.read_csv('%s/features/HT_Piton_testset.csv'%opt.opdict['outdir'],index_col=False)
    df_hash_64 = df_hash_64.reindex(columns=['Ponset'])

    df_hash_32 = pd.read_csv('%s/features/HT32_Piton_testset.csv'%opt.opdict['outdir'],index_col=False)
    df_hash_32 = df_hash_32.reindex(columns=['Ponset'])

    for ifile,numfile in enumerate(liste):
      file = os.path.join(opt.opdict['datadir'],'TestSet/SigEve_%d.mat'%numfile)
      print ifile,file
      mat = mio.loadmat(file)
      for comp in opt.opdict['channels']:
        ind = (numfile, 'BOR', comp)
        p_norm = df_norm.reindex(index=[str(ind)]).Ponset_freq
        p_hash_64 = df_hash_64.reindex(index=[str(ind)]).Ponset
        p_hash_32 = df_hash_32.reindex(index=[str(ind)]).Ponset
        dur = df_norm.reindex(index=[str(ind)]).Dur * 100
        dur_cl = df_clement.reindex(index=[str(ind)]).Dur * 100

        s = SeismicTraces(mat,comp)
        fig = plt.figure(figsize=(9,4))
        fig.set_facecolor('white')
        plt.plot(s.tr,'k')
        plt.plot([p_norm,p_norm],[np.min(s.tr),np.max(s.tr)],'r',lw=2.,label='norm')
        plt.plot([p_norm+dur,p_norm+dur],[np.min(s.tr),np.max(s.tr)],'r--',lw=2.)
        plt.plot([p_norm+dur_cl,p_norm+dur_cl],[np.min(s.tr),np.max(s.tr)],'--',c='orange',lw=2.)
        plt.plot([p_hash_64,p_hash_64],[np.min(s.tr),np.max(s.tr)],'g',lw=2.,label='hash_64')
        plt.plot([p_hash_32,p_hash_32],[np.min(s.tr),np.max(s.tr)],'y',lw=2.,label='hash_32')
        plt.legend()
        plt.show()
# ================================================================
def search_corr_feat():
  """
  Looks for correlation between attributes.
  Plots each attribute vs each other.
  """
  df = pd.read_csv('../results/Piton/features/Piton_testset.csv',index_col=False)
  df_train = pd.read_csv('../results/Piton/features/Piton_trainset.csv',index_col=False)

  print len(df[df.AsDec>50])
  df = df[df.AsDec<100]
  df_train = df_train[df_train.AsDec<100]

  list_feat = df.columns
  not_feat = ['Ponset_freq','Ponset_grad','EventType','Dur_freq','Dur_grad','Kurto_log','Kurto_log10']
  for ifeat,feat in enumerate(list_feat):
    if feat in not_feat:
      continue
    for f in list_feat[ifeat+1:]:
      if f in not_feat:
        continue
      fig = plt.figure()
      fig.set_facecolor('white')
      plt.scatter(df[feat],df[f],c='k')
      plt.scatter(df_train[feat],df_train[f],c='w')
      plt.xlabel(feat)
      plt.ylabel(f)
      plt.legend(['Test','Training'],numpoints=1)
      plt.show()
# ================================================================
def compare_training():
  """
  Compare the repartition of the training sets :
  decomposition in training (60%), CV (20%) and test (20%) sets.
  """
  from matplotlib.gridspec import GridSpec
  from options import read_binary_file
  libpath = '../lib/Piton'
  list_files = glob.glob(os.path.join(libpath,'learning*'))
  list_files.sort()

  df = pd.read_csv(os.path.join(libpath,'class_train_set.csv'))
  labels = np.array(df.Type.values)

  m = len(labels)
  mtraining = int(0.6*m)
  mcv = int(0.2*m)
  mtest = int(0.2*m)

  nbc, nbl = 3,4
  grid = GridSpec(nbl,nbc*3)
  colors = ['lightskyblue', 'lightcoral']
  fig = plt.figure(figsize=(18,12))
  fig.set_facecolor('white')

  for iter,file in enumerate(list_files):
    if iter%2:
      colors = ['lightskyblue', 'lightcoral']
    else:
      colors = ['powderblue', 'plum']

    dic = read_binary_file(file)
    train = labels[dic[:mtraining]]
    cv = labels[dic[mtraining:mtraining+mcv]]
    test = labels[dic[mtraining+mcv:]]

    prop_train = [len(train[train=='VT']),len(train[train=='EB'])]
    prop_test = [len(test[test=='VT']),len(test[test=='EB'])]
    prop_cv = [len(cv[cv=='VT']),len(cv[cv=='EB'])]

    num = iter%nbc + iter + iter/nbc * nbc
    row = iter/nbc
    col = iter%nbc * 3

    plt.subplot(grid[row,col],aspect='equal')
    plt.pie(prop_train,autopct='%1.1f%%',labels=['VT','EB'],colors=colors)
    plt.text(-0.5,1.4,'Training set')
    plt.text(-0.5,-1.4,r'$m_{training}=%d$'%mtraining)
    plt.subplot(grid[row,col+1],aspect='equal')
    plt.pie(prop_cv,autopct='%1.1f%%',labels=['VT','EB'],colors=colors)
    plt.text(-0.3,1.4,'CV set')
    plt.text(-0.3,-1.4,r'$m_{CV}=%d$'%mcv)
    plt.text(-.5,2.,'Tirage %d'%iter)
    plt.subplot(grid[row,col+2],aspect='equal')
    plt.pie(prop_test,autopct='%1.1f%%',labels=['VT','EB'],colors=colors)
    plt.text(-0.3,1.4,'Test set')
    plt.text(-0.3,-1.4,r'$m_{test}=%d$'%mtest)
  plt.savefig('../results/Piton/figures/tirages.png')
  plt.show()
# ================================================================
def plot_pdf_subsets():
  """
  Plots the pdfs of the training set, CV set and test set on the same 
  figure. One subfigure for each event type. 
  """
  from options import MultiOptions, read_binary_file
  opt = MultiOptions()

  feat_list = [('AsDec',0,1),('Bandwidth',5,0),('CentralF',1,0),('Centroid_time',4,0),('Dur',4,1),('Ene0-5',1,4),('Ene5-10',0,4),('Ene',0,3),('F_low',4,2),('F_up',0,7),('IFslope',7,8),('Kurto',2,0),('MeanPredF',1,4),('PredF',1,4),('RappMaxMean',0,1),('RappMaxMeanTF',4,0),('Skewness',2,5),('TimeMaxSpec',4,0),('Rectilinearity',8,3),('Planarity',1,2)]

  opt.opdict['feat_list'] = opt.opdict['feat_all']
  opt.opdict['feat_filepath'] = '../results/Piton/features/Piton_trainset.csv'
  opt.opdict['label_filename'] = '../lib/Piton/class_train_set.csv'
  x_all, y_all = opt.features_onesta('BOR','Z')
  print len(y_all)
  
  list_files = glob.glob(os.path.join('../lib/Piton','learning*'))
  list_files.sort()

  m = len(y_all)
  mtraining = int(0.6*m)
  mcv = int(0.2*m)
  mtest = int(0.2*m)

  for feat,best,worst in feat_list:
    print feat, best, worst
    fig = plt.figure(figsize=(10,4))
    fig.set_facecolor('white')

    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    # ALL
    opt.x = x_all.reindex(columns=[feat])
    opt.y = y_all.reindex(index=opt.x.index)
    opt.opdict['feat_list'] = [feat]
    opt.compute_pdfs()
    g = opt.gaussians
    ax1.plot(g[feat]['vec'],g[feat]['VT'],'k',lw=2.)
    ax2.plot(g[feat]['vec'],g[feat]['EB'],'k',lw=2.)

    labels = ['best','worst']
    colors = ['r','g']
    b_file = list_files[best]
    w_file = list_files[worst]
    for ifile,file in enumerate([b_file,w_file]):
      dic = read_binary_file(file)

      # TRAINING SET
      opt.x = x_all.reindex(columns=[feat],index=dic[:mtraining])
      opt.y = y_all.reindex(index=dic[:mtraining])
      opt.compute_pdfs()
      g_train = opt.gaussians
      ax1.plot(g_train[feat]['vec'],g_train[feat]['VT'],'-',c=colors[ifile],label=labels[ifile])
      ax2.plot(g_train[feat]['vec'],g_train[feat]['EB'],'-',c=colors[ifile],label=labels[ifile])

      # CV SET
      opt.x = x_all.reindex(columns=[feat],index=dic[mtraining:mtraining+mcv])
      opt.y = y_all.reindex(index=dic[mtraining:mtraining+mcv])
      opt.compute_pdfs()
      g_cv = opt.gaussians
      ax1.plot(g_cv[feat]['vec'],g_cv[feat]['VT'],'--',c=colors[ifile])
      ax2.plot(g_cv[feat]['vec'],g_cv[feat]['EB'],'--',c=colors[ifile])

      # TEST SET
      opt.x = x_all.reindex(columns=[feat],index=dic[mtraining+mcv:])
      opt.y = y_all.reindex(index=dic[mtraining+mcv:])
      opt.compute_pdfs()
      g_test = opt.gaussians
      ax1.plot(g_test[feat]['vec'],g_test[feat]['VT'],':',c=colors[ifile])
      ax2.plot(g_test[feat]['vec'],g_test[feat]['EB'],':',c=colors[ifile])

    ax1.set_title('VT')
    ax2.set_title('EB')
    ax1.legend()
    ax2.legend()
    plt.suptitle(feat)
    #plt.savefig('%s/subsets_Kurto.png'%(opt.opdict['fig_path']))
    plt.show()
      
# ================================================================
def plot_best_worst():
  """
  Plots the pdfs of the training set for the best and worst draws 
  and compare with the whole training set.
  """
  from options import MultiOptions, read_binary_file
  opt = MultiOptions()

  feat_list = [('AsDec',0,1),('Bandwidth',5,0),('CentralF',1,0),('Centroid_time',4,0),('Dur',4,1),('Ene0-5',1,4),('Ene5-10',0,4),('Ene',0,3),('F_low',4,2),('F_up',0,7),('IFslope',7,8),('Kurto',2,0),('MeanPredF',1,4),('PredF',1,4),('RappMaxMean',0,1),('RappMaxMeanTF',4,0),('Skewness',2,5),('TimeMaxSpec',4,0),('Rectilinearity',8,3),('Planarity',1,2)]

  opt.opdict['feat_list'] = opt.opdict['feat_all']
  opt.opdict['feat_filepath'] = '../results/Piton/features/Piton_trainset.csv'
  opt.opdict['label_filename'] = '../lib/Piton/class_train_set.csv'
  x_all, y_all = opt.features_onesta('BOR','Z')
  
  list_files = glob.glob(os.path.join('../lib/Piton','learning*'))
  list_files.sort()

  m = len(y_all)
  mtraining = int(0.6*m)
  mcv = int(0.2*m)
  mtest = int(0.2*m)

  for feat,best,worst in feat_list:
    print feat, best, worst
    fig = plt.figure()
    fig.set_facecolor('white')

    # ALL
    opt.x = x_all.reindex(columns=[feat])
    opt.y = y_all.reindex(index=opt.x.index)
    opt.opdict['feat_list'] = [feat]
    opt.compute_pdfs()
    g = opt.gaussians
    plt.plot(g[feat]['vec'],g[feat]['VT'],'k',lw=2.,label='VT')
    plt.plot(g[feat]['vec'],g[feat]['EB'],'k--',lw=2.,label='EB')

    labels = ['best','worst']
    colors = ['r','g']
    b_file = list_files[best]
    w_file = list_files[worst]
    for ifile,file in enumerate([b_file,w_file]):
      dic = read_binary_file(file)

      # TRAINING SET
      opt.x = x_all.reindex(columns=[feat],index=dic[:mtraining])
      opt.y = y_all.reindex(index=dic[:mtraining])
      opt.compute_pdfs()
      g_train = opt.gaussians
      plt.plot(g_train[feat]['vec'],g_train[feat]['VT'],'-',c=colors[ifile],label=labels[ifile])
      plt.plot(g_train[feat]['vec'],g_train[feat]['EB'],'--',c=colors[ifile])

    plt.legend()
    plt.title(feat)
    #plt.savefig('%s/best_worst.png'%opt.opdict['fig_path'])
    plt.show()
# ================================================================
def dataset_pies():
  """ 
  Plots diagrams of the training and test sets.
  """
  nb_vt_train = 67
  nb_eb_train = 124
  nb_vt_test = 3349
  nb_eb_test = 4133

  colors = ['yellowgreen','gold']

  fig = plt.figure()
  fig.set_facecolor('white')
  plt.subplot(121,aspect='equal')
  plt.pie([nb_vt_train,nb_eb_train],autopct='%1.1f%%',labels=['VT','EB'],colors=colors)
  plt.title('Training set')
  plt.text(-.3,-1.4,r'$m=%d$'%(nb_vt_train+nb_eb_train))
  plt.subplot(122,aspect='equal')
  plt.pie([nb_vt_test,nb_eb_test],autopct='%1.1f%%',labels=['VT','EB'],colors=colors)
  plt.text(-.3,-1.4,r'$m=%d$'%(nb_vt_test+nb_eb_test))
  plt.title('Test set')
  plt.savefig('../results/Piton/figures/repartition.png')
  plt.show()

# ================================================================
if __name__ == '__main__':
  #compare_clement()
  #compare_ponsets(set='test')
  #search_corr_feat()
  #compare_training()
  plot_pdf_subsets()
  #plot_best_worst()
  #dataset_pies()

