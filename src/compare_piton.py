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
  opt.opdict['feat_list'] = ['Dur','AsDec','RappMaxMean','Kurto','KRapp']
  opt.opdict['feat_log'] = ['AsDec','RappMaxMean','Kurto']
  #opt.opdict['feat_list'] = ['Ene']
  #opt.opdict['feat_log'] = ['Ene']
  opt.do_tri()
  opt.x = opt.xs[0]
  opt.y = opt.ys[0]
  opt.x.columns = opt.opdict['feat_list']
  opt.compute_pdfs()
  my_gauss = opt.gaussians

  if 'Kurto' in opt.opdict['feat_list'] and 'RappMaxMean' in opt.opdict['feat_list']:
    fig = plt.figure()
    fig.set_facecolor('white')
    plt.plot(np.log(opt.x.Kurto),np.log(opt.x.RappMaxMean),'ko')
    plt.xlabel('Kurto')
    plt.ylabel('RappMaxMean')
    plt.show()

  # Les calculs de Clément
  #opt.opdict['feat_list'] = ['Dur','AsDec','RappMaxMean','Kurto','Ene']
  opt.opdict['feat_log'] = []
  opt.opdict['feat_train'] = 'clement_train.csv'
  opt.opdict['feat_test'] = 'clement_test.csv'
  opt.do_tri()
  opt.x = opt.xs[0]
  opt.y = opt.ys[0]
  opt.compute_pdfs()

  # Trait plein --> Clément
  # Trait tireté --> moi
  opt.plot_superposed_pdfs(my_gauss,save=False)

# ================================================================
def plot_soutenance():
  """
  Plot des PDFs des 4 attributs définis par Clément pour le ppt 
  de la soutenance.
  """
  from options import MultiOptions
  opt = MultiOptions()
  opt.opdict['channels'] = ['Z']

  #opt.opdict['feat_train'] = 'clement_train.csv'
  #opt.opdict['feat_test'] = 'clement_test.csv'
  opt.opdict['feat_list'] = ['AsDec','Dur','Ene','KRapp']
  #opt.opdict['feat_log'] = ['AsDec','Dur','Ene','KRapp']
  opt.do_tri()
  opt.x = opt.xs[0]
  opt.y = opt.ys[0]
  opt.compute_pdfs()

  gauss = opt.gaussians

  fig = plt.figure(figsize=(12,2.5))
  fig.set_facecolor('white') 
  for ifeat,feat in enumerate(sorted(gauss)):
    ax = fig.add_subplot(1,4,ifeat+1)
    ax.plot(gauss[feat]['vec'],gauss[feat]['VT'],ls='-',c='b',lw=2.)
    ax.plot(gauss[feat]['vec'],gauss[feat]['EB'],ls='-',c='r',lw=2.)
    ax.set_title(feat)
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_ticklabels('')
    ax.yaxis.set_ticks_position('left')
    ax.yaxis.set_ticklabels('')
    if ifeat == 0:
      ax.legend(['VT','EB'],loc=1,prop={'size':10})
  plt.savefig('/home/nadege/Dropbox/Soutenance/pdfs.png')
  plt.show()

# ================================================================
def compare_lissage():
  """
  Comparaison des kurtosis avec deux lissages différents.
  """

  plot_envelopes()

  from options import MultiOptions
  opt = MultiOptions()
  opt.opdict['channels'] = ['Z']

  # Lissage sur des fenêtres de 0.5 s 
  opt.opdict['feat_list'] = ['Kurto']
  opt.opdict['feat_log'] = ['Kurto']
  opt.do_tri()
  opt.x = opt.xs[0]
  opt.y = opt.ys[0]
  opt.x.columns = opt.opdict['feat_list']
  opt.compute_pdfs()
  gauss_stand = opt.gaussians

  # Lissage sur des fenêtres de 1 s
  opt.opdict['feat_train'] = '0610_Piton_trainset.csv'
  opt.opdict['feat_test'] = '0610_Piton_testset.csv'
  opt.do_tri()
  opt.x = opt.xs[0]
  opt.y = opt.ys[0]
  opt.compute_pdfs()
  gauss_1s = opt.gaussians

  # Lissage sur des fenêtres de 5 s
  opt.opdict['feat_train'] = '1809_Piton_trainset.csv'
  opt.opdict['feat_test'] = '1809_Piton_testset.csv'
  opt.do_tri()
  opt.x = opt.xs[0]
  opt.y = opt.ys[0]
  opt.compute_pdfs()
  gauss_5s = opt.gaussians

  # Lissage sur des fenêtres de 10 s
  opt.opdict['feat_train'] = '0510_Piton_trainset.csv'
  opt.opdict['feat_test'] = '0510_Piton_testset.csv'
  opt.do_tri()
  opt.x = opt.xs[0]
  opt.y = opt.ys[0]
  opt.compute_pdfs()
  gauss_10s = opt.gaussians

  ### PLOT OF SUPERPOSED PDFs ###
  fig = plt.figure(figsize=(12,2.5))
  fig.set_facecolor('white') 
  for feat in sorted(opt.gaussians):
    maxi = int(np.max([gauss_stand[feat]['vec'],gauss_1s[feat]['vec'],gauss_5s[feat]['vec'],gauss_10s[feat]['vec']]))

    ax1 = fig.add_subplot(141)
    ax1.plot(gauss_stand[feat]['vec'],gauss_stand[feat]['VT'],ls='-',c='b',lw=2.,label='VT')
    ax1.plot(gauss_stand[feat]['vec'],gauss_stand[feat]['EB'],ls='-',c='r',lw=2.,label='EB')
    ax1.set_xlim([0,maxi])
    ax1.set_xlabel(feat)
    ax1.set_title('0.5 s')
    ax1.legend(prop={'size':10})

    ax2 = fig.add_subplot(142)
    ax2.plot(gauss_1s[feat]['vec'],gauss_1s[feat]['VT'],ls='-',c='b',lw=2.)
    ax2.plot(gauss_1s[feat]['vec'],gauss_1s[feat]['EB'],ls='-',c='r',lw=2.)
    ax2.set_xlim([0,maxi])
    ax2.set_xlabel(feat)
    ax2.set_title('1 s')
    ax2.set_yticklabels('')

    ax3 = fig.add_subplot(143)
    ax3.plot(gauss_5s[feat]['vec'],gauss_5s[feat]['VT'],ls='-',c='b',lw=2.)
    ax3.plot(gauss_5s[feat]['vec'],gauss_5s[feat]['EB'],ls='-',c='r',lw=2.)
    ax3.set_xlim([0,maxi])
    ax3.set_xlabel(feat)
    ax3.set_title('5 s')
    ax3.set_yticklabels('')

    ax4 = fig.add_subplot(144)
    ax4.plot(gauss_10s[feat]['vec'],gauss_10s[feat]['VT'],ls='-',c='b',lw=2.)
    ax4.plot(gauss_10s[feat]['vec'],gauss_10s[feat]['EB'],ls='-',c='r',lw=2.)
    ax4.set_xlim([0,maxi])
    ax4.set_xlabel(feat)
    ax4.set_title('10 s')
    ax4.set_yticklabels('')

    #plt.savefig('%s/features/comp_%s.png'%(opt.opdict['outdir'],feat))
    plt.show()


def plot_envelopes():
  """
  Plot d'un VT et d'un EB avec des enveloppes calculées avec 
  plusieurs paramètres de lissage.
  """
  from  options import read_binary_file
  from features_extraction_piton import process_envelope
  datadir = '../data/Piton/envelope'
  
  fig = plt.figure()
  fig.set_facecolor('white')

  colors = ['r','b','g','y']

  ### EB ###
  tr_eb = read_binary_file('%s/trace_EB'%datadir)
  time = np.linspace(0,len(tr_eb)*0.01,len(tr_eb))
  
  env_51 = process_envelope(tr_eb,w=51)
  env_101 = process_envelope(tr_eb,w=101)
  env_501 = process_envelope(tr_eb,w=501)
  env_1001 = process_envelope(tr_eb,w=1001)

  ax1 = fig.add_subplot(211)
  #ax1.plot(time,tr_eb,'k')
  ax1.plot(time[:-1],env_51,c=colors[0],label='0.5 s')
  ax1.plot(time[:-1],env_101,c=colors[1],label='1 s')
  ax1.plot(time[:-1],env_501,c=colors[2],lw=2.,label='5 s')
  ax1.plot(time[:-1],env_1001,c=colors[3],lw=2.,label='10 s')
  from mpl_toolkits.axes_grid1.inset_locator import inset_axes
  axins = inset_axes(ax1,width="30%",height="60%",loc=1)
  i1, i2 = 6000, 8000
  ax1.axvspan(time[i1],time[i2],color='gray',alpha=.3)
  axins.plot(time[i1:i2],env_51[i1:i2],c=colors[0])
  axins.plot(time[i1:i2],env_101[i1:i2],c=colors[1])
  axins.plot(time[i1:i2],env_501[i1:i2],c=colors[2],lw=2.)
  axins.plot(time[i1:i2],env_1001[i1:i2],c=colors[3],lw=2.)
  axins.xaxis.set_ticks_position('bottom')
  axins.yaxis.set_ticklabels('')
  axins.yaxis.set_visible(False)
  ax1.set_title('Eboulement')
  ax1.set_xlim([0,time[-1]])
  ax1.set_xticklabels('')
  ax1.legend(loc=2,prop={'size':10})

  ### VT ###
  tr_vt = read_binary_file('%s/trace_VT'%datadir)

  env_51 = process_envelope(tr_vt,w=51)
  env_101 = process_envelope(tr_vt,w=101)
  env_501 = process_envelope(tr_vt,w=501)
  env_1001 = process_envelope(tr_vt,w=1001)

  ax2 = fig.add_subplot(212)
  #ax2.plot(tr_vt,'k')
  ax2.plot(time[:-1],env_51,c=colors[0])
  ax2.plot(time[:-1],env_101,c=colors[1])
  ax2.plot(time[:-1],env_501,c=colors[2],lw=2.)
  ax2.plot(time[:-1],env_1001,c=colors[3],lw=2.)
  from mpl_toolkits.axes_grid1.inset_locator import inset_axes
  axins = inset_axes(ax2,width="30%",height="70%",loc=1)
  i1, i2 = 3000,5000
  ax2.axvspan(time[i1],time[i2],color='gray',alpha=.3)
  axins.plot(time[i1:i2],env_51[i1:i2],c=colors[0])
  axins.plot(time[i1:i2],env_101[i1:i2],c=colors[1])
  axins.plot(time[i1:i2],env_501[i1:i2],c=colors[2],lw=2.)
  axins.plot(time[i1:i2],env_1001[i1:i2],c=colors[3],lw=2.)
  axins.xaxis.set_ticks_position('bottom')
  axins.yaxis.set_ticklabels('')
  axins.yaxis.set_visible(False)
  ax2.set_title('Volcano-tectonique')
  ax2.set_xlim([0,time[-1]])
  ax2.set_xlabel('Time (s)')

  plt.figtext(0.03,0.89,'(a)')
  plt.figtext(0.03,0.46,'(b)')
  #plt.savefig('../results/Piton/features/envelopes.png')
  plt.show()

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
  opt.opdict['feat_filename'] = '../results/Piton/features/Piton_trainset.csv'
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
    plt.savefig('%s/subsets_%s.png'%(opt.opdict['fig_path'],feat))
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
  opt.opdict['feat_log'] = ['AsDec','Ene','Kurto','RappMaxMean']
  opt.opdict['feat_filename'] = '../results/Piton/features/Piton_trainset.csv'
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
    plt.savefig('%s/best_worst_%s.png'%(opt.opdict['fig_path'],feat))
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
  plot_soutenance()
  #compare_lissage()
  #compare_ponsets(set='test')
  #search_corr_feat()
  #compare_training()
  #plot_pdf_subsets()
  #plot_best_worst()
  #dataset_pies()

