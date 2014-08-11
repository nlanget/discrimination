#!/usr/bin/env python
# encoding: utf-8

import os,glob,sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def new_catalogue(opt):

  m = opt.man
  a = opt.auto

  for date in m.index:
    if m.Type[date] == '?' and a.Type[date] != '?':
      m.Type[date] = a.Type[date]

  m['Date'] = m.index
  m.to_csv('../lib/Ijen/Ijen_3class_all_SVM.csv',index=False)

# ================================================================

def plot_on_pdf(opt):

  """
  Affiche sur chaque densité de probabilité la valeur des attributs 
  des événements manuels indéterminés qui ont été classés dans une 
  autre classe automatiquement.
  """

  m = opt.man
  a = opt.auto
 
  opt.data_for_LR()
  comp = 'Z'
  opt.opdict['stations'] = ['IJEN']
  for sta in opt.opdict['stations']:
    opt.x, opt.y = opt.features_onesta(sta,'Z')
    opt.verify_index()
    opt.types = np.unique(opt.y.Type.values)

    for t in opt.types:
      if t == '?':
        continue
      list_dates = []
      for date in m.index:
        if m.Type[date] == '?' and a.Type[date] == t:
          list_dates.append(date)
      print "Found %d events automatically classified as '%s' instead of '?'"%(len(list_dates),t)
      for feat in opt.opdict['feat_list']:
        df = opt.x.reindex(columns=[feat],index=list_dates)
        list_coord = [m.Type[list_dates].values,df[feat].values,a.Type[list_dates].values]
        list_coord = np.array(list_coord)
        opt.plot_one_pdf(feat,list_coord)

# ================================================================
 
def plot_waveforms(opt):
  """
  Affiche les formes d'ondes des événements manuels indéterminés 
  qui ont été classés dans une autre classe automatiquement.
  """
  from obspy.core import read

  m = opt.man
  a = opt.auto
  opt.types = np.unique(a.Type.values)

  comp = 'Z'
  for sta in opt.opdict['stations']:

    for t in opt.types:
      if t == '?' or t == 'Tremor':
        continue
      list_dates = []
      for date in m.index:
        if m.Type[date] == '?' and a.Type[date] == t:
          date = str(date)
          files = glob.glob(os.path.join(opt.opdict['datadir'],sta,'EHZ.D/*%s_%s*'%(date[:8],date[8:])))
          if len(files) > 0:
            file = files[0]
            st = read(file)
            st.filter('bandpass',freqmin=1,freqmax=10)
            st.plot()

# ================================================================

def compare_pdfs_reclass():
  """
  Affiche et compare les pdfs avant et après reclassification automatique.
  """
  from results import AnalyseResults
  opt = AnalyseResults()

  opt.opdict['stations'] = ['IJEN']
  opt.opdict['channels'] = ['Z']
  opt.opdict['Types'] = ['Tremor','VulkanikB','?']
  opt.opdict['feat_list'] = ['Centroid_time','Dur','Ene0-5','F_up','Growth','Kurto','RappMaxMean','RappMaxMeanTF','Skewness','TimeMaxSpec','Width']

  for sta in opt.opdict['stations']:
    for comp in opt.opdict['channels']:
      opt.opdict['label_filename'] = '%s/Ijen_3class_all.csv'%opt.opdict['libdir']
      opt.x, opt.y = opt.features_onesta(sta,comp)
      opt.classname2number()
      opt.compute_pdfs()
      g1 = opt.gaussians

      opt.opdict['label_filename'] = '%s/Ijen_reclass_all.csv'%opt.opdict['libdir']
      opt.x, opt.y = opt.features_onesta(sta,comp)
      opt.classname2number()
      opt.compute_pdfs()
      g2 = opt.gaussians

      c = ['r','b','g']
      for feat in opt.opdict['feat_list']:
        fig = plt.figure()
        fig.set_facecolor('white')
        for it,t in enumerate(opt.types):
          plt.plot(g1[feat]['vec'],g1[feat][t],ls='-',color=c[it],label=t)
          plt.plot(g2[feat]['vec'],g2[feat][t],ls='--',color=c[it])
        plt.title(feat)
        plt.legend()
        #plt.savefig('../results/Ijen/comp_BrutReclass_%s.png'%feat)
        plt.show()
      
# ================================================================

def compare_pdfs_train():
  """
  Affiche et compare les pdfs des différents training sets.
  """
  from options import MultiOptions
  opt = MultiOptions()

  opt.opdict['stations'] = ['IJEN']
  opt.opdict['channels'] = ['Z']
  opt.opdict['Types'] = ['Tremor','VulkanikB','?']
 
  opt.opdict['train_file'] = '%s/train_10'%(opt.opdict['libdir'])
  opt.opdict['label_filename'] = '%s/Ijen_reclass_all.csv'%opt.opdict['libdir']

  train = opt.read_binary_file(opt.opdict['train_file'])
  nb_tir = len(train)

  for sta in opt.opdict['stations']:
    for comp in opt.opdict['channels']:
      opt.x, opt.y = opt.features_onesta(sta,comp)

  X = opt.x
  Y = opt.y
  c = ['r','b','g']
  lines = ['-','--','-.',':','-','--','-.',':','*','v']
  features = opt.opdict['feat_list']
  for feat in features:
    print feat
    opt.opdict['feat_list'] = [feat]
    fig = plt.figure()
    fig.set_facecolor('white')
    for tir in range(nb_tir):
      tr = map(int,train[tir])
      opt.x = X.reindex(index=tr,columns=[feat])
      opt.y = Y.reindex(index=tr)
      opt.classname2number()
      opt.compute_pdfs()
      g = opt.gaussians

      for it,t in enumerate(opt.types):
        plt.plot(g[feat]['vec'],g[feat][t],ls=lines[tir],color=c[it])
    plt.title(feat)
    plt.legend(opt.types)
    plt.show()

# ================================================================

def plot_test_vs_train():
  """
  For multiple training set draws
  """
  import cPickle
  path = '../results/Ijen'
  filenames = ['SVM/results_svm_8c_52f','SVM/results_svm_8c_52f_trFix','SVM/results_svm_8c_8f','SVM/results_svm_8c_8f_trFix']
  labels = ['SVM prop','SVM fix','SVM prop 8f','SVM fix 8f']

  fig = plt.figure()
  fig.set_facecolor('white')
  colors = ['b','c','m','y','g','r','b','c','m','y','g','r']
  markers = ['*','o','h','v','d','s','o','v','s','*','h','d']
  k = 0
  for filename in filenames:
    filename = os.path.join(path,filename)
    with open(filename,'r') as file:
      my_depickler = cPickle.Unpickler(file)
      dic = my_depickler.load()
      file.close()

    for key in sorted(dic):
      if key == 'features':
        continue
      DIC = dic[key]
      continue
 
    p_tr,p_test = [],[]
    for i in sorted(DIC):
      p_tr.append(DIC[i]['%'][0])
      p_test.append(DIC[i]['%'][1])
    plt.plot(p_tr,p_test,marker=markers[k],color=colors[k],lw=0,label=labels[k])
    k = k+1
  plt.legend(numpoints=1,loc='upper left')
  plt.plot([0,100],[0,100],'k--')
  plt.xlim([60,100])
  plt.ylim([60,100])
  plt.figtext(.7,.15,'Sur-apprentissage')
  plt.xlabel('% training set')
  plt.ylabel('% test set')
  #plt.savefig('../results/Ijen/figures/SVM_training.png')
  plt.show()


if __name__ == '__main__':
  from results import AnalyseResults
  res = AnalyseResults()

  #plot_test_vs_train()
  #new_catalogue(res)
  #plot_on_pdf(res)
  plot_waveforms(res)
  #compare_pdfs_reclass()
  #compare_pdfs_train()
