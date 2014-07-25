#!/usr/bin/env python
# encoding: utf-8

import os,glob,sys
from obspy.core import read,utcdatetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from LR_functions import comparison
from do_classification import confusion, create_training_set

# ================================================================

def read_binary_file(filename):
  import cPickle
  with open(filename,'rb') as test:
    my_depickler = cPickle.Unpickler(test)
    DIC = my_depickler.load()
    test.close()
  return DIC

# ================================================================

def plot_rates(DIC):
  """
  Plot of the correct classification rates of the test set against the training set 
  for each iteration and for each class.
  """
  fig = plt.figure()
  fig.set_facecolor("white")
  colors = ['b','c','m','c','k','y','g','r']
  markers = ['*','o','h','v','d','s','d','h']
  k = 0
  for ikey,key in enumerate(sorted(DIC[0])):
    if key != "i_train":
      boot_tr, boot_test = np.array([]), np.array([])
      if k == 1:
        ikey = ikey-1
      for iter in sorted(DIC):
        if iter != 'features':
          boot_tr = np.append(boot_tr,DIC[iter][key]['rate_%s'%key][0])
          boot_test = np.append(boot_test,DIC[iter][key]['rate_%s'%key][1])
      plt.plot(boot_tr,boot_test,marker=markers[ikey],color=colors[ikey],lw=0,label=key)
    else:
      k = 1
  plt.legend(numpoints=1,loc='upper left')
  plt.xlim([0,100])
  plt.ylim([0,100])
  plt.plot([0,100],[0,100],'k--')
  plt.xlabel('% training set')
  plt.ylabel('% test set')
  plt.savefig('../results/Ijen/figures/1B1_test-vs-train_svm-red.png')
  plt.show()
  #boot_tr = np.reshape(boot_tr,(len(DIC),len(DIC[0])))
  #boot_test = np.reshape(boot_test,(len(DIC),len(DIC[0])))

# ================================================================

def plot_training(DIC):

  df_ref = pd.read_csv('../lib/Ijen/Ijen_reclass_indet.csv')
  types = np.unique(df_ref.Type.values)
  lmax = len(df_ref)

  #types = ['LowFrequency']
  best, worst = np.array([]), np.array([])
  for type in types:
    fig = plt.figure()
    fig.set_facecolor("white")
    all_itrain, p_train = [],[]
    for iter in sorted(DIC):
      if iter != 'features':
        dates = DIC[iter]["i_train"]
        df = df_ref.reindex(index=dates)
        df = df[df.Type==type]
        b = iter*np.ones(len(df),dtype=int)
        plt.plot(df['i'],b,'k.')
        plt.ylim([-1,len(DIC)])
        plt.text(lmax-500,iter+.1,"%.2f%%"%DIC[iter]["%s"%type]["rate_%s"%type][0])
        all_itrain.append(df['i'].index)
        p_train.append(DIC[iter]["%s"%type]["rate_%s"%type][0])
    plt.title(type)
    i_p_train_sort = np.argsort(p_train)
    b = i_p_train_sort[-3:]
    w = i_p_train_sort[:3]
    i_best = np.intersect1d(np.intersect1d(DIC[b[0]]["i_train"],DIC[b[1]]["i_train"]),DIC[b[2]]["i_train"])
    i_worst = np.intersect1d(np.intersect1d(DIC[w[0]]["i_train"],DIC[w[1]]["i_train"]),DIC[w[2]]["i_train"])
    i_best_excl = np.intersect1d(i_best,np.setxor1d(i_best,i_worst))
    i_worst_excl = np.intersect1d(i_worst,np.setxor1d(i_best,i_worst))
    best = np.union1d(best,i_best_excl)
    worst = np.union1d(worst,i_worst_excl)
    #plot_compare_pdf(all_itrain,type,p_train,DIC['features'])
    #plot_train_events(i_best_excl,type)
    plot_train_pdf(i_best_excl,types,DIC['features'])
    plot_train_pdf(i_worst_excl,types,DIC['features'])
  print len(best)
  print len(worst)
  plt.show()

# ================================================================

def plot_compare_pdf(i_train,t,perc,list_features):

  from scipy.stats.kde import gaussian_kde
  df = pd.read_csv('../results/Ijen/ijen_0309.csv')
  df.index = map(str,list(df[df.columns[0]]))
  for feat in list_features:
    a = df.reindex(columns=[feat])
    fig = plt.figure()
    fig.set_facecolor('white')
    lss = '-'
    for i in range(len(i_train)):
      b = a.reindex(index=i_train[i])
      vec = np.linspace(b.min()[feat],b.max()[feat],200)
      kde = gaussian_kde(b[feat].values)
      c = np.cumsum(kde(vec))[-1]
      if i > 6:
        lss = '--'
      plt.plot(vec,kde(vec)/c,ls=lss,label="%d - %.2f%%"%(i,perc[i]))
    plt.title("%s - %s"%(t,feat))
    plt.legend(numpoints=1)
    plt.show()

# ================================================================

def plot_train_pdf(i_train,types,list_features):

  df = pd.read_csv('../results/Ijen/ijen_0309.csv')
  df.index = map(str,list(df[df.columns[0]]))
  df = df.reindex(index=i_train)
  #df = df.dropna(how='any')
  y = df.reindex(columns=['Type'])
  x = df.reindex(columns=list_features)

  from Ijen_extract_features import plot_pdf_feat
  plot_pdf_feat(x,y,types,save=False)

# ================================================================

def plot_train_events(i_best,type):

  for date in i_best:
    date_ok = date[:8]+'_'+date[8:]
    file = glob.glob('/home/nadege/Desktop/IJEN_GOOD/DATA/*%s*.SAC'%date_ok)[0]
    print file
    st = read(file)
    st[0].plot()

# ================================================================

def plot_features_vs(DIC):
  """
  Plot des attributs les uns vs les autres pour voir s'ils sont corrélés.
  """
  list_feat = DIC['features']
  df_ref = pd.read_csv('../results/Ijen/ijen_0605_1sta.csv')
  df_ref = df_ref.reindex(columns=list_feat)
  print df_ref.columns
  print list_feat

  for ifeat,feat1 in enumerate(list_feat):
    for feat2 in list_feat[ifeat+1:]:
      print feat1,feat2
      fig = plt.figure()
      fig.set_facecolor('white')
      plt.plot(df_ref[feat1].values,df_ref[feat2].values,'ko')
      plt.xlabel(feat1)
      plt.ylabel(feat2)
      plt.show()

# ================================================================

def class_histograms(DIC):
  """
  Le titre de la figure correspond à la classe extraite automatiquement 
  avec le nombre d'événements.
  Chaque barre de l'histogramme correspond à la classe d'appartenance 
  d'origine (donc classification manuelle) des événements. 
  """

  types = sorted(DIC[0])
  if 'i_train' in types:
    types.remove('i_train')
  print types
  N = np.array(range(len(types)-1))

  colors_all = create_color_scale(types)
  width = 0.1
  for icl,cl in enumerate(types):
    t = np.setxor1d(np.array(types),np.array([cl]))
    if 'LowFrequency' in t:
      t[t=='LowFrequency'] = 'LF'
    if 'HarmonikTremor' in t:
      t[t=='HarmonikTremor'] = 'Tremor'
    if 'TektonikLokal' in t:
      t[t=='TektonikLokal'] = 'TektoLokal'
    fig, ax = plt.subplots()
    fig.set_facecolor('white')
    s_cl = ""
    nb_cl = 0
    for iter in sorted(DIC):
      if iter != 'features':
        s_cl = s_cl + " %s"%DIC[iter][cl]['nb']
        nb_cl = nb_cl + DIC[iter][cl]['nb']
        DIC[iter][cl]['nb_other'].sort()
        nb = [tup[1] for tup in DIC[iter][cl]['nb_other']]
        colors = associate_color(colors_all,[tup[0] for tup in DIC[iter][cl]['nb_other']])
        #nb.insert(icl,DIC[iter][cl]['nb'])
        rects = ax.bar(N+iter*width,nb,width,color=colors)
    ax2 = ax.twinx()
    ylab = []
    for yt in ax.get_yticks()*100./(nb_cl/10):
      ylab.append("%.1f"%yt)
    ax2.set_yticks(range(len(ax.get_yticks())))
    ax2.set_yticklabels(ylab)
    ax2.set_ylabel("% wrt the total number of classified events")

    ax.set_xticks(N+len(t)/2*width)
    ax.set_xticklabels(t)
    ax.set_ylabel("Number of events")
    ax.set_title("%s \n%s"%(cl,s_cl))

  plt.show()

# ================================================================

def event_classes(DIC,save=False):

  """
  - for each event, looks for the class it belongs to for each training set ; 
  if the event is not classified, class label is set to 99 by default.
  - it the labels are different from a set to another, then displays the 
  waveform and ask the user to assign a class manually.
  - if save=True: saves the new classification in a new file.
  """

  from options import Options
  opt = Options()
  all = np.array(map(str,list(opt.raw_df[opt.raw_df.columns[0]])))
  types = np.unique(opt.raw_df.Type).values
  df = opt.raw_df
  df.index = all

  manuals = opt.manuals
  manuals.index = map(str,list(manuals[manuals.columns[0]]))

  new_class = []

  all_classes = []
  for event in all:
    l = []
    for iter in sorted(DIC):
      if iter != 'features':
        marker = 0
        for it,t in enumerate(types):
          if event in DIC[iter][t]['index_ok']:
            l.append(it)
            marker = 1
            continue
          for tup in DIC[iter][t]['i_other']:
             if event in tup[1]:
               itype = np.where(types==t)[0][0]
               l.append(itype)
               marker = 1
        if marker == 0:
          l.append(99)

    print event, l
    cl = np.unique(np.array(l))
    date_ok = event[:8]+'_'+event[8:]
    mant = manuals.reindex(index=[event],columns=['Type']).values[0][0]
    m = 0
    if len(cl) == 1 and cl[0] != 99 and types[cl[0]] != mant:
      m = 1
    if len(cl) > 1 or m==1:
      if 99 not in cl:
        print "\t", mant, types[cl]
      else:
        print "\t", mant, types[cl[:-1]], 'unclassified'
        m = 2
      if m == 2 and len(types[cl[:-1]])== 1:
        final_cl = types[cl[:-1]][0]
      elif len(cl) == 1: # if the automatic classification is systematically the same whatever training set is used, BUT different from the manual classification, then...
        final_cl = types[cl][0]
      #else:
        file = glob.glob('/home/nadege/Desktop/IJEN_GOOD/DATA/*%s*.SAC'%date_ok)[0]
        a = df.reindex(index=[event])
        st = read(file)
        st[0].plot()
        for feat in opt.opdict['feat_list']:
          if feat != 'NbPeaks':
            opt.plot_one_pdf(feat,[(mant,a[feat].values,final_cl)])
        if save:
          final_cl = str(raw_input("\t Quelle classe ? "))
        print "\n"
    else:
      if l[0] != 99:
        final_cl = types[l[0]]
      else:
        final_cl = 'Unclass'
    new_class.append(final_cl)
  if save:
    df_new = manuals.reindex(columns=['Filename','Type'])
    df_new['NewType'] = new_class
    df_new.to_csv('../lib/Ijen/svm_reclass.csv')

# ================================================================

def search_and_reclass(DIC,cl):
  """
  Pour l'extraction One-vs-All.
  Recherche des événements "mal classés" communs à chacune des extractions.
  """
  l = np.array([])
  for tir in sorted(DIC):
    if tir != 'features':
      for tup in DIC[tir][cl]['i_other']:
        if tup[0] == '?':
          l = np.concatenate((l,tup[1]))

  from collections import Counter
  dic = Counter(list(l))
  ld = []
  for d in dic.keys():
    if dic[d] > 1:
      ld.append(d)
  print len(ld)

  from classification import spectral_content
  manual_file = '../lib/Ijen/Ijen_reclass_indet.csv'
  datadir = '/media/disk/CLASSIFICATION/ijen'
  spectral_content(manual_file,datadir,list_dates=ld)

# ================================================================

def plot_curves(filename):
  """
  Evolution de la taille et de la composition du test set au cours 
  des extractions pour la méthode "one-by-one".
  """

  EXT = read_binary_file(filename)
  for key in sorted(EXT[0]):
    all_nb = []
    for num_ext in sorted(EXT):
      all_nb.append(EXT[num_ext][key])

    all_nb = np.array(all_nb)
    fig = plt.figure()
    fig.set_facecolor('white')
    for i in range(all_nb.shape[1]):
      plt.plot(range(len(EXT)),all_nb[:,i],'-')
    plt.xlabel('Extraction number')
    plt.ylabel('Number of events in the test set')
    plt.title('%s'%key.split('_')[1])
  plt.show()

# ================================================================

def plot_curves_bis(DIC,cl):
  """
  Evolution du nombre d'événements bien classés (ie communs au 
  catalogue manuel) en fonction du nombre d'événements classés
  pour une classe cl donnée.
  """
  nb, nb_com = [],[]
  for tir in sorted(DIC):
    if tir == 'features':
      continue
    dic = DIC[tir][cl]
    nb.append(dic['nb'])
    nb_com.append(dic['nb_common'])

  fig = plt.figure()
  fig.set_facecolor('white')
  plt.plot(nb,nb_com,'kv')
  nbmin = np.min(np.array(nb_com))
  nbmax = np.max(np.array(nb))
  print nbmin, nbmax
  nbmin = np.around(nbmin/100.)*100
  nbmax = np.around(nbmax/100.)*100
  print nbmin, nbmax
  plt.plot([nbmin,nbmax],[nbmin,nbmax],'k:')
  #plt.plot([150,300],[130,230],'r:')
  #plt.xlim([100,300])
  #plt.ylim([100,300])
  plt.xlabel('Number of classified events')
  plt.ylabel('Number of common events')
  plt.title(cl)
  #plt.savefig('../results/Ijen/figures/evolution_VA.png')
  plt.show()


# ================================================================

def read_extraction_results(filename):

  from results import AnalyseResultsExtraction
  res = AnalyseResultsExtraction()

  from obspy.core import utcdatetime,read

  DIC = read_binary_file(filename)
  #class_histograms(DIC) # plot extraction results as histograms
  #plot_rates(DIC) # plot extraction results as training set vs test set
  #plot_training(DIC)
  #event_classes(DIC)
  #search_and_reclass(DIC,'Tremor')
  #plot_features_vs(DIC)
  #plot_curves_bis(DIC,'VulkanikB')

# ================================================================
if __name__ == '__main__' :
  read_extraction_results('../results/Ijen/1B1/1B1_ijen_redac_IJEN_svm')
  #plot_curves('../results/Ijen/1B1/stats_OBO')
