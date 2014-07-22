#!/usr/bin/env python
# encoding: utf-8

import os, sys, glob
import pandas as pd
import numpy as np


def classifier(opt):

  """
  Classification of the different types of events.
  opt is an object of the class Options()
  """
  opt.data_for_LR()
  opt.tri()

  X = opt.x
  Y = opt.y

  dic_results = {}
  for isc in sorted(opt.xs):

    print "==========",opt.trad[isc],"=========="
    subdic = {}

    if isc > 0:
      if opt.trad[isc][0] == sta_prev:
        marker_sta = 1
      else:
        marker_sta = 0
        sta_prev = opt.trad[isc][0]
    else:
      marker_sta = 0
      sta_prev = opt.trad[isc][0]

    if len(opt.xs[isc]) == 0:
      continue

    x_test = opt.xs[isc]
    y_test = opt.ys[isc]

    opt.types = np.unique(y_test.Type)
    K = len(opt.types)

    for b in range(opt.opdict['boot']):
      print "\n-------------------- # iter: %d --------------------\n"%(b+1)

      subsubdic = {}

      print "# types in the test set:",len(opt.types)

      subsubdic['list_ev'] = np.array(y_test.index)

      x_test.index = range(x_test.shape[0])
      y_test.index = range(y_test.shape[0])
      print x_test.shape, y_test.shape
      if x_test.shape[0] != y_test.shape[0]:
        print "Test set: Incoherence in x and y dimensions"
        sys.exit()

      if opt.opdict['method'] == 'kmean':
        # K-Mean
        print "********** KMean **********"
        CLASS_test = implement_kmean(x_test,K)

      if K > 4:
        plot_diagrams(CLASS_test,y_test)
        #results_histo(CLASS_test,y_test)
        results_diagrams(CLASS_test,y_test)
      else:
        all_diagrams(CLASS_test,y_test)

      opt.x = x_test
      opt.y = y_test
      opt.compute_pdfs()
      g_test = opt.gaussians
      
      opt.y = CLASS_test
      opt.compute_pdfs()
      g_unsup = opt.gaussians

      plot_and_compare_pdfs(g_test,g_unsup)
      sys.exit()

      subsubdic['%'] = pourcentages
      trad_CLASS_test = []
      for i in CLASS_test:
        i = int(i)
        trad_CLASS_test.append(opt.types[i])
      subsubdic['classification'] = trad_CLASS_test
      subdic[b] = subsubdic

    dic_results[opt.trad[isc]] = subdic

  dic_results['features'] = opt.opdict['feat_list']

# ================================================================

def implement_kmean(x,nb_class):
  """
  Implements K-Mean.
  """
  from sklearn.cluster import KMeans

  kmean = KMeans(k=nb_class)
  kmean.fit(x.values)
  y_kmean = kmean.predict(x.values)
  y_kmean = pd.DataFrame(y_kmean)
  y_kmean.index = x.index
  y_kmean.columns = ['Type']
  return y_kmean


# ================================================================

def plot_diagrams(y_auto,y_man):
  """
  Plots the repartition diagrams of each class for the manual classification 
  and for the automatic classification.
  """
  import matplotlib.pyplot as plt
  nb_class = len(np.unique(y_auto.Type.values))
  types = np.unique(y_man.Type.values)
  colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral','lightgreen','khaki','plum','powderblue']

  nb_auto = [len(y_auto[y_auto.Type==it]) for it in range(nb_class)]
  nb_man = [len(y_man[y_man.Type==t]) for t in types]

  fig = plt.figure(figsize=(9,4))
  fig.set_facecolor('white')
  plt.subplot(121,title='Manual classes')
  plt.pie(nb_man,labels=types,autopct='%1.1f%%',colors=colors)
  plt.subplot(122,title='Automatic classes')
  plt.pie(nb_auto,labels=range(nb_class),autopct='%1.1f%%',colors=colors)
  #plt.savefig('../results/Ijen/figures/Unsupervised/unsup_all.png')
  plt.show()

# ================================================================

def results_histo(y_auto,y_man):
  """
  Analysis of the results obtained by unsupervised learning.
  Plots of histograms showing the repartition of the manual classes inside 
  automatic classes. 
  """
  import matplotlib.pyplot as plt
  nb_class = len(np.unique(y_auto.Type.values))
  types = np.unique(y_man.Type.values)
  len_class = len(types)
  width = 0.5

  for i in range(nb_class):
    a = y_auto[y_auto.values.ravel()==i]
    b = y_man.reindex(index=a.index,columns=['Type'])
    print i, len(a)

    nbs = [len(b[b.values.ravel()==j]) for j in types]
    print nbs

    # plot histograms
    fig,ax = plt.subplots()
    fig.set_facecolor('white')
    rects = ax.bar(range(len_class),nbs,width)
    ax.set_xticks(range(len_class))
    ax.set_xticklabels(types)
    ax.set_xlabel("Correspondence to manual classification")
    ax.set_ylabel("Number of events")
    ax.set_title("Class %d - size = %d"%(i,len(a)))


    def autolabel(rects):
      # attach some text labels
      for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%d'%int(height),
                ha='center', va='bottom')

    autolabel(rects)
    plt.show()

# ================================================================

def results_diagrams(y_auto,y_man):
  """
  Analysis of the results obtained by unsupervised learning.
  Plots of diagrams showing the repartition of the manual classes inside 
  automatic classes. 
  """
  import matplotlib.pyplot as plt
  from matplotlib.gridspec import GridSpec
  nb_class = len(np.unique(y_auto.Type.values))
  types = np.unique(y_man.Type.values)
  len_class = len(types)
  colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral','lightgreen','khaki','plum','powderblue']

  if nb_class <= 4:
    nb_l, nb_c = 1, nb_class
  else:
    nb_l, nb_c = 2, int(np.ceil(nb_class/2.))

  labels = types.copy()
  if 'VulkanikB' in labels:
    labels[labels=='VulkanikB'] = 'VB'
  if 'VulkanikA' in labels:
    labels[labels=='VulkanikA'] = 'VA'
  if 'Tektonik' in labels:
    labels[labels=='Tektonik'] = 'Tecto'
  if 'Longsoran' in labels:
    labels[labels=='Longsoran'] = 'Eb'
  if 'Hembusan' in labels:
    labels[labels=='Hembusan'] = 'Hem.'

  fig = plt.figure(figsize=(5*nb_c,4.5*nb_l))
  fig.set_facecolor('white')
  grid = GridSpec(nb_l,nb_c)
  for i in range(nb_class):
    a = y_auto[y_auto.values.ravel()==i]
    b = y_man.reindex(index=a.index,columns=['Type'])
    print i, len(a)

    nbs = [len(b[b.values.ravel()==j]) for j in types]
    print nbs

    if i < 4:
      nl, nc = 0, i
    else:
      nl, nc = 1, i-4
    plt.subplot(grid[nl,nc],title='Class %d'%i)
    plt.pie(nbs,labels=labels,autopct='%1.1f%%',colors=colors)

  #plt.savefig('../results/Ijen/figures/Unsupervised/unsup_details.png')
  plt.show()

# ================================================================

def plot_and_compare_pdfs(g1,g2):

  import matplotlib.pyplot as plt
  c = ['r','b','g','m','c','y','k']
  for feat in sorted(g1):
    fig = plt.figure()
    fig.set_facecolor('white')
    for it,t in enumerate(sorted(g1[feat])):
      if t != 'vec':
        plt.plot(g1[feat]['vec'],g1[feat][t],ls='-',color=c[it],label=t)
    for it,t in enumerate(sorted(g2[feat])):
      if t != 'vec':
        plt.plot(g2[feat]['vec'],g2[feat][t],ls='--',color=c[-it-1],label='Class %d'%t)
    plt.title(feat)
    plt.legend()
    #plt.savefig('../results/Ijen/figures/Unsupervised/pdf_%s.png'%feat)
    plt.show()

# ================================================================

def all_diagrams(y_auto,y_man):
  """
  Combines plot_diagrams and results_diagrams on the same figure 
  when the number of classes is < 5.
  """

  import matplotlib.pyplot as plt
  from matplotlib.gridspec import GridSpec

  nb_class = len(np.unique(y_auto.Type.values))
  types = np.unique(y_man.Type.values)
  len_class = len(types)

  colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']

  nb_auto = [len(y_auto[y_auto.Type==it]) for it in range(nb_class)]
  nb_man = [len(y_man[y_man.Type==t]) for t in types]

  nb_l, nb_c = 2, len_class*2
  grid = GridSpec(nb_l,nb_c)
  fig = plt.figure(figsize=(12,8))
  fig.set_facecolor('white')
  plt.subplot(grid[0,:nb_c/2],title='Manual classes')
  plt.pie(nb_man,labels=types,autopct='%1.1f%%',colors=colors)
  plt.axis("equal")
  plt.subplot(grid[0,nb_c/2:],title='Automatic classes')
  plt.pie(nb_auto,labels=range(nb_class),autopct='%1.1f%%',colors=colors)
  plt.axis("equal")

  labels = types.copy()
  if 'VulkanikB' in labels:
    labels[labels=='VulkanikB'] = 'VB'
  if 'VulkanikA' in labels:
    labels[labels=='VulkanikA'] = 'VA'
  if 'Tektonik' in labels:
    labels[labels=='Tektonik'] = 'Tecto'
  if 'Longsoran' in labels:
    labels[labels=='Longsoran'] = 'Eb'
  if 'Hembusan' in labels:
    labels[labels=='Hembusan'] = 'Hem.'

  for i in range(nb_class):
    a = y_auto[y_auto.values.ravel()==i]
    b = y_man.reindex(index=a.index,columns=['Type'])
    print i, len(a)

    nbs = [len(b[b.values.ravel()==j]) for j in types]
    print nbs

    plt.subplot(grid[1,2*i:2*i+2],title='Class %d'%i)
    plt.pie(nbs,labels=labels,autopct='%1.1f%%',colors=colors)
    plt.axis("equal")

  plt.figtext(.1,.95,'(a)',fontsize=16)
  plt.figtext(.1,.5,'(b)',fontsize=16)
  #plt.savefig('../results/Ijen/figures/Unsupervised/unsup_diagrams.png')
  plt.show()
