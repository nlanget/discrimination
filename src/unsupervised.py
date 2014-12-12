#!/usr/bin/env python
# encoding: utf-8

import os, sys, glob
import pandas as pd
import numpy as np
from options import write_binary_file, read_binary_file
import matplotlib.pyplot as plt

def classifier(opt):

  """
  Classification of the different types of events.
  opt is an object of the class Options()
  By default, the number of classes K that is searched in the dataset is equal to 
  the number of classes in the catalog, but it can be modified directly in the 
  code.
  """
  opt.do_tri()

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

    opt.classname2number()
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

      if opt.opdict['method'] == 'kmeans':
        # K-Mean
        print "********** KMean **********"
        K=2
        CLASS_test = implement_kmean(x_test,K)


      trad,dicocl = {},{}
      for t in opt.types:
        dicocl[t] = []
      for i in range(K):
        auto = CLASS_test[CLASS_test.values.ravel()==i]
        man = y_test.reindex(index=auto.index,columns=['Type'])
        print "Size of class %d : %d"%(i,len(auto))

        nbs = [len(man[man.values.ravel()==j]) for j in opt.types]
        trad[i] = np.array(opt.types)[np.argsort(nbs)]
        for j in range(len(opt.types)):
          print "\tNumber of %s : %d"%(trad[i][j],np.sort(nbs)[j])
          dicocl[trad[i][j]].append(np.sort(nbs)[j])

      if K == len(opt.types):
        types_trad = np.array(opt.types).copy()
        for key in sorted(dicocl):
          types_trad[np.argmax(dicocl[key])] = key
      else:
        types_trad=[]

      ### PLOT DIAGRAMS ###
      if opt.opdict['plot_confusion'] or opt.opdict['save_confusion']:
        if K > 4:
          plot_diagrams(CLASS_test,y_test)
          #results_histo(CLASS_test,y_test)
          results_diagrams(CLASS_test,y_test)
        else:
          all_diagrams(CLASS_test,y_test,trad=types_trad)
        if opt.opdict['save_confusion']:
          savefig = '%s/unsup_diagrams_%df_ini.png'%(opt.opdict['fig_path'],len(opt.opdict['feat_list']))
          plt.savefig(savefig)
          print "Figure saved in %s"%savefig
        if opt.opdict['plot_confusion']:
          plt.show()
        else:
          plt.close()

      opt.x = x_test
      opt.y = y_test

      if opt.opdict['plot_pdf']:
        opt.compute_pdfs()
        g_test = opt.gaussians
      
        opt.y = CLASS_test
        opt.compute_pdfs()
        g_unsup = opt.gaussians

        plot_and_compare_pdfs(g_test,g_unsup)
      
      subsubdic['NumClass'] = CLASS_test.values.ravel()
      if list(types_trad):
        trad_CLASS_test = []
        for i in CLASS_test.values:
          i = int(i)
          trad_CLASS_test.append(types_trad[i])
        subsubdic['StrClass'] = trad_CLASS_test
        subsubdic['Equivalence'] = types_trad
      subdic[b] = subsubdic

    dic_results[opt.trad[isc]] = subdic

  dic_results['header'] = {}
  dic_results['header']['features'] = opt.opdict['feat_list']
  dic_results['header']['types'] = opt.opdict['types']
  dic_results['header']['catalog'] = opt.opdict['label_test']

  print "Save results in file %s"%opt.opdict['result_path']
  write_binary_file(opt.opdict['result_path'],dic_results)

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

def plot_diagrams(y_auto,y_man,save=False):
  """
  Plots the repartition diagrams of each class for the manual classification 
  and for the automatic classification.
  """
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
  plt.figtext(.1,.9,'(a)',fontsize=16)
  plt.figtext(.52,.9,'(b)',fontsize=16)
  if save:
    plt.savefig('../results/Ijen/figures/Unsupervised/unsup_all.png')

# ================================================================

def results_histo(y_auto,y_man,save=False):
  """
  Analysis of the results obtained by unsupervised learning.
  Plots of histograms showing the repartition of the manual classes inside 
  automatic classes. 
  """
  nb_class = len(np.unique(y_auto.Type.values))
  types = np.unique(y_man.Type.values)
  len_class = len(types)
  width = 0.5

  for i in range(nb_class):
    a = y_auto[y_auto.values.ravel()==i]
    b = y_man.reindex(index=a.index,columns=['Type'])

    nbs = [len(b[b.values.ravel()==j]) for j in types]

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

def results_diagrams(y_auto,y_man,save=False):
  """
  Analysis of the results obtained by unsupervised learning.
  Plots of diagrams showing the repartition of the manual classes inside 
  automatic classes. 
  """
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

    nbs = [len(b[b.values.ravel()==j]) for j in types]

    if i < 4:
      nl, nc = 0, i
    else:
      nl, nc = 1, i-4
    plt.subplot(grid[nl,nc],title='Class %d'%i)
    plt.pie(nbs,labels=labels,autopct='%1.1f%%',colors=colors)

  plt.figtext(.1,.9,'(c)',fontsize=16)
  if save:
    plt.savefig('../results/Ijen/figures/Unsupervised/unsup_details.png')

# ================================================================

def plot_and_compare_pdfs(g1,g2):

  c = ['r','b','g','m','c','y','k']#,'orange']
  for feat in sorted(g1):
    fig = plt.figure()
    fig.set_facecolor('white')
    for it,t in enumerate(sorted(g1[feat])):
      if t != 'vec':
        plt.plot(g1[feat]['vec'],g1[feat][t],ls='-',color=c[it],lw=2.,label=t)
    for it,t in enumerate(sorted(g2[feat])):
      if t != 'vec':
        plt.plot(g2[feat]['vec'],g2[feat][t],ls='--',color=c[-it-1],lw=2.,label='Class %d'%t)
    plt.title(feat)
    plt.legend()
    plt.savefig('../results/Ijen/KMEANS/figures/unsup_pdf_%s_ini.png'%feat)
    plt.show()

# ================================================================

def all_diagrams(y_auto,y_man,trad=None):
  """
  Combines plot_diagrams and results_diagrams on the same figure 
  when the number of unsupervised classes is < 5.
  """

  import matplotlib.pyplot as plt
  from matplotlib.gridspec import GridSpec

  nb_class = len(np.unique(y_auto.Type.values))
  types = np.unique(y_man.Type.values)
  len_class = len(types)

  colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral','lightgreen','khaki','plum','powderblue']

  nb_auto = [len(y_auto[y_auto.Type==it]) for it in range(nb_class)]
  nb_man = [len(y_man[y_man.Type==t]) for t in types]

  nb_l, nb_c = 2, nb_class*2
  grid = GridSpec(nb_l,nb_c)
  fig = plt.figure(figsize=(12,8))
  fig.set_facecolor('white')
  ax= fig.add_subplot(grid[0,:nb_c/2])
  ax.pie(nb_man,labels=types,autopct='%1.1f%%',colors=colors)
  #ax.text(.4,-.1,r'# events = %d'%np.sum(nb_man),transform=ax.transAxes)
  ax.axis("equal")

  ax = fig.add_subplot(grid[0,nb_c/2:])
  ax.pie(nb_auto,labels=range(nb_class),autopct='%1.1f%%',colors=colors)
  ax.axis("equal")

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

    nbs = [len(b[b.values.ravel()==j]) for j in types]

    ax = fig.add_subplot(grid[1,2*i:2*i+2])
    ax.pie(nbs,labels=labels,autopct='%1.1f%%',colors=colors)
    if list(trad):
      ax.set_title(r'Class %d $\approx$ %s'%(i,trad[i]))
    else:
      ax.set_title(r'Class %d'%i)
    ax.text(.3,-.1,r'# events = %d'%np.sum(nbs),transform=ax.transAxes)
    if nb_class == 2:
      plt.axis("equal")

  plt.figtext(.1,.92,r'(a) Manual classes     (# events = %d)'%np.sum(nb_man),fontsize=16)
  plt.figtext(.55,.92, r'(b) $K$-means classes',fontsize=16)
  plt.figtext(.1,.47,'(c)',fontsize=16)


# ================================================================

def compare_unsup_indet():

  """
  Essaie de faire un lien entre les événements indéterminés mal classés par 
  LR ou SVM avec classes non-supervisées.
  """
  from matplotlib.gridspec import GridSpec

  print "### COMPARE UNSUP AND SUP ###"
  from results import AnalyseResults
  opt = AnalyseResults()

  m = opt.man
  a = opt.auto
  unsup = read_binary_file('../results/Ijen/KMEANS/results_kmeans_3c_11f_ini')

  nb_auto = [len(opt.auto[opt.auto.Type==t]) for t in opt.opdict['types']]
  NB_class = len(opt.opdict['types'])

  for cl in opt.opdict['types']:
  #for cl in ['?']:
    m = opt.man[opt.man.Type==cl]
    a = opt.auto.reindex(index=m.index)

    colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']

    opt.data_for_LR()
    opt.opdict['channels'] = 'Z'
    opt.opdict['stations'] = ['IJEN']
    for sta in opt.opdict['stations']:
      for comp in opt.opdict['channels']:
        u = pd.DataFrame(index=unsup[(sta, comp)][0]['list_ev'],columns=['Type','NumType'])
        u['Type'] = unsup[(sta, comp)][0]['StrClass']
        u['NumType'] = unsup[(sta, comp)][0]['NumClass']
        u = u.reindex(index=m.index)
        trad = unsup[(sta, comp)][0]['Equivalence']


        fig = plt.figure(figsize=(12,8))
        fig.set_facecolor('white')
        nb_l, nb_c = 2, NB_class*2
        grid = GridSpec(nb_l,nb_c)

        ax = fig.add_subplot(grid[0,:nb_c/2])
        ax.pie(nb_auto,labels=opt.opdict['types'],autopct='%1.1f%%',colors=colors)
        ax.text(.4,-.1,r'# events = %d'%np.sum(nb_auto),transform=ax.transAxes)
        ax.axis("equal")

        nbs = [len(a[a.Type==t]) for t in opt.opdict['types']]
        ax = fig.add_subplot(grid[0,nb_c/2:])
        ax.pie(nbs,labels=opt.opdict['types'],autopct='%1.1f%%',colors=colors)
        ax.text(.4,-.1,r'# events = %d'%np.sum(nbs),transform=ax.transAxes)
        ax.axis("equal")

        lab_c = np.array(trad).copy()
        for it,t in enumerate(opt.opdict['types']):
          i = np.where(np.array(trad)==t)[0][0]
          lab_c[it] = i

        for it,t in enumerate(opt.opdict['types']):
          ared = a[a.Type==t]
          ured = u.reindex(index=ared.index)
          nbs = [len(ured[ured.Type==ty]) for ty in opt.opdict['types']]
          ax = fig.add_subplot(grid[1,2*it:2*it+2])
          ax.pie(nbs,labels=lab_c,autopct='%1.1f%%',colors=colors)
          ax.text(.3,-.1,r'# %s = %d'%(t,np.sum(nbs)),transform=ax.transAxes)
          #ax.set_title(t)

        plt.figtext(.1,.92,'(a) %s'%opt.opdict['method'].upper(),fontsize=16)
        plt.figtext(.55,.92,'(b) Manual repartition of %s'%cl,fontsize=16)
        plt.figtext(.1,.45,r'(c) $K$-means',fontsize=16)
        for it, t in enumerate(trad):
          plt.figtext(.3+it*.15,.45,r'%s $\approx$ %s'%(it,trad[it]))
        plt.savefig('../results/Ijen/KMEANS/figures/unsup_compSVM_%s.png'%cl)

  plt.show()


def plot_waveforms():

  """
  Plot the waveforms of unsupervised classes.
  """

  from matplotlib.gridspec import GridSpec
  from options import read_binary_file, Options
  from obspy.core import read
  opt = Options()
  DIC = read_binary_file(opt.opdict['result_path'])

  for stac in sorted(DIC):
    if stac == 'header':
      continue
    station = stac[0]
    comp = stac[1]
    datapath = glob.glob(os.path.join(opt.opdict['datadir'],station,'*%s*'%comp))[0]
    for tir in sorted(DIC[stac]):
      list_ev = DIC[stac][tir]['list_ev']
      nclass = DIC[stac][tir]['NumClass']
      K = len(np.unique(nclass))
      fig = plt.figure()
      fig.set_facecolor('white')
      grid = GridSpec(2*K,3)
      for j,N in enumerate(np.unique(nclass)):
        index = list(np.where(nclass==N)[0])
        ev = list_ev[index]
        permut = np.random.permutation(ev)
        for i in range(3):
          E = permut[i]
          file = glob.glob(os.path.join(datapath,'*%s_%s*'%(str(E)[:8],str(E)[8:])))[0]
          st = read(file)
          st.filter('bandpass',freqmin=1,freqmax=10)
          if i in [0,1]:
            ax = fig.add_subplot(grid[2*j,i+1])
          else:
            ax = fig.add_subplot(grid[2*j+1,:])
          ax.plot(st[0],'k')
          ax.set_axis_off()
        ax = fig.add_subplot(grid[2*j,0])
        ax.text(.2,.5,N,transform=ax.transAxes)
        ax.set_axis_off()

  save = True
  if save:
    savename = '%s/WF_K%dclass.png'%(opt.opdict['fig_path'],K)
    print "Saved in %s"%savename
    plt.savefig(savename)
  plt.show()
          



if __name__ == '__main__':
  compare_unsup_indet()
  #plot_waveforms()
