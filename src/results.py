#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import pandas as pd
import os,sys,glob
from options import MultiOptions
import matplotlib.pyplot as plt
from options import read_binary_file

class AnalyseResults(MultiOptions):
  """
  Functions to analyse and interpret the classification results from a result file.
  For logistic regression and SVM only.
  """
  def __init__(self):
    MultiOptions.__init__(self)

    print "\nANALYSIS OF %s"%self.opdict['result_path']
    self.opdict['class_auto_file'] = 'auto_class.csv'
    self.opdict['class_auto_path'] = '%s/%s/%s'%(self.opdict['outdir'],self.opdict['method'].upper(),self.opdict['class_auto_file'])

    self.bootstrap_overall()
    self.bootstrap_per_class()
    #sys.exit()
    self.concatenate_results()
    self.display_results()


  def read_result_file(self):
    """
    Reads the file containing the results
    """
    dic = read_binary_file(self.opdict['result_path'])
    self.opdict['feat_list'] = dic['header']['features']
    self.opdict['label_filename'] = '%s/%s'%(self.opdict['libdir'],dic['header']['catalog'])
    print "Nb features :", len(self.opdict['feat_list'])
    print "Types :", dic['header']['types']
    self.results = dic
    self.opdict['stations'] = [key[0] for key in sorted(dic)]
    self.opdict['channels'] = [key[1] for key in sorted(dic)]
    self.opdict['Types'] = dic['header']['types']
    del dic['header']


  def concatenate_results(self):
    """
    Does a synthesis of all classifications (in cases where this is a multi-station or 
    multi-component classification)
    Stores the automatic classification into a .csv file.
    The index of the DataFrame structure contains the list of events.
    The columns of the DataFrame structure contain, for each event : 
      - Type : automatic class
      - Nb : number of stations implied in the classification process
      - NbDiff : number of different classes found
      - % : proportion of the final class among all classes. If the proportion are equal (for ex., 50-50), write ?
    """
    self.read_result_file()

    list_ev = []
    for key in sorted(self.results):
      list_ev = list_ev + list(self.results[key][0]['list_ev'])
    list_ev = np.array(list_ev)
    list_ev_all = np.unique(list_ev)

    df = pd.DataFrame(index=list_ev_all,columns=sorted(self.results),dtype=str)
    for key in sorted(self.results):
      for iev,event in enumerate(self.results[key][0]['list_ev']):
        df[key][event] = self.results[key][0]['classification'][iev]

    struct = pd.DataFrame(index=list_ev_all,columns=['Type','Nb','NbDiff','%'],dtype=str)
    for iev in range(len(df.index)):
      w = np.where(df.values[iev]!='n')
      struct['Nb'][df.index[iev]] = len(w[0])
      t = df.values[iev][w[0]]
      t_uniq = np.unique(t)
      struct['NbDiff'][df.index[iev]] = len(t_uniq) 

      if len(t_uniq) == 1:
        struct['Type'][df.index[iev]] = t_uniq[0]
        struct['%'][df.index[iev]] = 100.
      else:
        prop = []
        for i,ty in enumerate(t_uniq):
          a = len(t[t==ty])
          prop.append(a*100./len(w[0]))
        if len(np.unique(prop)) > 1:
          imax = np.argmax(np.array(prop))
          struct['Type'][df.index[iev]] = t_uniq[imax]
          struct['%'][df.index[iev]] = np.max(np.array(prop))
        else:
          struct['Type'][df.index[iev]] = '?'
          struct['%'][df.index[iev]] = 1./len(t_uniq)*100

    struct.to_csv(self.opdict['class_auto_path'])
    #self.plot_stats(struct)


  def plot_stats(self,struct):
    """
    Plot some statistics from the class_auto file.
    Use of the pandas package.
    """
    histos = struct.reindex(columns=['%','Nb'])
    histos.hist()
    plt.ylabel('Number of events')

    fig = plt.figure()
    fig.set_facecolor('white')
    plt.plot(struct['Nb'].values,struct['%'].values,'k+')
    plt.xlim([0,struct['Nb'].max()+1])
    plt.ylim([20,101])
    plt.xlabel('Number of classifications')
    plt.ylabel('% of same classification')
    plt.show()


  def bootstrap_overall(self):
    """
    In cases where several training set draws were carried out.
    Displays the mean global classification rate as well as the standard deviation.
    """
    self.read_result_file()
    print "OVERALL STATISTICS"
    for key in sorted(self.results):
      print key
      p_test, p_train = [],[]
      for tir in sorted(self.results[key]):
        p_train.append(self.results[key][tir]['%'][0])
        p_test.append(self.results[key][tir]['%'][1])

      print "\t Training", np.mean(p_train), np.std(p_train)
      print "\t   MAX", np.max(p_train), "tirage", np.argmax(p_train)
      print "\t   MIN", np.min(p_train), "tirage", np.argmin(p_train)
      print "\t Test", np.mean(p_test), np.std(p_test)
      print "\t   MAX", np.max(p_test), "tirage", np.argmax(p_test)
      print "\t   MIN", np.min(p_test), "tirage", np.argmin(p_test)
      print "\n"
      

  def bootstrap_per_class(self):
    """
    In cases where several training set draws were carried out.
    Displays the mean classification rate as well as the standard deviation for 
    each class of event.
    """
    from options import name2num
    self.read_result_file()
    labels = self.read_classification()
    labels.index = labels.Date
    labels = labels.reindex(columns=['Type'])

    print "STATISTICS PER CLASS (TEST SET)"
    for key in sorted(self.results):
      print key

      p = {}
      for t in self.opdict['Types']:
        p[t] = []

      for tir in sorted(self.results[key]):
        lab = labels.reindex(index=self.results[key][tir]['list_ev'])

        df = pd.DataFrame(index=self.results[key][tir]['list_ev'])
        df['Type'] = self.results[key][tir]['classification']

        for t in self.opdict['Types']:
          m = lab[lab.Type==t]
          a = df[df.Type==t]
          ev_m = np.array(m.index)
          ev_a = np.array(a.index)
          common = np.intersect1d(ev_a,ev_m)
          p[t].append(len(common)*1./len(m)*100)

      for t in self.opdict['Types']:
        print "\t",t, np.mean(p[t]), np.std(p[t])
        print "\t  MAX", np.max(p[t]), np.argmax(p[t])
        print "\t  MIN", np.min(p[t]), np.argmin(p[t])
      print "\n" 


  def read_manual_auto_files(self):
    """
    Reads the file with manual classification.
    Reads the file with automatic classification.
    """
    print self.opdict['result_path']
    print self.opdict['label_filename']
    print self.opdict['class_auto_path']

    self.auto = pd.read_csv(self.opdict['class_auto_path'],index_col=False)
    self.auto = self.auto.reindex(columns=['Type'])
    self.auto.Type[self.auto.Type=='LowFrequency'] = 'LF'

    self.man = self.read_classification()
    self.man.index = self.man.Date
    self.man = self.man.reindex(columns=['Type'],index=self.auto.index)

    self.man = self.man.dropna(how='any')
    self.auto = self.auto.reindex(index=self.man.index)

    for i in range(len(self.man.values)):
      self.man.values[i][0] = self.man.values[i][0].replace(" ","")


  def display_results(self):
    """
    Displays the success rate of the classification process.
    """
    self.read_manual_auto_files()

    N = len(self.man)
    list_man = self.man.values.ravel()
    list_auto = self.auto.values.ravel()
    sim = np.where(list_man==list_auto)[0]
    list_auto_sim = list_auto[sim]
    print "%% of well classified events : %.2f"%(len(sim)*100./N)
    print "\n"

    types = np.unique(list_auto)
    print "\nPercentages of well-classified events"
    for t in types:
      if t != '?':
        l1 = len(np.where(list_man==t)[0])
        l2 = len(np.where(list_auto_sim==t)[0])
        print "%% for type %s : %.2f (%d out of %d)"%(t,l2*100./l1,l2,l1)

    print "\nRepartition of automatic classification"
    for t in types:
      l1 = len(np.where(list_man==t)[0])
      l2 = len(np.where(list_auto==t)[0])
      print "%s : manual = %d vs auto = %d"%(t,l1,l2)
    print "\n"


  def plot_confusion(self):
    """
    Plots the confusion matrix (test set only).
    """
    from do_classification import plot_confusion_mat
    from sklearn.metrics import confusion_matrix
    self.do_tri()
    self.classname2number()

    m = self.man
    a = self.auto
    for i in self.numt:
      m['Type'][m.Type==self.types[i]] = i
      a['Type'][a.Type==self.types[i]] = i
    a = a[a.Type!='?']
    m = m.reindex(index=a.index)

    cmat = confusion_matrix(m.values[:,0],a.values[:,0])
    plot_confusion_mat(cmat,self.types,'Test',self.opdict['method'])
    if self.opdict['save_confusion']:
      plt.savefig('%s/figures/test_%s.png'%(self.opdict['outdir'],self.opdict['result_file'][8:]))
    plt.show()



class AnalyseResultsExtraction(MultiOptions):

  def __init__(self):
    MultiOptions.__init__(self)
    print "ANALYSIS OF %s"%self.opdict['result_path']
    self.results = read_binary_file(self.opdict['result_path'])
    self.opdict['feat_list'] = self.results['features']
    del self.results['features']

    self.do_analysis()


  def do_analysis(self):
    for sta in self.opdict['stations']:
      for comp in self.opdict['channels']:
        self.x, self.y = self.features_onesta(sta,comp)
        self.verify_index()

        if len(self.x) == 0:
          continue

        self.composition_dataset()
        #self.repartition_extraction()

        #self.plot_all_diagrams()
        #self.plot_diagrams_one_draw()
        #self.plot_pdf_extract()
        self.unclass_histo()
        self.unclass_diagram()
        if 'unclass_all' in self.__dict__.keys():
          self.analyse_unclass()
        if self.opdict['method'] == 'ova':
          self.search_repetition()
          self.analyse_repeat()


  def repartition_extraction(self):
    """
    Affiche le diagramme de répartition des classes extraites par l'extracteur.
    A comparer entre extracteurs, et avec le diagramme de répartition manuel.
    """
    colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral','lightgreen','khaki','plum','powderblue']

    if self.opdict['method'] == '1b1':
      for tir in sorted(self.results):
        Ns = [len(self.results[tir][cl]['i_test']) for cl in sorted(self.results[tir])]
        N = np.max(Ns)
        nbs = [self.results[tir][cl]['nb'] for cl in sorted(self.results[tir])]
        labels = sorted(self.results[tir])
        nbs.append(N-np.sum(nbs))
        labels.append('unclass')

        fig = plt.figure(figsize=(6,6))
        fig.set_facecolor('white')
        plt.pie(nbs,labels=labels,autopct='%1.1f',colors=colors)
        plt.figtext(0.4,0.1,'# events = %d'%np.sum(nbs))
        plt.title('Repartition after the extraction')
      plt.show()

    elif self.opdict['method'] == 'ova':
      self.search_repetition()
      self.repeat = map(int,list(self.repeat))
      for tir in sorted(self.results):
        nbs, labels = [],[]
        N = len(self.results[tir]['i_test'])
        for cl in sorted(self.results[tir]):
          if cl in ['i_train','i_test']:
            continue
          labels.append(cl)
          nb_ini = self.results[tir][cl]['nb']
          for ev in self.repeat:
            if ev in self.results[tir][cl]['i_other'] or ev in self.results[tir][cl]['index_ok']:
              nb_ini = nb_ini-1
          nbs.append(nb_ini)

        nbs.append(N-np.sum(nbs))
        labels.append('unclass')

        fig = plt.figure(figsize=(6,6))
        fig.set_facecolor('white')
        plt.pie(nbs,labels=labels,autopct='%1.1f',colors=colors)
        plt.figtext(0.4,0.1,'# events = %d'%np.sum(nbs))
        plt.title('Repartition after the extraction')
      plt.show()


  def unclass_histo(self):
    """
    Classe d'appartenance d'origine (classification manuelle) des événements 
    non classés à l'issue de l'extraction.
    """

    tuniq = np.unique(self.y.Type)
    N = np.array(range(len(np.unique(tuniq))))
    width = 0.1
    colors = [(0,0,1),(1,0,0),(0,1,0),(1,1,0),(0,0,0),(0,1,1),(1,0,1),(1,1,1),(.8,.5,0),(0,.5,.5)]

    self.unclass_all = np.array([])

    fig, ax = plt.subplots()
    fig.set_facecolor('white')
    for iter in sorted(self.results):
      rest = np.array([])
      if len(self.results[iter]['i_test']) == len(self.results[iter])-2:
        y_tir = self.y.reindex(index=map(int,self.results[iter]['i_test'][0]))
      else:
        y_tir = self.y.reindex(index=map(int,self.results[iter]['i_test']))
      for cl in sorted(self.results[iter]):
        if cl in ['i_train','i_test']:
          continue
        for tup in self.results[iter][cl]['i_other']:
          print tup[1]
          raw_input("pause")
          rest = np.hstack((rest,tup[1]))
        rest = np.append(rest,self.results[iter][cl]['index_ok'])

      rest_uniq = np.unique(rest)
      rest_uniq = np.array(map(int,list(rest_uniq)))
      unclass = np.setdiff1d(np.array(y_tir.index),rest_uniq)
      self.unclass_all = np.concatenate((self.unclass_all,unclass))

      unclass_types = self.y.reindex(index=unclass,columns=['Type']).values
      nb = [len(unclass_types[unclass_types==t]) for t in tuniq]
      rects = ax.bar(N+iter*width,nb,width,color=colors)

    ax2 = ax.twinx()
    ylab = []
    for yt in ax.get_yticks()*100./len(y_tir):
      ylab.append("%.1f"%yt)
    ax2.set_yticks(range(len(ax.get_yticks())))
    ax2.set_yticklabels(ylab)
    ax2.set_ylabel("% wrt the total number of events of the dataset")

    types = tuniq.values
    if 'VulkanikA' in types:
      types[types=='VulkanikA'] = 'VA'
    if 'VulkanikB' in types:
      types[types=='VulkanikB'] = 'VB'
    ax.set_xticks(N+len(self.results)/2*width)
    ax.set_xticklabels(types)
    ax.set_xlabel('Manual class of unclassified events')
    ax.set_ylabel('Number of unclassified events')
    ax.set_title("Unclassified events")

    plt.show()


  def unclass_diagram(self):
    """
    Représentation en diagrammes de la répartition des événements non classés à l'issue des extractions.
    Une figure par tirage.
    (a) Répartition en classes manuelles des non classés.
    (b) Proportions, pour chaque classe manuelle, des événements non classés.
    """
    from matplotlib.gridspec import GridSpec

    types = np.unique(self.y.Type.values)
    N = np.array(range(len(np.unique(types))))
    width = 0.1
    colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral','lightgreen','khaki','plum','powderblue']

    self.unclass_all = np.array([])

    for iter in sorted(self.results):
      fig = plt.figure(figsize=(12,8))
      fig.set_facecolor('white')
      grid = GridSpec(3,4)
      rest = np.array([])
      if 'i_test' in sorted(self.results[iter]):
        y_tir = self.y.reindex(index=map(int,self.results[iter]['i_test']))
      for cl in sorted(self.results[iter]):
        if cl in ['i_train','i_test']:
          continue
        for tup in self.results[iter][cl]['i_other']:
          rest = np.hstack((rest,tup[1]))
        rest = np.append(rest,self.results[iter][cl]['index_ok'])

      rest_uniq = np.unique(rest)
      rest_uniq = np.array(map(int,list(rest_uniq)))
      unclass = np.setdiff1d(np.array(y_tir.index),rest_uniq)
      self.unclass_all = np.concatenate((self.unclass_all,unclass))      

      unclass_types = self.y.reindex(index=unclass,columns=['Type']).values
      nb = [len(unclass_types[unclass_types==t]) for t in types]

      ax = fig.add_subplot(grid[:2,:2],aspect=1,title='Unclassified events')
      ax.pie(nb,labels=types,autopct='%1.1f%%',colors=colors)
      ax.text(.35,0,"# events = %d"%np.sum(nb),transform=ax.transAxes)

      orga = [(0,2),(0,3),(1,2),(1,3),(2,0),(2,1),(2,2),(2,3)]
      for it,t in enumerate(types):
        nb = len(unclass_types[unclass_types==t])
        nb_tot = len(y_tir[y_tir.Type==t])
        il = orga[it][0]
        icol = orga[it][1]
        ax = fig.add_subplot(grid[il,icol],aspect=1,title=t)
        ax.pie([nb,nb_tot-nb],explode=(.05,0),labels=['unclass',''],autopct='%1.1f%%',colors=((.5,.5,.5),'w'))
        ax.text(0.2,-.02,"# events = %d"%nb_tot,transform=ax.transAxes)
      plt.figtext(.1,.91,'(a)',fontsize=18)
      plt.figtext(.5,.91,'(b)',fontsize=18)
    plt.show()


  def analyse_unclass(self):
    """
    Recherche les événements qui ne sont systématiquement pas classés.
    Affiche le diagramme de répartition des classes manuelles.
    Plotte les pdfs correspondantes (limitées aux 3 classes ayant le plus grand 
    d'événements non classés).
    """
    from collections import Counter
    dic = Counter(list(self.unclass_all))
    ld = []
    for d in dic.keys():
      if dic[d] == 10:
        ld.append(d)
  
    colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral','lightgreen','khaki','plum','powderblue']
 
    types = np.unique(self.y.Type.values)
    unc_type = self.y.reindex(index=ld,columns=['Type']).values
    nb = [len(unc_type[unc_type==t]) for t in types]
    nb_dataset = [len(self.y[self.y.Type==t]) for t in types]

    from matplotlib.gridspec import GridSpec
    grid = GridSpec(1,2)
    fig = plt.figure(figsize=(12,5))
    fig.set_facecolor('white')
    ax = fig.add_subplot(grid[0,0])
    ax.pie(nb_dataset,labels=types,autopct='%1.1f%%',colors=colors)
    ax.set_title('(a) Whole dataset')
    ax.text(.4,0,"# events = %d"%np.sum(nb_dataset),transform=ax.transAxes)

    ax = fig.add_subplot(grid[0,1])
    ax.pie(nb,labels=types,autopct='%1.1f%%',colors=colors)
    ax.set_title('(b) Unclassified events')
    ax.text(.4,0,"# events = %d"%np.sum(nb),transform=ax.transAxes)
    plt.show()

    self.types = types[np.argsort(nb)][-3:]
    self.compute_pdfs()
    g_ok = self.gaussians

    X = self.x.copy()
    Y = self.y.copy()
    self.x = self.x.reindex(index=ld)
    self.y = self.y.reindex(index=ld)
    if len(self.y) > 0:
      self.compute_pdfs()
      g_unclass = self.gaussians

      colors = ['r','g','b']
      for feat in sorted(g_ok):
        if feat == 'NbPeaks':
          continue
        fig = plt.figure()
        fig.set_facecolor('white')
        for it,t in enumerate(self.types):
          plt.plot(g_ok[feat]['vec'],g_ok[feat][t],ls='-',c=colors[it],label=t)
          plt.plot(g_unclass[feat]['vec'],g_unclass[feat][t],ls='--',c=colors[it])
        plt.title(feat)
        plt.legend()
        plt.show()

    self.x = X
    self.y = Y


  def search_repetition(self):
    """
    Evénements qui sont classés plusieurs fois, et ce, dans des classes différentes.
    One-vs-All method only.
    """

    types = self.y.Type

    N = np.array(range(len(np.unique(self.y.Type))))
    width = 0.1
    colors = [(0,0,1),(1,0,0),(0,1,0),(1,1,0),(0,0,0),(0,1,1),(1,0,1),(1,1,1),(.8,.5,0),(0,.5,.5)]
    max = 0

    fig,ax = plt.subplots()
    fig.set_facecolor('white')
    self.repeat = np.array([])
    for iter in sorted(self.results):
      rest = np.array([])
      for cl in sorted(self.results[iter]):
        if cl == 'i_train' or cl == 'i_test':
          continue
        for tup in self.results[iter][cl]['i_other']:
          rest = np.hstack((rest,tup[1]))
        rest = np.append(rest,self.results[iter][cl]['index_ok'])

      rest = np.array(map(int,list(rest)))
      rest_uniq,i_rest_uniq = np.unique(rest,return_index=True)
      i_not_uniq = np.setxor1d(np.array(range(len(rest))),i_rest_uniq)
      rest_not_uniq = rest[i_not_uniq]
      self.repeat = np.concatenate((self.repeat,rest_not_uniq))

      notu_types = self.y.reindex(index=rest_not_uniq,columns=['Type']).values
      nb = [len(notu_types[notu_types==t]) for t in np.unique(self.y.Type)]
      rects = ax.bar(N+iter*width,nb,width,color=colors)
      if np.max(nb) > max:
        max = np.max(nb)

    types = np.unique(self.y.Type.values)
    if 'VulkanikA' in types:
      types[types=='VulkanikA'] = 'VA'
    if 'VulkanikB' in types:
      types[types=='VulkanikB'] = 'VB'
    ax.set_xticks(N+len(self.results)/2*width)
    ax.set_xticklabels(types)
    ax.set_xlabel('Manual class of repeated events')
    ax.set_ylabel('Number of repeated events')
    ax.set_title("Repeated events")
  
    plt.show()


  def analyse_repeat(self):
    """
    Recherche les événements qui sont systématiquement classés plus d'une 
    fois.
    Affiche le diagramme de répartition des classes manuelles.
    """
    from collections import Counter
    dic = Counter(list(self.repeat))
    ld = []
    for d in dic.keys():
      if dic[d] == 10:
        ld.append(d)

    colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral','lightgreen','khaki','plum','powderblue']

    if len(ld) > 0:
      types = np.unique(self.y.Type.values)
      rep_type = self.y.reindex(index=ld,columns=['Type']).values
      nb = [len(rep_type[rep_type==t]) for t in types]

      from matplotlib.gridspec import GridSpec
      grid = GridSpec(1,1)
      fig = plt.figure(figsize=(6,6))
      fig.set_facecolor('white')
      plt.subplot(grid[0,0])
      plt.pie(nb,labels=types,autopct='%1.1f%%',colors=colors)
      plt.title('Repeated events')
      plt.show()


  def plot_all_diagrams(self):
    """
    Le titre de la figure correspond à la classe extraite automatiquement 
    avec le nombre d'événements (une figure par classe).
    Chaque couple de diagrammes circulaires correspond à un tirage donné :
      * le 1er donne le taux de bonne extraction de la classe considérée 
    (par rapport au nombre total extrait)
      * le 2ème montre la provenance (ie classes manuelles) des événements 
    classés automatiquement dans la classe extraite mais qui n'ont pas la 
    bonne classe d'origine (= reste).
    """
    from matplotlib.gridspec import GridSpec

    types = sorted(self.results[0])
    if 'i_train' in types:
      types.remove('i_train')
    if 'i_test' in types:
      types.remove('i_test')
    print types
    N = np.array(range(len(types)-1))

    colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral','lightgreen','khaki','plum','powderblue']
    width = 0.1
    for icl,cl in enumerate(types):
      t = np.setxor1d(np.array(types),np.array([cl]))
      nbc,nbl = 3,4
      grid = GridSpec(nbl,nbc*2)
      fig = plt.figure(figsize=(18,12))
      fig.set_facecolor('white')
      for iter in sorted(self.results):
        dic = self.results[iter][cl]

        if dic['nb'] == 0:
          continue

        num = iter%nbc + iter + iter/nbc * nbc
        row = iter/nbc
        col = iter%nbc * 2

        valok = dic['rate_%s'%cl][1]
        ax = fig.add_subplot(grid[row,col],aspect=1)
        classname = cl
        if classname == 'VulkanikB':
          classname = 'VB'
        if classname == 'VulkanikA':
          classname = 'VA'
        ax.pie([valok,100-valok],explode=(.05,0),labels=(classname,'R'),autopct='%1.1f%%',colors=('w',(.5,.5,.5)))
        ax.text(0.2,0,'# events = %d'%dic['nb'],transform=ax.transAxes)

        fracs = [tup[1]*1./(dic['nb']-dic['nb_common']) for tup in dic['nb_other'] if tup[1]!=0]
        labels = np.array([tup[0] for tup in dic['nb_other'] if tup[1]!=0])
        if 'VulkanikB' in labels:
          labels[labels=='VulkanikB'] = 'VB'
        if 'Tremor' in labels:
          labels[labels=='Tremor'] = 'Tr'
        if 'VulkanikA' in labels:
          labels[labels=='VulkanikA'] = 'VA'
        if 'Tektonik' in labels:
          labels[labels=='Tektonik'] = 'Tecto'
        if 'Hembusan' in labels:
          labels[labels=='Hembusan'] = 'Hem'
        if 'Longsoran' in labels:
          labels[labels=='Longsoran'] = 'Eb'
        if 'Hibrid' in labels:
          labels[labels=='Hibrid'] = 'Hy'
        plt.subplot(grid[row,col+1],aspect=1)
        plt.pie(fracs,labels=labels,autopct='%1.1f%%',colors=colors)
      plt.suptitle('Extraction of %s'%cl)

    plt.show()


  def plot_diagrams_one_draw(self):
    """
    Même chose que plot_all_diagrams, mais affiche sur la même figure 
    les résultats de toutes les classes pour un tirage donné.
    Note : nombre de classes limité aux 4 principales !!
    (Une figure par tirage)
    """
    from matplotlib.gridspec import GridSpec

    types = sorted(self.results[0])
    if 'i_train' in types:
      types.remove('i_train')
    if 'i_test' in types:
      types.remove('i_test')
    if len(types) > 4:
      types = ['Tektonik','Tremor','VulkanikB','VulkanikA']
    N = np.array(range(len(types)-1))

    colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral','lightgreen','khaki','plum','powderblue']
    width = 0.1

    for tir in sorted(self.results):
      fig = plt.figure(figsize=(12,7))
      fig.set_facecolor('white')
      nbl,nbc = 2,2
      grid = GridSpec(nbl,nbc*2)
      for icl, cl in enumerate(types):
        t = np.setxor1d(np.array(types),np.array([cl]))
        dic = self.results[tir][cl]

        if dic['nb'] == 0:
          continue

        num = icl%nbc + icl + icl/nbc * nbc
        row = icl/nbc
        col = icl%nbc * 2

        valok = dic['rate_%s'%cl][1]
        ax = fig.add_subplot(grid[row,col],aspect=1)
        classname = cl
        if classname == 'VulkanikB':
          classname = 'VB'
        if classname == 'VulkanikA':
          classname = 'VA'
        ax.pie([valok,100-valok],explode=(.05,0),labels=(classname,'R'),autopct='%1.1f%%',colors=('w',(.5,.5,.5)))
        ax.text(0.2,-.1,"# events = %d"%dic['nb'],transform=ax.transAxes)

        fracs = [tup[1]*1./(dic['nb']-dic['nb_common']) for tup in dic['nb_other'] if tup[1]!=0]
        labels = np.array([tup[0] for tup in dic['nb_other'] if tup[1]!=0])

        if 'VulkanikB' in labels:
          labels[labels=='VulkanikB'] = 'VB'
        if 'Tremor' in labels:
          labels[labels=='Tremor'] = 'Tr'
        if 'VulkanikA' in labels:
          labels[labels=='VulkanikA'] = 'VA'
        if 'Tektonik' in labels:
          labels[labels=='Tektonik'] = 'Tecto'
        if 'Hembusan' in labels:
          labels[labels=='Hembusan'] = 'Hem'
        if 'Longsoran' in labels:
          labels[labels=='Longsoran'] = 'Eb'
        if 'Hibrid' in labels:
          labels[labels=='Hibrid'] = 'Hy'
        plt.subplot(grid[row,col+1],aspect=1)
        plt.pie(fracs,labels=labels,autopct='%1.1f%%',colors=colors)
        if icl == 0:
          plt.figtext(.1,.87,'(a) Extraction of %s'%types[icl])
        elif icl == 1:
          plt.figtext(.55,.87,'(b) Extraction of %s'%types[icl])
        elif icl == 2:
          plt.figtext(.1,.43,'(c) Extraction of %s'%types[icl])
        elif icl == 3:
          plt.figtext(.55,.43,'(d) Extraction of %s'%types[icl])

    plt.show()


  def plot_pdf_extract(self):
    """
    Plot des pdf des features pour :
      - la classe extraite vs ce qui reste
      - les différentes classes manuelles qui composent la classe extraite
    """
    X_ini = self.x.copy()
    Y_ini = self.y.copy()
    for tir in sorted(self.results):
      print "TIRAGE",tir
      for cl in sorted(self.results[tir]):
        if cl == 'i_train' or cl == 'i_test':
          continue
        if self.results[tir][cl]['nb'] == 0:
          continue
        print '***',cl
        # Ensemble des événements composant la classe extraite
        index_ext = self.results[tir][cl]['index_ok']
        for i in range(len(self.results[tir][cl]['i_other'])):
          index_ext = np.concatenate((index_ext,self.results[tir][cl]['i_other'][i][1]))
        index_ext = np.array(map(int,list(index_ext)))

        # Classe extraite vs Reste
        self.x = X_ini.copy()
        self.y = Y_ini.copy()
        self.y.Type[index_ext] = 0
        self.y[self.y.Type!=0] = 1
        self.verify_index()
        self.types = [cl,'Rest']
        #self.plot_all_pdfs()

        # Classifications manuelles au sein de la classe extraite
        #del self.gaussians
        self.x = X_ini.reindex(index=index_ext)
        self.y = Y_ini.reindex(index=index_ext)
        self.verify_index()
        self.y[self.y.Type==cl] = 0
        k = 0
        acl = [cl]
        for i in range(len(self.results[tir][cl]['i_other'])):
          clo = self.results[tir][cl]['i_other'][i][0]
          if len(self.results[tir][cl]['i_other'][i][1]) > 1:
            self.y[self.y.Type==clo] = k+1
            acl.append(clo)
            k = k+1
        self.types = acl
        self.plot_all_pdfs()

    self.x = X_ini
    self.y = Y_ini


