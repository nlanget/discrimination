#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import pandas as pd
import os,sys,glob
from options import MultiOptions

class AnalyseResults(MultiOptions):

  def __init__(self,opt):
    MultiOptions.__init__(self,opt)

    self.opdict['class_auto_file'] = 'auto_class.csv'
    self.opdict['class_auto_path'] = '%s/%s/%s'%(self.opdict['outdir'],self.opdict['method'].upper(),self.opdict['class_auto_file'])

    self.concatenate_results()
    self.display_results()


  def read_result_file(self):
    """
    Reads the file containing the results
    """
    dic = self.read_binary_file(self.opdict['result_path'])
    self.opdict['feat_list'] = dic['features']
    del dic['features']
    self.results = dic
    self.opdict['stations'] = [key[0] for key in sorted(dic)]
    self.opdict['channels'] = [key[1] for key in sorted(dic)]


  def concatenate_results(self):
    """
    Does a synthesis of all classifications
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
    Plots the confusion matrix (test set).
    """
    from do_classification import confusion
    import matplotlib.pyplot as plt

    self.tri()
    self.classname2number()

    m = self.man
    a = self.auto
    for i in self.numt:
      m['Type'][m.Type==self.types[i]] = i
      a['Type'][a.Type==self.types[i]] = i
    a = a[a.Type!='?']
    m = m.reindex(index=a.index)

    confusion(m,a.values[:,0],self.types,'test',self.opdict['method'],plot=True)
    if self.opdict['save_confusion']:
      plt.savefig('%s/figures/test_%s.png'%(self.opdict['outdir'],self.opdict['result_file'][8:]))
    plt.show()



class AnalyseResultsExtraction(MultiOptions):

  def __init__(self,opt):
    MultiOptions.__init__(self,opt)
    self.results = self.read_binary_file(self.opdict['result_path'])
    self.opdict['feat_list'] = self.results['features']
    del self.results['features']

    self.do_analysis()


  def do_analysis(self):
    for sta in self.opdict['stations']:
      for comp in self.opdict['channels']:
        self.x, self.y = self.features_onesta(sta,comp)
        self.verify_index()

        #self.unclass_histo()
        #self.unclass_diagram()
        #self.plot_all_diagrams()
        self.plot_diagrams_one_draw()
        if self.opdict['method'] == 'ova':
          self.search_repetition()


  def unclass_histo(self):
    """
    Classe d'appartenance d'origine (classification manuelle) des événements 
    non classés à l'issue de l'extraction.
    """
    import matplotlib.pyplot as plt

    tuniq = np.unique(self.y.Type)
    N = np.array(range(len(np.unique(tuniq))))
    width = 0.1
    colors = [(0,0,1),(1,0,0),(0,1,0),(1,1,0),(0,0,0),(0,1,1),(1,0,1),(1,1,1),(.8,.5,0),(0,0.5,0.5)]

    fig, ax = plt.subplots()
    fig.set_facecolor('white')
    for iter in sorted(self.results):
      rest = np.array([])
      for cl in sorted(self.results[iter]):
        if cl == 'i_train':
          continue
        for tup in self.results[iter][cl]['i_other']:
          rest = np.hstack((rest,tup[1]))
        rest = np.append(rest,self.results[iter][cl]['index_ok'])

      rest_uniq = np.unique(rest)
      rest_uniq = np.array(map(int,list(rest_uniq)))
      unclass = np.setdiff1d(np.array(self.x.index),rest_uniq)
    
      unclass_types = self.y.reindex(index=unclass,columns=['Type']).values
      nb = [len(unclass_types[unclass_types==t]) for t in tuniq]
      rects = ax.bar(N+iter*width,nb,width,color=colors)

    ax2 = ax.twinx()
    ylab = []
    for yt in ax.get_yticks()*100./len(self.x):
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
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    types = np.unique(self.y.Type.values)
    N = np.array(range(len(np.unique(types))))
    width = 0.1
    colors = [(0,0,1),(1,0,0),(0,1,0),(1,1,0),(0,0,0),(0,1,1),(1,0,1),(1,1,1),(.8,.5,0),(0,0.5,0.5)]

    for iter in sorted(self.results):
      fig = plt.figure(figsize=(12,8))
      fig.set_facecolor('white')
      grid = GridSpec(3,4)
      rest = np.array([])
      for cl in sorted(self.results[iter]):
        if cl == 'i_train':
          continue
        for tup in self.results[iter][cl]['i_other']:
          rest = np.hstack((rest,tup[1]))
        rest = np.append(rest,self.results[iter][cl]['index_ok'])

      rest_uniq = np.unique(rest)
      rest_uniq = np.array(map(int,list(rest_uniq)))
      unclass = np.setdiff1d(np.array(self.x.index),rest_uniq)
      
      unclass_types = self.y.reindex(index=unclass,columns=['Type']).values
      nb = [len(unclass_types[unclass_types==t]) for t in types]

      plt.subplot(grid[:2,:2],aspect=1,title='Unclassified events')
      plt.pie(nb,labels=types,autopct='%1.1f%%')

      orga = [(0,2),(0,3),(1,2),(1,3),(2,0),(2,1),(2,2),(2,3)]
      for it,t in enumerate(types):
        nb = len(unclass_types[unclass_types==t])
        nb_tot = len(self.y[self.y.Type==t])
        il = orga[it][0]
        icol = orga[it][1]
        plt.subplot(grid[il,icol],aspect=1,title=t)
        plt.pie([nb,nb_tot-nb],explode=(.05,0),labels=['unclass',''],autopct='%1.1f%%',colors=((.5,.5,.5),'w'))
      plt.figtext(.1,.91,'(a)',fontsize=18)
      plt.figtext(.5,.91,'(b)',fontsize=18)
    #plt.savefig('../results/Ijen/figures/unclass_OBO.png')
    plt.show()



  def search_repetition(self):
    """
    Evénements qui sont classés plusieurs fois, et ce, dans des classes différentes.
    One-vs-All method only.
    """
    import matplotlib.pyplot as plt

    types = self.y.Type

    N = np.array(range(len(np.unique(self.y.Type))))
    width = 0.1
    colors = [(0,0,1),(1,0,0),(0,1,0),(1,1,0),(0,0,0),(0,1,1),(1,0,1),(1,1,1),(.8,.5,0),(0,.5,.5)]
    max = 0

    fig,ax = plt.subplots()
    fig.set_facecolor('white')
    for iter in sorted(self.results):
      rest = np.array([])
      for cl in sorted(self.results[iter]):
        if cl == 'i_train':
          continue
        for tup in self.results[iter][cl]['i_other']:
          rest = np.hstack((rest,tup[1]))
        rest = np.append(rest,self.results[iter][cl]['index_ok'])

      rest = np.array(map(int,list(rest)))
      rest_uniq,i_rest_uniq = np.unique(rest,return_index=True)
      i_not_uniq = np.setxor1d(np.array(range(len(rest))),i_rest_uniq)
      print iter,len(rest),len(rest_uniq)
      rest_not_uniq = rest[i_not_uniq]

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
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    types = sorted(self.results[0])
    if 'i_train' in types:
      types.remove('i_train')
    print types
    N = np.array(range(len(types)-1))

    colors_all = create_color_scale(types)
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
        plt.subplot(grid[row,col],aspect=1)
        classname = cl
        if classname == 'VulkanikB':
          classname = 'VB'
        if classname == 'VulkanikA':
          classname = 'VA'
        plt.pie([valok,100-valok],explode=(.05,0),labels=(classname,'R'),autopct='%1.1f%%',colors=('w',(.5,.5,.5)))

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
        plt.pie(fracs,labels=labels,autopct='%1.1f%%')
      plt.suptitle('Extraction of %s'%cl)

    plt.show()


  def plot_diagrams_one_draw(self):
    """
    Même chose que plot_all_diagrams, mais affiche sur la même figure 
    les résultats de toutes les classes pour un tirage donné.
    Note : nombre de classes limité aux 4 principales !!
    (Une figure par tirage)
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    types = sorted(self.results[0])
    if 'i_train' in types:
      types.remove('i_train')
    if len(types) > 4:
      types = ['Tektonik','Tremor','VulkanikB','VulkanikA']
    N = np.array(range(len(types)-1))

    colors_all = create_color_scale(types)
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
        plt.subplot(grid[row,col],aspect=1)
        classname = cl
        if classname == 'VulkanikB':
          classname = 'VB'
        if classname == 'VulkanikA':
          classname = 'VA'
        plt.pie([valok,100-valok],explode=(.05,0),labels=(classname,'R'),autopct='%1.1f%%',colors=('w',(.5,.5,.5)))

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
        plt.pie(fracs,labels=labels,autopct='%1.1f%%')
      plt.figtext(.1,.87,'(a) Extraction of tectonics')
      plt.figtext(.55,.87,'(b) Extraction of tremors')
      plt.figtext(.1,.43,'(c) Extraction of VB')
      plt.figtext(.55,.43,'(d) Extraction of VA')
      #plt.savefig('../results/Ijen/figures/OBO_SVM_tir%d.png'%tir)

    plt.show()






def create_color_scale(clist):
  all_colors = [(0,0,1),(1,0,0),(0,1,0),(1,1,0),(0,0,0),(0,1,1),(1,0,1),(1,1,1),(.8,.5,0),(0,0.5,0.5)]
  colors = []
  for ic,c in enumerate(clist):
    colors.append((c,all_colors[ic]))
  return colors


def associate_color(tuplist,names):
  colors = []
  for tup in tuplist:
    name = tup[0]
    if name in names:
      colors.append(tup[1])
  return colors
