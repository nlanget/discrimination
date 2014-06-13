#!/usr/bin/env python
# encoding: utf-8

import os,glob,sys
from obspy.core import read,utcdatetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from LR_functions import comparison
from do_classification import confusion, create_training_set

def one_by_one(x_test_ref0,y_test_ref0,types,numt,otimes_ref,file,boot=1,method='lr'):

  """
  Extract one class after each other by order of importance. The events which are 
  classified are deleted from the next extraction.
  boot = number of training sets to be generated
  method = 'lr' for Logistic Regression / 'svm' for SVM
  """

  import cPickle
  from LR_functions import do_all_logistic_regression

  len_numt = len(numt)
  DIC = {}
  DIC['features'] = x_test_ref0.columns 

  for b in range(boot):

    otimes = map(str,list(otimes_ref.values))
    otimes = np.array(otimes)

    x_test = x_test_ref0.copy()
    y_test_ref = y_test_ref0.copy()

    print "\n\tONE BY ONE EXTRACTION ------ iteration %d"%b
    dic = {}

    for n in range(len_numt):

      sub_dic={}
      y_train_ref = create_training_set(y_test_ref,numt)
      x_train = x_test.reindex(index=map(int,y_train_ref.index))
      sub_dic["i_train"] = otimes[map(int,list(y_train_ref.index))]
      y_train_ref.index = range(y_train_ref.shape[0])
      x_train.index = range(x_train.shape[0])

      if x_train.shape[0] != y_train_ref.shape[0]:
        print "Training set: Incoherence in x and y dimensions"
        sys.exit()

      if x_test.shape[0] != y_test_ref.shape[0]:
        print "Test set: Incoherence in x and y dimensions"
        sys.exit()

      if len(otimes) != len(y_test_ref):
        print "Warning !! Check lengths !"
        sys.exit()

      y_train = y_train_ref.copy()
      y_test = y_test_ref.copy()
      y_train[y_train_ref.Type==n] = 0
      y_test[y_test_ref.Type==n] = 0
      y_train[y_train_ref.Type!=n] = 1
      y_test[y_test_ref.Type!=n] = 1

      t = [types[n],'Rest']
      print y_train.shape[0], y_test.shape[0]

      print "----------- %s vs all -----------"%types[n]

      if method == 'lr':
        print "Logistic Regression\n"
        LR_train,theta,LR_test,wtr = do_all_logistic_regression(x_train,y_train,x_test,y_test,output=True)
      elif method == 'svm':
        kern = 'nonlinear'
        print "SVM\n"
        from sklearn.grid_search import GridSearchCV
        from sklearn import svm
        C_range = 10.0 ** np.arange(-2, 5)
        if kern == 'linear':
          param_grid = dict(C=C_range)
          grid = GridSearchCV(svm.LinearSVC(), param_grid=param_grid, n_jobs=-1)
        else:
          gamma_range = 10.0 ** np.arange(-3,3)
          param_grid = dict(gamma=gamma_range, C=C_range)
          grid = GridSearchCV(svm.SVC(kernel='rbf'), param_grid=param_grid, n_jobs=-1)
        grid.fit(x_train.values, y_train.values.ravel())
        LR_train = grid.best_estimator_.predict(x_train)
        LR_test = grid.best_estimator_.predict(x_test)

      print "\t Training set"
      for i in range(2):
        print i, t[i], len(np.where(y_train.values[:,0]==i)[0]), len(np.where(LR_train==i)[0])
      print "\n"
      cmat_train = confusion(y_train,LR_train,types,'training','LogReg',plot=False,output=True)

      print "\t Test set"
      for i in range(2):
        print i, t[i], len(np.where(y_test.values[:,0]==i)[0]), len(np.where(LR_test==i)[0])
      print "\n"
      cmat_test = confusion(y_test,LR_test,types,'test','LogReg',plot=False,output=True)

      # Fill the dictionary
      i_com = np.where((y_test.values.ravel()-LR_test)==0)[0]
      i_lr = np.where(LR_test==0)[0]
      i_ok_class = np.intersect1d(i_com,i_lr) # events classified in the class of interest by the LR and identical to the manual classification

      sub_dic["nb"] = len(i_lr) # total number of events classified in the class of interest
      sub_dic["nb_common"] = len(i_ok_class)
      sub_dic["index_ok"] = otimes[i_ok_class]
      sub_dic["nb_other"],sub_dic["i_other"] = [],[]
      for k in range(len_numt):
        if k != n:
          i_other_man = list(y_test_ref[y_test_ref.Type==k].index)
          ii = np.intersect1d(i_lr,i_other_man)
          sub_dic["nb_other"].append((types[k],len(ii)))
          sub_dic["i_other"].append((types[k],otimes[ii]))
      sub_dic["rate_%s"%types[n]] = (cmat_train[0,0],cmat_test[0,0])
      sub_dic["rate_rest"] = (cmat_train[1,1],cmat_test[1,1])
      sub_dic["nb_manuals"] = ((types[n],len(y_test[y_test.Type==0])),('Rest',len(y_test[y_test.Type==1])))

      i_ok = np.where(LR_test!=0)[0]
      y_test_ref = y_test_ref.reindex(index=i_ok)
      x_test = x_test.reindex(index=i_ok)
      otimes = otimes[i_ok]

      y_test_ref.index = range(y_test_ref.shape[0])
      x_test.index = range(x_test.shape[0])

      dic[types[n]] = sub_dic

    DIC[b] = dic

  print "One-by-One results stored in %s"%file
  with open(file,'wb') as test:
    my_pickler=cPickle.Pickler(test)
    my_pickler.dump(DIC)
    test.close()

# ================================================================

def one_vs_all(x_test,y_test_ref,types,numt,otimes_ref,file,boot=1,method='lr'):

  """
  Extract one class among the whole data.
  """

  import cPickle
  from LR_functions import do_all_logistic_regression

  len_numt = len(numt)

  DIC = {}
  DIC['features'] = x_test.columns
  for b in range(boot):

    print "\n\tONE VS ALL EXTRACTION ------ iteration %d"%b

    dic = {}
    otimes = map(str,list(otimes_ref.values))
    otimes = np.array(otimes)

    y_train_ref = create_training_set(y_test_ref,numt)
    x_train = x_test.reindex(index=y_train_ref.index)
    dic["i_train"] = otimes[map(int,list(y_train_ref.index))]
    y_train_ref.index = range(y_train_ref.shape[0])
    x_train.index = range(x_train.shape[0])

    for n in range(len_numt):

      y_train = y_train_ref.copy()
      y_test = y_test_ref.copy()
      y_train[y_train_ref.Type==n] = 0
      y_test[y_test_ref.Type==n] = 0
      y_train[y_train_ref.Type!=n] = 1
      y_test[y_test_ref.Type!=n] = 1

      print y_train.shape[0], y_test.shape[0]

      print "----------- %s vs all -----------"%types[n]
      print_type = [types[n],'All']

      if method == 'lr':
        print "Logistic Regression\n"
        LR_train,theta,LR_test,wtr = do_all_logistic_regression(x_train,y_train,x_test,y_test,output=True)
      elif method == 'svm':
        kern = 'nonlinear'
        print "SVM\n"
        from sklearn.grid_search import GridSearchCV
        from sklearn import svm
        C_range = 10.0 ** np.arange(-2, 5)
        if kern == 'linear':
          param_grid = dict(C=C_range)
          grid = GridSearchCV(svm.LinearSVC(), param_grid=param_grid, n_jobs=-1)
        else:
          gamma_range = 10.0 ** np.arange(-3,3)
          param_grid = dict(gamma=gamma_range, C=C_range)
          grid = GridSearchCV(svm.SVC(kernel='rbf'), param_grid=param_grid, n_jobs=-1)
        grid.fit(x_train.values, y_train.values.ravel())
        LR_train = grid.best_estimator_.predict(x_train)
        LR_test = grid.best_estimator_.predict(x_test)

      print "\t Training set"
      for i in range(2):
        print i, print_type[i], len(np.where(y_train.values[:,0]==i)[0]), len(np.where(LR_train==i)[0])
      print "\n"
      cmat_train = confusion(y_train,LR_train,types,'training','LogReg',plot=True,output=True)
      plt.close()

      print "\t Test set"
      for i in range(2):
        print i, print_type[i], len(np.where(y_test.values[:,0]==i)[0]), len(np.where(LR_test==i)[0])
      print "\n"
      cmat_test = confusion(y_test,LR_test,types,'test','LogReg',plot=True,output=True)
      plt.close()

      # Fill the dictionary
      sub_dic={}
      i_com = np.where((y_test.values.ravel()-LR_test)==0)[0]
      i_lr = np.where(LR_test==0)[0]
      i_ok_class = np.intersect1d(i_com,i_lr) # events classified in the class of interest by the LR and identical to the manual classification
      sub_dic["nb"] = len(i_lr) # total number of events classified in the class of interest
      sub_dic["nb_common"] = len(i_ok_class) # total number of well classified events
      sub_dic["index_ok"] = otimes[i_ok_class] # index of well classified events
      sub_dic["nb_other"],sub_dic["i_other"] = [],[]
      for k in range(len_numt):
        if k != n:
          i_other_man = list(y_test_ref[y_test_ref.Type==k].index)
          ii = np.intersect1d(i_lr,i_other_man)
          sub_dic["nb_other"].append((types[k],len(ii))) # number of events belonging to another class
          sub_dic["i_other"].append((types[k],otimes[ii])) # index of events belonging to another class
      sub_dic["rate_%s"%types[n]] = (cmat_train[0,0],cmat_test[0,0]) # % success rate of the extracted class
      sub_dic["rate_rest"] = (cmat_train[1,1],cmat_test[1,1]) # % success rate of the rest
      sub_dic["nb_manuals"] = ((types[n],len(y_test[y_test.Type==0])),('Rest',len(y_test[y_test.Type==1])))
      dic[types[n]] = sub_dic

    DIC[b] = dic

  print "One-vs-All results stored in %s"%file
  with open(file,'wb') as test:
    my_pickler=cPickle.Pickler(test)
    my_pickler.dump(DIC)
    test.close()

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
  colors = ['b','k','b','c','m','y','g','r']
  markers = ['*','o','h','v','d','s','s','d']
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

def plot_pdf_extract(DIC):
  """
  Plot des pdf des features pour :
    - la classe extraite vs ce qui reste
    - les différentes classes manuelles qui composent la classe extraite
  """
  from Ijen_extract_features import plot_pdf_feat
  df_ref = pd.read_csv('../results/Ijen/ijen_0605_1sta.csv')
  dates = df_ref[df_ref.columns[0]].values
  df_ref.index = map(str,list(dates))
  types = df_ref.reindex(columns=['EventType'])
  df_ref = df_ref.reindex(columns=DIC['features'])

  for tir in sorted(DIC):
    if tir != 'features':
      print "TIRAGE",tir
      for cl in sorted(DIC[tir]):
        if cl != 'i_train':
          print '***',cl
          # Ensemble des événements composant la classe extraite
          index_ext = DIC[tir][cl]['index_ok']
          for i in range(len(DIC[tir][cl]['i_other'])):
            index_ext = np.concatenate((index_ext,DIC[tir][cl]['i_other'][i][1]))

          # Classe extraite vs Reste
          x_all = df_ref.copy()
          y_all = types.copy()
          y_all.EventType[index_ext] = 0
          y_all[y_all.EventType!=0] = 1
          x_all.index = range(len(x_all))
          y_all.index = range(len(y_all))
          plot_pdf_feat(x_all,y_all,[cl,'Rest'],save=False,output=False)

          # Classifications manuelles au sein de la classe extraite
          x_ext = df_ref.reindex(index=index_ext)
          y_ext = types.reindex(index=index_ext)
          x_ext.index = range(len(x_ext))
          y_ext.index = range(len(y_ext))
          y_ext[y_ext.EventType==cl] = 0
          k = 0
          acl = [cl]
          for i in range(len(DIC[tir][cl]['i_other'])):
            k = k+1
            clo = DIC[tir][cl]['i_other'][i][0]
            y_ext[y_ext.EventType==clo] = k
            acl.append(clo)
          plot_pdf_feat(x_ext,y_ext,acl,save=False,output=False)

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

def plot_diagrams(DIC):
  """
  Le titre de la figure correspond à la classe extraite automatiquement 
  avec le nombre d'événements.
  Chaque couple de diagrammes circulaires correspond à un tirage donné :
    * le 1er donne le taux de bonne extraction de la classe considérée 
  (par rapport au nombre total extrait)
    * le 2ème montre la provenance (ie classes manuelles) des événements 
  classés automatiquement dans la classe extraite mais qui n'ont pas la 
  bonne classe d'origine (= reste).
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
      t[t=='TektonikLokal'] = 'Tekto'
    from matplotlib.gridspec import GridSpec
    nbc,nbl = 3,4
    grid = GridSpec(nbl,nbc*2)
    fig = plt.figure(figsize=(18,12))
    fig.set_facecolor('white')
    for iter in sorted(DIC):
      if iter != 'features':
        dic = DIC[iter][cl]

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
        if classname == 'Tremor':
          classname = 'T'
        plt.pie([valok,100-valok],explode=(.05,0),labels=(classname,'R'),autopct='%1.1f%%',colors=('w',(.5,.5,.5)))

        fracs = [tup[1]*1./(dic['nb']-dic['nb_common']) for tup in dic['nb_other'] if tup[1]!=0]
        labels = np.array([tup[0] for tup in dic['nb_other'] if tup[1]!=0])
        if 'VulkanikB' in labels:
          labels[labels=='VulkanikB'] = 'VB'
        if 'Tremor' in labels:
          labels[labels=='Tremor'] = 'T'
        plt.subplot(grid[row,col+1],aspect=1)
        plt.pie(fracs,labels=labels,autopct='%1.1f%%')
    plt.suptitle('Extraction of %s'%cl)

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

def search_repetition(DIC):

  df = pd.read_csv('../results/Ijen/ijen_0605_1sta.csv')
  all = np.array(map(str,list(df[df.columns[0]])))
  df.index = list(all)

  df.EventType[df.EventType=='VulkanikA'] = 'VulkanikB'
  types = df.EventType
  types = types.dropna()
  df = df.reindex(index=types.index)

  N = np.array(range(len(np.unique(df.EventType))))
  width = 0.1
  colors = [(0,0,1),(1,0,0),(0,1,0),(1,1,0),(0,0,0),(0,1,1),(1,0,1),(1,1,1),(.8,.5,0),(0,.5,.5)]
  max = 0

  fig, ax = plt.subplots()
  fig.set_facecolor('white')
  for iter in sorted(DIC):
    if iter != 'features':
      rest = np.array([])
      for cl in sorted(DIC[iter]):
        if cl != 'i_train':
          #N = np.array(range(len(DIC[iter][cl])+1))
          for tup in DIC[iter][cl]['i_other']:
            rest = np.hstack((rest,tup[1]))
          rest = np.append(rest,DIC[iter][cl]['index_ok'])

      rest_uniq,i_rest_uniq = np.unique(rest,return_index=True)
      i_not_uniq = np.setxor1d(np.array(range(len(rest))),i_rest_uniq)
      rest_not_uniq = rest[i_not_uniq]

      notu_types = df.reindex(index=rest_not_uniq,columns=['EventType']).values
      nb = [len(notu_types[notu_types==t]) for t in np.unique(df.EventType)]
      rects = ax.bar(N+iter*width,nb,width,color=colors)
      if np.max(nb) > max:
        max = np.max(nb)

  types = np.unique(df.EventType.values)
  if 'HarmonikTremor' in types:
    types[types=='HarmonikTremor'] = 'Tremor'
  if 'LowFrequency' in types:
    types[types=='LowFrequency'] = 'LF'
  if 'TektonikLokal' in types:
    types[types=='TektonikLokal'] = 'Tekto'
  if 'VulkanikA' in types:
    types[types=='VulkanikA'] = 'VA'
  if 'VulkanikB' in types:
    types[types=='VulkanikB'] = 'VB'
  ax.set_xticks(N+(len(DIC)-1)/2*width)
  ax.set_xticklabels(types)
  ax.set_xlabel('Manual class of repeated events')
  ax.set_ylabel('Number of repeated events')
  ax.set_title("Repeated events")
  
  plt.show()

# ================================================================

def stats_unclass(DIC):
  """
  Classe d'appartenance d'origine (classification manuelle) des événements 
  non classés à l'issue de l'extraction.
  """

  df = pd.read_csv('../results/Ijen/ijen_0605_1sta.csv')
  all = np.array(map(str,list(df[df.columns[0]])))
  df.index = list(all)

  #df.EventType[df.EventType=='VulkanikA'] = 'VulkanikB'

  tuniq = np.unique(df.EventType)

  N = np.array(range(len(np.unique(tuniq))))
  width = 0.1
  colors = [(0,0,1),(1,0,0),(0,1,0),(1,1,0),(0,0,0),(0,1,1),(1,0,1),(1,1,1),(.8,.5,0),(0,0.5,0.5)]

  fig, ax = plt.subplots()
  fig.set_facecolor('white')
  for iter in sorted(DIC):
    if iter != 'features':
      rest = np.array([])
      for cl in sorted(DIC[iter]):
        if cl != 'i_train':
          for tup in DIC[iter][cl]['i_other']:
            rest = np.hstack((rest,tup[1]))
          rest = np.append(rest,DIC[iter][cl]['index_ok'])

      rest_uniq = np.unique(rest)
      unclass = np.setdiff1d(all,rest_uniq)
      
      unclass_types = df.reindex(index=unclass,columns=['EventType']).values
      nb = [len(unclass_types[unclass_types==t]) for t in tuniq]
      rects = ax.bar(N+iter*width,nb,width,color=colors)

  types = tuniq.values
  if 'HarmonikTremor' in types:
    types[types=='HarmonikTremor'] = 'Tremor'
  if 'LowFrequency' in types:
    types[types=='LowFrequency'] = 'LF'
  if 'TektonikLokal' in types:
    types[types=='TektonikLokal'] = 'Tekto'
  if 'VulkanikA' in types:
    types[types=='VulkanikA'] = 'VA'
  if 'VulkanikB' in types:
    types[types=='VulkanikB'] = 'VB'
  ax.set_xticks(N+(len(DIC)-1)/2*width)
  ax.set_xticklabels(types)
  ax.set_xlabel('Manual class of unclassified events')
  ax.set_ylabel('Number of unclassified events')
  ax.set_title("Unclassified events")

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
            opt.plot_one_pdf(feat,(mant,a[feat].values,final_cl))
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

def read_extraction_results(filename):

  from obspy.core import utcdatetime,read

  DIC = read_binary_file(filename)
  #search_repetition(DIC) # statistics on multiclassified events (One-vs-All only)
  #stats_unclass(DIC) # statistics on unclassified events
  #class_histograms(DIC) # plot extraction results as histograms
  plot_diagrams(DIC) # plot extraction results as diagrams
  #plot_rates(DIC) # plot extraction results as training set vs test set
  #plot_training(DIC)
  #plot_pdf_extract(DIC)
  #event_classes(DIC)
  #search_and_reclass(DIC,'Tremor')
  #plot_features_vs(DIC)

# ================================================================
if __name__ == '__main__' :
  read_extraction_results()
