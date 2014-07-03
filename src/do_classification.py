#!/usr/bin/env python
# encoding: utf-8

import os,glob,sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from LR_functions import comparison


def classifier(opt):

  """
  Classification of the different types of events.
  opt is an object of the class Options()
  """


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
    #if len(opt.xs[isc]) < 100:
    #  continue

    opt.x = opt.xs[isc]
    opt.y = opt.ys[isc]

    set = pd.DataFrame(index=opt.ys[isc].index,columns=['Otime'])
    set['Otime'] = opt.xs[isc].index

    opt.classname2number()
    x_test = opt.x
    y_ref = opt.y

    K = len(opt.types)

    for b in range(opt.opdict['boot']):
      print "\n-------------------- # iter: %d --------------------\n"%(b+1)

      y_test = y_ref.copy()

      if marker_sta == 0:
        y_train = create_training_set(y_ref,opt.numt)
        list_ev_train = y_train.index
      else:
        y_train = y_ref.reindex(index=list_ev_train)
        y_train = y_train.dropna(how='any')

      y = y_train.copy()

      in_train = np.intersect1d(np.array(y_test.index),np.array(y_train.index))
      set[b] = np.zeros(set.shape[0])
      set[b][in_train] = 1

      y_train = y.copy()
      x_train = opt.x.reindex(index=y_train.index)
      x_test = opt.x.reindex(index=y_test.index)

      print "# types in the test set:",len(np.unique(y_test.values.ravel()))
      print "# types in the training set:",len(np.unique(y_train.values.ravel()))

      x_train.index = range(x_train.shape[0])
      y_train.index = range(y_train.shape[0])
      print x_train.shape, y_train.shape
      if x_train.shape[0] != y_train.shape[0]:
        print "Training set: Incoherence in x and y dimensions"
        sys.exit()

      subdic['list_ev'] = np.array(y_test.index)

      x_test.index = range(x_test.shape[0])
      y_test.index = range(y_test.shape[0])
      print x_test.shape, y_test.shape
      if x_test.shape[0] != y_test.shape[0]:
        print "Test set: Incoherence in x and y dimensions"
        sys.exit()

      if opt.opdict['plot_pdf']:
        opt.plot_all_pdfs(save=False)

      if opt.opdict['method'] == '1b1':
        # EXTRACTEURS
        print "********** EXTRACTION 1-BY-1 **********"
        from extraction import one_by_one
        savefile = '%s/1B1_%s_%s'%(opt.opdict['outdir'],opt.opdict['feat_filename'].split('.')[0],opt.trad[isc][0])
        one_by_one(x_test,y_test,opt.types,opt.numt,set['Otime'],savefile,boot=10,method='svm')

      elif opt.opdict['method'] == 'ova':
        print "********** EXTRACTION 1-VS-ALL **********"
        from extraction import one_vs_all
        savefile = '%s/OVA_%s_%s'%(opt.opdict['outdir'],opt.opdict['feat_filename'].split('.')[0],opt.trad[isc][0])
        one_vs_all(x_test,y_test,opt.types,opt.numt,set['Otime'],savefile,boot=10,method='svm')

      elif opt.opdict['method'] == 'svm':
        # SVM
        print "********** SVM **********"
        CLASS_test, pourcentages = implement_svm(x_train,x_test,y_train,y_test,opt.types,opt.opdict['boot'])

      elif opt.opdict['method'] == 'lr': 
        # LOGISTIC REGRESSION
        print "********* Logistic regression **********"
        from LR_functions import do_all_logistic_regression
        CLASS_train,theta,CLASS_test,pourcentages,wtr = do_all_logistic_regression(x_train,y_train,x_test,y_test,output=True,perc=True)
        print "\t Training set"
        for i in range(K):
          print i, opt.types[i], len(np.where(y_train.values[:,0]==i)[0]), len(np.where(CLASS_train==i)[0])
        print "\n"
        if opt.opdict['boot'] == 1:
          confusion(y_train,CLASS_train,opt.st,'training','LogReg',plot=opt.opdict['plot_confusion'])

        print "\t Test set"
        for i in range(K):
          print i, opt.types[i], len(np.where(y_test.values[:,0]==i)[0]), len(np.where(CLASS_test==i)[0])
        print "\n"
        if opt.opdict['boot'] == 1:
          confusion(y_test,CLASS_test,opt.st,'test','LogReg',plot=opt.opdict['plot_confusion'])
          if opt.opdict['plot_confusion']:
            plt.show()

      subdic['%'] = pourcentages
      trad_CLASS_test = []
      for i in CLASS_test:
        i = int(i)
        trad_CLASS_test.append(opt.types[i])
      subdic['classification'] = trad_CLASS_test

    dic_results[opt.trad[isc]] = subdic

  import cPickle
  with open(opt.opdict['result_path'],'w') as file:
    my_pickler = cPickle.Pickler(file)
    my_pickler.dump(dic_results)
    file.close()

# ================================================================

def create_training_set(y_ref,numt):

  """
  Generates a training set randomly from the test set.
  The proportions of each class constituting the test set are kept.
  """

  y = y_ref.copy()
  y_train = pd.DataFrame(columns=y.columns)
  list_index = np.array([])
  for i in numt:
    a = y[y.Type==i]
    nb = int(np.floor(0.4*len(a)))
    if nb < 3:
      nb = 3
    y_train = y_train.append(a[:nb])
    r = np.floor(len(a)*np.random.rand(nb))
    while len(np.unique(r)) != nb:
      r_uniq = np.unique(r)
      len_runiq = len(r_uniq)
      r = np.hstack((r_uniq,np.floor(len(a)*np.random.rand(nb-len_runiq))))
    list_index = np.hstack((list_index,list(a.index[list(r)])))
  y_train.index = list_index

  return y_train

# ================================================================
def confusion(y,LR_train,l,set,method,plot=False,output=False):
  """
  Computes the confusion matrix
  """
  from sklearn.metrics import confusion_matrix
  cmat = confusion_matrix(y.values[:,0],LR_train)
  cmat = np.array(cmat,dtype=float)
  for i in range(cmat.shape[0]):
    cmat[i,:] = cmat[i,:]*100./len(np.where(y.values[:,0]==i)[0])

  if plot:
    plt.matshow(cmat,cmap=plt.cm.gray_r)
    for i in range(cmat.shape[0]):
      for j in range(cmat.shape[0]):
        if cmat[j,i] >= np.max(cmat)/2.:
          col = 'w'
        else:
          col = 'k'
        if cmat.shape[0] <= 4:
          plt.text(i,j,"%.2f"%cmat[j,i],color=col)
        else:
          plt.text(i,j,"%d"%cmat[j,i],color=col)
    plt.title('Confusion matrix - %s - %s'%(set,method))
    plt.xlabel('Prediction')

    plt.ylabel('Observation')
    #plt.xticks(range(len(l)),l)
    #plt.yticks(range(len(l)),l)
    plt.colorbar()
    #plt.show()
  if output:
    return cmat
# ================================================================
def plot_clustering(x,x_data,y_train,y_clus,K):

  for i in range(K):
    print "# class %d = %d"%(i,len(np.where(y_train==i)[0]))

  if x.shape[1] >= 2:
    fig = plt.figure()
    fig.set_facecolor('white')
    plt.scatter(x.values[:,0],x.values[:,1],c=y_train,cmap=plt.cm.gray)
    plt.title('Training set')
    plt.xlabel(x.columns[0])
    plt.ylabel(x.columns[1])

    fig = plt.figure()
    fig.set_facecolor('white')
    plt.scatter(x_data.values[:,0],x_data.values[:,1],c=y_clus,cmap=plt.cm.gray)
    plt.title('Test set')
    plt.xlabel(x_data.columns[0])
    plt.ylabel(x_data.columns[1])
# ================================================================
def implement_svm(x_train,x_test,y_train,y_test,types,b,plot=False):
  # do grid search
  from sklearn.grid_search import GridSearchCV
  from sklearn import svm
  print "doing grid search"
  C_range = 10.0 ** np.arange(-2, 5)
  gamma_range = 10.0 ** np.arange(-3,3)
  param_grid = dict(gamma=gamma_range, C=C_range)
  grid = GridSearchCV(svm.SVC(), param_grid=param_grid, n_jobs=-1)
  grid.fit(x_train.values, y_train.values.ravel())
  print "The best classifier is: ", grid.best_estimator_
  print "Number of support vectors for each class: ", grid.best_estimator_.n_support_
  y_train_SVM = grid.best_estimator_.predict(x_train)
  y_test_SVM = grid.best_estimator_.predict(x_test)

  print "\t *Training set"
  diff = y_train.values.ravel() - y_train_SVM
  p_tr = float(len(np.where(diff==0)[0]))/y_train.shape[0]*100
  print "Correct classification: %.2f%%"%p_tr
  for i in range(len(np.unique(y_train.values))):
    print i, types[i], len(np.where(y_train.values[:,0]==i)[0]), len(np.where(y_train_SVM==i)[0])
  if b == 1:
    confusion(y_train,y_train_SVM,types,'training','SVM',plot=plot)

  print "\t *Test set"
  diff = y_test.values.ravel() - y_test_SVM
  p_test = float(len(np.where(diff==0)[0]))/y_test.shape[0]*100
  print "Correct classification: %.2f%%"%p_test
  for i in range(len(np.unique(y_train.values))):
    print i, types[i], len(np.where(y_test.values[:,0]==i)[0]), len(np.where(y_test_SVM==i)[0])
  if b == 1:
    confusion(y_test,y_test_SVM,types,'test','SVM',plot=plot)
    if plot:
      plt.show()
  return y_test_SVM,(p_tr,p_test)
# ================================================================
def class_final(opt):
  """
  Store the automatic classification into a .csv file.
  The index of the DataFrame structure contains the list of events.
  The columns of the DataFrame structure contain, for each event : 
    - Type : automatic class
    - Nb : number of stations implied in the classification process
    - NbDiff : number of different classes found
    - % : proportion of the final class among all classes. If the proportion are equal (for ex., 50-50), write ?
  """
  filename = opt.opdict['result_path']

  import cPickle
  with open(filename,'r') as file:
    my_depickler = cPickle.Unpickler(file)
    dic = my_depickler.load()
    file.close()

  list_ev = []
  for key in sorted(dic):
    list_ev = list_ev + list(dic[key]['list_ev'])
  list_ev = np.array(list_ev)
  list_ev_all = np.unique(list_ev)

  df = pd.DataFrame(index=list_ev_all,columns=sorted(dic),dtype=str)
  for key in sorted(dic):
    for iev,event in enumerate(dic[key]['list_ev']):
      df[key][event] = dic[key]['classification'][iev]

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

  struct.to_csv(opt.opdict['class_auto_path'])
# ================================================================
def final_result(opt):
  """
  Compare manual and automatic classifications.
  """
  manual = opt.opdict['label_filename']
  auto = opt.opdict['class_auto_path']
  print auto

  a = pd.read_csv(auto,index_col=False)
  a = a.reindex(columns=['Type'])

  m = pd.read_csv(manual,index_col=False)
  m = m.reindex(columns=['Type'],index=a.index)

  m = m.dropna(how='any')
  a = a.reindex(index=m.index)

  for i in range(len(m.values)):
    m.values[i][0] = m.values[i][0].replace(" ","")

  N = len(m)
  list_man = m.values.ravel()
  list_auto = a.values.ravel()
  sim = np.where(list_man==list_auto)[0]
  list_auto_sim = list_auto[sim]
  print "% of well classified events :", len(sim)*100./N
  print "\n"

  types = np.unique(list_auto)
  print np.unique(list_auto)
  print np.unique(list_man)
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

