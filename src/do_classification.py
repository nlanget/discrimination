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

    if len(opt.opdict['stations']) == 1 and opt.opdict['boot'] > 1:
      if os.path.exists(opt.opdict['train_file']):
        TRAIN_Y = opt.read_binary_file(opt.opdict['train_file'])
      else:
        TRAIN_Y = []

    for b in range(opt.opdict['boot']):
      print "\n-------------------- # iter: %d --------------------\n"%(b+1)

      subsubdic = {}

      y_test = y_ref.copy()

      if len(opt.opdict['stations']) == 1 and opt.opdict['boot'] > 1:
        if len(TRAIN_Y) > b:
          y_train = y_ref.reindex(index=TRAIN_Y[b])
          y_train = y_train.dropna(how='any')
        else:
          y_train = create_training_set(y_ref,opt.numt)
          list_ev_train = y_train.index        
          TRAIN_Y.append(list(y_train.index))

      else:
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

      subsubdic['list_ev'] = np.array(y_test.index)

      x_test.index = range(x_test.shape[0])
      y_test.index = range(y_test.shape[0])
      print x_test.shape, y_test.shape
      if x_test.shape[0] != y_test.shape[0]:
        print "Test set: Incoherence in x and y dimensions"
        sys.exit()

      if opt.opdict['plot_pdf']:
        opt.plot_all_pdfs(save=opt.opdict['save_pdf'])

      if opt.opdict['method'] == '1b1':
        # EXTRACTEURS
        print "********** EXTRACTION 1-BY-1 **********"
        from extraction import one_by_one
        one_by_one(opt,x_test,y_test,set['Otime'],boot=10,method='svm')
        continue

      elif opt.opdict['method'] == 'ova':
        print "********** EXTRACTION 1-VS-ALL **********"
        from extraction import one_vs_all
        one_vs_all(opt,x_test,y_test,set['Otime'],boot=10,method='svm')
        continue

      elif opt.opdict['method'] == 'svm':
        # SVM
        print "********** SVM **********"
        CLASS_test, pourcentages = implement_svm(x_train,x_test,y_train,y_test,opt.types,opt.opdict)

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
          confusion(y_train,CLASS_train,opt.types,'Training','Logistic regression',plot=opt.opdict['plot_confusion'])
          if opt.opdict['plot_confusion'] and opt.opdict['save_confusion']:
            plt.savefig('%s/figures/training_%s.png'%(opt.opdict['outdir'],opt.opdict['result_file'][8:]))

        print "\t Test set"
        for i in range(K):
          print i, opt.types[i], len(np.where(y_test.values[:,0]==i)[0]), len(np.where(CLASS_test==i)[0])
        print "\n"
        if opt.opdict['boot'] == 1:
          confusion(y_test,CLASS_test,opt.types,'Test','Logistic regression',plot=opt.opdict['plot_confusion'])
          if opt.opdict['plot_confusion']:
            if opt.opdict['save_confusion']:
              plt.savefig('%s/figures/test_%s.png'%(opt.opdict['outdir'],opt.opdict['result_file'][8:]))
            plt.show()

      subsubdic['%'] = pourcentages
      trad_CLASS_test = []
      for i in CLASS_test:
        i = int(i)
        trad_CLASS_test.append(opt.types[i])
      subsubdic['classification'] = trad_CLASS_test
      subdic[b] = subsubdic

    dic_results[opt.trad[isc]] = subdic

  dic_results['features'] = opt.opdict['feat_list']

  opt.write_binary_file(opt.opdict['result_path'],dic_results)

  if not os.path.exists(opt.opdict['train_file']) and opt.opdict['boot'] > 1:
    opt.write_binary_file(opt.opdict['train_file'],TRAIN_Y)

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
def confusion(y,y_auto,l,set,method,plot=False,output=False):
  """
  Computes the confusion matrix
  """
  from sklearn.metrics import confusion_matrix
  cmat = confusion_matrix(y.values[:,0],y_auto)
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
          plt.text(i,j,"%d"%np.around(cmat[j,i]),color=col)
    plt.title('%s set - %s'%(set,method.upper()))
    plt.xlabel('Prediction')
    plt.ylabel('Observation')
    if len(l) <= 4:
      plt.xticks(range(len(l)),l)
      plt.yticks(range(len(l)),l)
  if output:
    return cmat
# ================================================================
def implement_svm(x_train,x_test,y_train,y_test,types,opdict):
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
  if opdict['boot'] == 1:
    confusion(y_train,y_train_SVM,types,'Training','SVM',plot=opdict['plot_confusion'])
    if opdict['plot_confusion'] and opdict['save_confusion']:
      plt.savefig('%s/figures/training_%s.png'%(opdict['outdir'],opdict['result_file'][8:]))

  print "\t *Test set"
  diff = y_test.values.ravel() - y_test_SVM
  p_test = float(len(np.where(diff==0)[0]))/y_test.shape[0]*100
  print "Correct classification: %.2f%%"%p_test
  for i in range(len(np.unique(y_train.values))):
    print i, types[i], len(np.where(y_test.values[:,0]==i)[0]), len(np.where(y_test_SVM==i)[0])
  if opdict['boot'] == 1:
    confusion(y_test,y_test_SVM,types,'Test','SVM',plot=opdict['plot_confusion'])
    if opdict['plot_confusion']:
      if opdict['save_confusion']:
        plt.savefig('%s/figures/test_%s.png'%(opdict['outdir'],opdict['result_file'][8:]))
      plt.show()
  return y_test_SVM,(p_tr,p_test)

