#!/usr/bin/env python
# encoding: utf-8

import os,glob,sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from LR_functions import comparison
from options import read_binary_file, write_binary_file

def classifier(opt):

  """
  Classification of the different types of events.
  opt is an object of the class Options()
  """

  opt.do_tri()
  X = opt.x
  Y = opt.y

  list_attr = opt.__dict__.keys()
  if 'train_x' in list_attr:
    X_TRAIN = opt.train_x
    Y_TRAIN = opt.train_y

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


    # About the training set
    if len(opt.opdict['stations']) == 1 and opt.opdict['boot'] > 1 and 'train_x' not in list_attr:
      if os.path.exists(opt.opdict['train_file']):
        TRAIN_Y = read_binary_file(opt.opdict['train_file'])
      else:
        TRAIN_Y = []
    elif 'train_x' in list_attr:
      opt.x = opt.xs_train[isc]
      opt.y = opt.ys_train[isc]
      if opt.opdict['plot_pdf']:
        opt.compute_pdfs()
        g_train = opt.gaussians
        del opt.gaussians
      opt.classname2number()
      x_ref_train = opt.x
      y_ref_train = opt.y


    # About the test set
    opt.x = opt.xs[isc]
    opt.y = opt.ys[isc]
    if opt.opdict['plot_pdf']:
      opt.compute_pdfs()
 
    set = pd.DataFrame(index=opt.ys[isc].index,columns=['Otime'])
    set['Otime'] = opt.xs[isc].index

    opt.classname2number()
    x_test = opt.x
    y_ref = opt.y
    x_ref = opt.x

    K = len(opt.types)

    for b in range(opt.opdict['boot']):
      print "\n-------------------- # iter: %d --------------------\n"%(b+1)

      subsubdic = {}

      x_test = x_ref.copy()
      y_test = y_ref.copy()

      if 'train_x' not in list_attr:
        x_train = x_test.copy()
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

      else:
        x_train = x_ref_train.copy()
        y_train = y_ref_train.copy()

      y = y_train.copy()

      in_train = np.intersect1d(np.array(y_test.index),np.array(y_train.index))
      set[b] = np.zeros(set.shape[0])
      set[b][in_train] = 1

      y_train = y.copy()
      x_train = x_train.reindex(index=y_train.index)
      x_test = x_test.reindex(index=y_test.index)

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
        if 'train_x' in list_attr:
          opt.plot_superposed_pdfs(g_train,save=opt.opdict['save_pdf'])
        else:
          opt.plot_all_pdfs(save=opt.opdict['save_pdf'])

      if opt.opdict['method'] == '1b1':
        # EXTRACTEURS
        print "********** EXTRACTION 1-BY-1 **********"
        one_by_one(opt,x_test,y_test,set['Otime'],boot=10,method='svm')
        continue

      elif opt.opdict['method'] == 'ova':
        print "********** EXTRACTION 1-VS-ALL **********"
        one_vs_all(opt,x_test,y_test,set['Otime'],boot=10,method='svm')
        continue

      elif opt.opdict['method'] == 'svm':
        # SVM
        print "********** SVM **********"
        kern = 'Lin'
        if kern == 'NonLin':
          CLASS_test, pourcentages = implement_svm(x_train,x_test,y_train,y_test,opt.types,opt.opdict,kern='NonLin')
        elif kern == 'Lin':
          CLASS_test, pourcentages, CLASS_train, theta_vec = implement_svm(x_train,x_test,y_train,y_test,opt.types,opt.opdict,kern='Lin')
          theta,threshold = {},{}
          for it in range(len(theta_vec)):
            theta[it+1] = theta_vec[it]
            threshold[it+1] = None

      elif opt.opdict['method'] == 'lrsk':
        # LOGISTIC REGRESSION (scikit learn)
        print "********* Logistic regression (sklearn) **********"
        CLASS_test, pourcentages, CLASS_train, theta_vec = implement_lr_sklearn(x_train,x_test,y_train,y_test,opt.types,opt.opdict)
        theta,threshold = {},{}
        for it in range(len(theta_vec)):
          theta[it+1] = theta_vec[it]
          threshold[it+1] = None

      elif opt.opdict['method'] == 'lr':
        # LOGISTIC REGRESSION
        print "********* Logistic regression **********"
        from LR_functions import do_all_logistic_regression
        wtr = np.array([])
        if 'learn_file' in sorted(opt.opdict):
          learn_filename = '%s_%d'%(opt.opdict['learn_file'],b+1)
          if os.path.exists(learn_filename):
            wtr_ini = read_binary_file(learn_filename)
            wtr = wtr_ini[:len(y_train)]
            wtr = np.array(wtr)
            wtr[wtr>len(wtr)-1] = wtr_ini[len(y_train):]
        CLASS_train,theta,CLASS_test,threshold,pourcentages,wtr = do_all_logistic_regression(x_train,y_train,x_test,y_test,output=True,perc=True,wtr=wtr,ret_thres=True)
        if 'learn_file' in sorted(opt.opdict):
          if not os.path.exists(learn_filename):
            wtr = write_binary_file(learn_filename,wtr)
        print "\t Training set"
        for i in range(K):
          print i, opt.types[i], len(np.where(y_train.values[:,0]==i)[0]), len(np.where(CLASS_train==i)[0])
        print "\n"
        if opt.opdict['boot'] == 1:
          confusion(y_train,CLASS_train,opt.types,'Training','Logistic regression',plot=opt.opdict['plot_confusion'])
          if opt.opdict['plot_confusion'] and opt.opdict['save_confusion']:
            plt.savefig('%s/figures/training_%s_%s_%s.png'%(opt.opdict['outdir'],opt.opdict['result_file'][8:],opt.trad[isc][0],opt.trad[isc][1]))

        print "\t Test set"
        for i in range(K):
          print i, opt.types[i], len(np.where(y_test.values[:,0]==i)[0]), len(np.where(CLASS_test==i)[0])
        print "\n"
        if opt.opdict['boot'] == 1:
          confusion(y_test,CLASS_test,opt.types,'Test','Logistic regression',plot=opt.opdict['plot_confusion'])
          if opt.opdict['plot_confusion']:
            if opt.opdict['save_confusion']:
              plt.savefig('%s/figures/test_%s_%s_%s.png'%(opt.opdict['outdir'],opt.opdict['result_file'][8:],opt.trad[isc][0],opt.trad[isc][1]))
            plt.show()


      if opt.opdict['plot_prec_rec']:
        from LR_functions import normalize,plot_precision_recall
        x_train, x_test = normalize(x_train,x_test)
        plot_precision_recall(x_train,y_train,x_test,y_test,theta)
        plt.show()


      n_feat = x_train.shape[1] # number of features
      if len(opt.types) == 2:
        if opt.opdict['plot_sep'] or opt.opdict['save_sep']:
          print "Theta values:",theta

          from LR_functions import normalize
          x_train, x_test = normalize(x_train,x_test)

          x_train_good = x_train.reindex(index=y_train[y_train.Type.values==CLASS_train].index)
          x_train_bad = x_train.reindex(index=y_train[y_train.Type.values!=CLASS_train].index)
          good_train = y_train.reindex(index=x_train_good.index)
          p_good_cl0 = len(good_train[good_train.Type==0])*1./len(y_train[y_train.Type==0])*100
          p_good_cl1 = len(good_train[good_train.Type==1])*1./len(y_train[y_train.Type==1])*100

          x_test_good = x_test.reindex(index=y_test[y_test.Type.values==CLASS_test].index)
          x_test_bad = x_test.reindex(index=y_test[y_test.Type.values!=CLASS_test].index)
          p_good_test = len(x_test_good)*1./len(x_test)*100
          text = [p_good_cl0,p_good_cl1,p_good_test,100-p_good_test]

          if n_feat == 1:
            from LR_functions import hypothesis
            from plot_functions import plot_hyp_func_1f, plot_sep_1f
            mins=[x_train.min(),x_test.min()]
            maxs=[x_train.max(),x_test.max()]
            syn, hyp = hypothesis(mins,maxs,theta[1])
            plot_sep_1f(x_train,y_train,theta=theta[1],str_t=opt.types,x_ok=x_test_good,x_bad=x_test_bad,text=text)
            plot_hyp_func_1f(x_train,y_train,syn,hyp,threshold=threshold[1],str_t=opt.types,x_ok=x_test_good,x_bad=x_test_bad,text=text)
            name = opt.opdict['feat_list'][0]

          elif n_feat == 2:
            from plot_functions import plot_sep_2f
            plot_sep_2f(x_train,y_train.Type,opt.types,x_test,y_test.Type,x_test_bad,theta=theta[1],text=text)
            name = '%s_%s'%(opt.opdict['feat_list'][0],opt.opdict['feat_list'][1])

          elif n_feat == 3:
            from plot_functions import plot_db_3d
            plot_db_3d(x_train,y_train.Type,theta[1],title='Training set')
            plot_db_3d(x_test,y_test.Type,theta[1],title='Test set')
            name = '%s_%s_%s'%(opt.opdict['feat_list'][0],opt.opdict['feat_list'][1],opt.opdict['feat_list'][2])

        if opt.opdict['save_sep']:
          plt.savefig('%s/HYP/sep_%s.png'%(opt.opdict['fig_path'],name))
        if opt.opdict['plot_sep']:
          plt.show()
        else:
          plt.close()

      subsubdic['%'] = pourcentages
      trad_CLASS_test = []
      for i in CLASS_test:
        i = int(i)
        trad_CLASS_test.append(opt.types[i])
      subsubdic['classification'] = trad_CLASS_test
      subdic[b] = subsubdic

    dic_results[opt.trad[isc]] = subdic

  dic_results['header'] = {}
  dic_results['header']['features'] = opt.opdict['feat_list']
  dic_results['header']['types'] = opt.opdict['Types']
  dic_results['header']['catalog'] = opt.opdict['label_test']

  if opt.opdict['method'] == 'lr' or opt.opdict['method'] == 'lrsk' or opt.opdict['method'] == 'svm':
    write_binary_file(opt.opdict['result_path'],dic_results)

  if 'train_file' in sorted(opt.opdict):
    if not os.path.exists(opt.opdict['train_file']) and opt.opdict['boot'] > 1:
      write_binary_file(opt.opdict['train_file'],TRAIN_Y)

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
  y_train.index = map(int,list_index)
  return y_train

# ================================================================

def create_training_set_fix(y_ref,numt):
  """
  Generates a training set randomly from the test set.
  The number of events in each class is the same. 
  """
  nb = 400
  y = y_ref.copy()
  y_train = pd.DataFrame(columns=y.columns)
  list_index = np.array([])
  for i in numt:
    a = y[y.Type==i]
    if len(a) < 3:
      continue
    aperm = np.random.permutation(a.index)
    aperm = aperm[:nb]
    list_index = np.hstack((list_index,list(aperm)))
  y_train = y.reindex(index=map(int,list_index))
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
        if cmat[j,i] >= np.max(cmat)/2. or cmat[j,i] > 50:
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

def implement_svm(x_train,x_test,y_train,y_test,types,opdict,kern='NonLin'):
  """
  Implements SVM from scikit learn package.
  """
  from LR_functions import normalize
  x_train, x_test = normalize(x_train,x_test)

  # do grid search
  from sklearn.grid_search import GridSearchCV
  from sklearn import svm
  print "doing grid search"
  C_range = 10.0 ** np.arange(-2, 5)
  if kern == 'NonLin':
    gamma_range = 10.0 ** np.arange(-3,3)
    param_grid = dict(gamma=gamma_range, C=C_range)
    grid = GridSearchCV(svm.SVC(), param_grid=param_grid, n_jobs=-1)
  elif kern == 'Lin':
    param_grid = dict(C=C_range)
    grid = GridSearchCV(svm.LinearSVC(), param_grid=param_grid, n_jobs=-1)
  grid.fit(x_train.values, y_train.values.ravel())
  print "The best classifier is: ", grid.best_estimator_
  if kern == 'NonLin':
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

  if kern == 'Lin':
    return y_test_SVM,(p_tr,p_test),y_train_SVM,grid.best_estimator_.raw_coef_
  else:
    return y_test_SVM,(p_tr,p_test)

# ================================================================

def implement_lr_sklearn(x_train,x_test,y_train,y_test,types,opdict):
  """
  Implements logistic regression from scikit learn package.
  """
  from LR_functions import normalize
  #x_train, x_test = normalize(x_train,x_test)

  from sklearn.grid_search import GridSearchCV
  from sklearn.linear_model import LogisticRegression

  print "doing grid search"
  C_range = 10.0 ** np.arange(-2, 5)
  param_grid = dict(C=C_range)
  grid = GridSearchCV(LogisticRegression(), param_grid=param_grid, n_jobs=-1)
  grid.fit(x_train.values, y_train.values.ravel())
  print "The best classifier is: ", grid.best_estimator_
  y_train_LR = grid.best_estimator_.predict(x_train)
  y_test_LR = grid.best_estimator_.predict(x_test)

  print "\t *Training set"
  diff = y_train.values.ravel() - y_train_LR
  p_tr = float(len(np.where(diff==0)[0]))/y_train.shape[0]*100
  print "Correct classification: %.2f%%"%p_tr
  for i in range(len(np.unique(y_train.values))):
    print i, types[i], len(np.where(y_train.values[:,0]==i)[0]), len(np.where(y_train_LR==i)[0])
  if opdict['boot'] == 1:
    confusion(y_train,y_train_LR,types,'Training','LR',plot=opdict['plot_confusion'])
    if opdict['plot_confusion'] and opdict['save_confusion']:
      plt.savefig('%s/figures/training_%s.png'%(opdict['outdir'],opdict['result_file'][8:]))

  print "\t *Test set"
  diff = y_test.values.ravel() - y_test_LR
  p_test = float(len(np.where(diff==0)[0]))/y_test.shape[0]*100
  print "Correct classification: %.2f%%"%p_test
  for i in range(len(np.unique(y_train.values))):
    print i, types[i], len(np.where(y_test.values[:,0]==i)[0]), len(np.where(y_test_LR==i)[0])
  if opdict['boot'] == 1:
    confusion(y_test,y_test_LR,types,'Test','LR',plot=opdict['plot_confusion'])
    if opdict['plot_confusion']:
      if opdict['save_confusion']:
        plt.savefig('%s/figures/test_%s.png'%(opdict['outdir'],opdict['result_file'][8:]))
      plt.show()
  return y_test_LR,(p_tr,p_test),y_train_LR,grid.best_estimator_.raw_coef_

# ================================================================

def one_by_one(opt,x_test_ref0,y_test_ref0,otimes_ref,boot=1,method='lr'):

  """
  Extract one class after each other by order of importance. The events which are 
  classified are deleted from the next extraction.
  boot = number of training sets to be generated
  method = 'lr' for Logistic Regression / 'svm' for SVM
  """

  from LR_functions import do_all_logistic_regression

  types = opt.types
  numt = opt.numt

  len_numt = len(numt)
  # Dictionary for results
  DIC = {}
  DIC['features'] = x_test_ref0.columns 

  EXT = {}
  for num_ext in range(len_numt):
    EXT[num_ext] = {}
    EXT[num_ext]['nb_tot'] = []
    for t in numt:
      EXT[num_ext]['nb_%s'%types[t]] = []

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

      EXT[n]['nb_tot'].append(len(x_test))
      for t in numt:
        EXT[n]['nb_%s'%types[t]].append(len(y_test[y_test.Type==t]))

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

  file = opt.opdict['result_path']
  print "One-by-One results stored in %s"%file
  write_binary_file(file,DIC)

  file = '%s/stats_OBO'%os.path.dirname(opt.opdict['result_path'])
  write_binary_file(file,EXT)

# ================================================================

def one_vs_all(opt,x_test,y_test_ref,otimes_ref,boot=1,method='lr'):

  """
  Extract one class among the whole data.
  """

  from LR_functions import do_all_logistic_regression

  types = opt.types
  numt = opt.numt
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

  file = opt.opdict['result_path']
  print "One-vs-All results stored in %s"%file
  write_binary_file(file,DIC)
