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

  list_attr = opt.__dict__.keys()
  if not 'x' in list_attr:
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

      print "# types in the test set:",len(np.unique(y_test.NumType.values.ravel()))
      print "# types in the training set:",len(np.unique(y_train.NumType.values.ravel()))
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

      opt.train_x = x_train
      opt.x = x_test
      opt.train_y = y_train
      opt.y = y_test

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
        out = implement_svm(x_train,x_test,y_train,y_test,opt.types,opt.opdict,kern=kern,proba=opt.opdict['probas'])

        if 'thetas' in sorted(out):
          theta_vec = out['thetas']
          theta,threshold = {},{}
          for it in range(len(theta_vec)):
            theta[it+1] = np.append(theta_vec[it][-1],theta_vec[it][:-1])
            threshold[it+1] = 0.5

      elif opt.opdict['method'] == 'lrsk':
        # LOGISTIC REGRESSION (scikit learn)
        print "********* Logistic regression (sklearn) **********"
        out = implement_lr_sklearn(x_train,x_test,y_train,y_test)
        theta,threshold = {},{}
        for it in range(len(out['thetas'])):
          theta[it+1] = out['thetas'][it]
          threshold[it+1] = 0.5

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
            if len(wtr[wtr>len(wtr)-1]) > 0:
              rep_wtr = np.array(wtr_ini[len(y_train):])
              wtr[wtr>len(wtr)-1] = rep_wtr[rep_wtr<len(wtr)]
        out = do_all_logistic_regression(x_train,y_train,x_test,y_test,wtr=wtr)
        theta = out['thetas']
        threshold = out['threshold']
        if 'learn_file' in sorted(opt.opdict):
          if not os.path.exists(learn_filename):
            wtr = write_binary_file(learn_filename,out['training_set'])

      CLASS_test = out['label_test']
      CLASS_train = out['label_train']

      # TRAINING SET
      print "\t *TRAINING SET"
      y_train_np = y_train.NumType.values.ravel()  
      from sklearn.metrics import confusion_matrix
      cmat = confusion_matrix(y_train_np,CLASS_train)
      p_tr = dic_percent(cmat,y_train.shape[0],opt.types,verbose=True)
      if opt.opdict['plot_confusion'] or opt.opdict['save_confusion']:
        plot_confusion_mat(cmat,opt.types,'Training',opt.opdict['method'].upper())
        if opt.opdict['save_confusion']:
          plt.savefig('%s/figures/training_%s.png'%(opt.opdict['outdir'],opt.opdict['result_file'][8:]))

      # TEST SET
      print "\t *TEST SET"
      y_test_np = y_test.NumType.values.ravel()
      cmat = confusion_matrix(y_test_np,CLASS_test)
      p_test = dic_percent(cmat,y_test.shape[0],opt.types,verbose=True)
      if opt.opdict['plot_confusion'] or opt.opdict['save_confusion']:
        plot_confusion_mat(cmat,opt.types,'Training',opt.opdict['method'].upper())
        if opt.opdict['save_confusion']:
          plt.savefig('%s/figures/test_%s.png'%(opt.opdict['outdir'],opt.opdict['result_file'][8:]))
        if opt.opdict['plot_confusion']:
          plt.show()
        else:
          plt.close()

      # PLOT PRECISION AND RECALL
      if opt.opdict['plot_prec_rec']:
        from LR_functions import normalize,plot_precision_recall
        x_train, x_test = normalize(x_train,x_test)
        plot_precision_recall(x_train,y_train,x_test,y_test,theta)

      try:
        opt.theta = theta
        opt.threshold = threshold
      except:
        opt.theta = None
        opt.threshold = None

      pourcentages = (p_tr['global'],p_test['global'])
      opt.success = p_test

      n_feat = x_train.shape[1] # number of features
      if n_feat < 4:
        if opt.opdict['plot_sep'] or opt.opdict['save_sep']:
          print "\nPLOTTING"
          print "Theta values:",theta
          print "Threshold:", threshold

        # COMPARE AND PLOT LR AND SVM RESULTS
        if opt.opdict['method']=='lr' and opt.opdict['compare']:
          dir = 'LR_SVM_SEP'
          out_svm = implement_svm(x_train,x_test,y_train,y_test,opt.types,opt.opdict,kern='Lin')
          cmat_svm_tr = confusion_matrix(y_train_np,out_svm['label_train'])
          cmat_svm_test = confusion_matrix(y_test_np,out_svm['label_test'])
          svm_ptr = dic_percent(cmat_svm_tr,y_train.shape[0],opt.types)
          svm_pt = dic_percent(cmat_svm_test,y_test.shape[0],opt.types)
          theta_svm,t_svm = {},{}
          for it in range(len(out_svm['thetas'])):
            theta_svm[it+1] = np.append(out_svm['thetas'][it][-1],out_svm['thetas'][it][:-1])
            t_svm[it+1] = 0.5

        else:
          dir = '%s_SEP'%opt.opdict['method']

        from LR_functions import normalize
        x_train, x_test = normalize(x_train,x_test)

        x_train_good = x_train.reindex(index=y_train[y_train.NumType.values==CLASS_train].index)
        x_train_bad = x_train.reindex(index=y_train[y_train.NumType.values!=CLASS_train].index)
        good_train = y_train.reindex(index=x_train_good.index)

        x_test_good = x_test.reindex(index=y_test[y_test.NumType.values==CLASS_test].index)
        x_test_bad = x_test.reindex(index=y_test[y_test.NumType.values!=CLASS_test].index)

        # PLOT FOR 1 ATTRIBUTE AND 2 CLASSES
        if n_feat == 1 and len(opt.opdict['types']) == 2:
          name = opt.opdict['feat_list'][0]
          from plot_functions import plot_hyp_func_1f
          if opt.opdict['method']=='lr' and opt.opdict['compare']:
            plot_hyp_func_1f(x_train,y_train,theta,opt.opdict['method'],threshold=threshold,x_ok=x_test_good,x_bad=x_test_bad,p_test=p_test,p_tr=p_tr,th_comp=theta_svm,pcomp_test=svm_pt,pcomp_tr=svm_ptr)
          else:
            plot_hyp_func_1f(x_train,y_train,theta,opt.opdict['method'],threshold=threshold,x_ok=x_test_good,x_bad=x_test_bad,p_test=p_test,p_tr=p_tr)

        # PLOT FOR 2 ATTRIBUTES AND 2 to 3 CLASSES
        elif n_feat == 2:
          from plot_2features import plot_2f_all
          if opt.opdict['method']=='lr' and opt.opdict['compare']:
            plot_2f_all(theta,threshold,p_test,opt.opdict['method'],x_train,y_train,x_test,y_test,x_test_bad,opt.types,text=text,th_comp=theta_svm,t_comp=t_svm,p=svm_p)
          else:
            plot_2f_all(theta,threshold,p_test,opt.opdict['method'],x_train,y_train,x_test,y_test,x_test_bad,opt.types,text=text)
          name = '%s_%s'%(opt.opdict['feat_list'][0],opt.opdict['feat_list'][1])

        # PLOT FOR 3 ATTRIBUTES
        elif n_feat == 3:
          from plot_functions import plot_db_3d
          plot_db_3d(x_train,y_train.NumType,theta[1],title='Training set')
          plot_db_3d(x_test,y_test.NumType,theta[1],title='Test set')
          name = '%s_%s_%s'%(opt.opdict['feat_list'][0],opt.opdict['feat_list'][1],opt.opdict['feat_list'][2])

        if opt.opdict['save_sep']:
          plt.savefig('%s/%s/CL_sep_%s.png'%(opt.opdict['fig_path'],dir,name))
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
      if opt.opdict['probas']:
        subsubdic['proba'] = probas
      subdic[b] = subsubdic

    dic_results[opt.trad[isc]] = subdic

  dic_results['header'] = {}
  dic_results['header']['features'] = opt.opdict['feat_list']
  dic_results['header']['types'] = opt.opdict['types']
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
    a = y[y.NumType==i]
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
    a = y[y.NumType==i]
    if len(a) < 3:
      continue
    aperm = np.random.permutation(a.index)
    aperm = aperm[:nb]
    list_index = np.hstack((list_index,list(aperm)))
  y_train = y.reindex(index=map(int,list_index))
  return y_train

# ================================================================

def plot_confusion_mat(cmat,l,set,method):
  """
  Plots the confusion matrix
  """
  cmat = np.array(cmat,dtype=float)
  for i in range(cmat.shape[0]):
    cmat[i,:] = cmat[i,:]*100./np.sum(cmat[i,:])

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

# ================================================================
def dic_percent(cmat,nb_ev,types,verbose=False):
  p = {}
  p['global'] = np.sum(np.diag(cmat))*1./nb_ev*100
  NB_class = cmat.shape[0]
  for i in range(NB_class):
    l_man = np.sum(cmat[i,:])
    l_auto = np.sum(cmat[:,i])
    p_cl = cmat[i,i]*1./l_man*100
    p[('%s'%types[i],i)] = '%.2f'%p_cl
    if verbose:
        print "Extraction of %s (%d) : %.2f%%"%(types[i],i,p_cl)
  return p
# ================================================================

def implement_svm(x_train,x_test,y_train,y_test,types,opdict,kern='NonLin',proba=False):
  """
  Implements SVM from scikit learn package.
  Options : 
  - kernel : could be 'Lin' (for linear) or 'NonLin' (for non-linear). In the latter 
  case, the kernel is a gaussian kernel.
  - proba : tells if the probability estimates must be computed

  Returns : 
  - y_test_SVM : classification predicted by SVM for the test set
  - (p_tr,p_test) : success rates of the training and test sets respectively

  If proba is True, also returns :
  - the probability estimates for each element of the dataset

  If kernel is linear, also returns : 
  - y_train_SVM : classification predicted by SVM for the training set
  - grid.best_estimator_.raw_coef_ : coefficients of the linear decision boundary
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
    grid = GridSearchCV(svm.SVC(probability=proba), param_grid=param_grid, n_jobs=-1)

  elif kern == 'Lin':
    param_grid = dict(C=C_range)
    grid = GridSearchCV(svm.LinearSVC(), param_grid=param_grid, n_jobs=-1)

  grid.fit(x_train.values, y_train.NumType.values.ravel())
  print "The best classifier is: ", grid.best_estimator_

  if kern == 'NonLin':
    print "Number of support vectors for each class: ", grid.best_estimator_.n_support_
  y_train_SVM = grid.best_estimator_.predict(x_train)
  y_test_SVM = grid.best_estimator_.predict(x_test)

  output = {}
  output['label_test'] = y_test_SVM
  output['label_train'] = y_train_SVM
  if proba:
    probabilities = grid.best_estimator_.predict_proba(x_test)
    output['probas'] = {}
    for k in range(NB_class):
      output['probas'][types[k]] = probabilities[:,k]
  if kern == 'Lin':
    output['thetas'] = grid.best_estimator_.raw_coef_
  return output

# ================================================================

def implement_lr_sklearn(x_train,x_test,y_train,y_test):
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
  grid.fit(x_train.values, y_train.NumType.values.ravel())
  print "The best classifier is: ", grid.best_estimator_
  y_train_LR = grid.best_estimator_.predict(x_train)
  y_test_LR = grid.best_estimator_.predict(x_test)

  output = {}
  output['label_test'] = y_test_LR
  output['label_train'] = y_train_LR
  output['thetas'] = grid.best_estimator_.raw_coef_
  return output

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
        EXT[n]['nb_%s'%types[t]].append(len(y_test[y_test.NumType==t]))

      y_train[y_train_ref.NumType==n] = 0
      y_test[y_test_ref.NumType==n] = 0
      y_train[y_train_ref.NumType!=n] = 1
      y_test[y_test_ref.NumType!=n] = 1

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
        grid.fit(x_train.values, y_train.NumType.values.ravel())
        LR_train = grid.best_estimator_.predict(x_train)
        LR_test = grid.best_estimator_.predict(x_test)

      print "\t Training set"
      for i in range(2):
        print i, t[i], len(np.where(y_train.NumType.values[:,0]==i)[0]), len(np.where(LR_train==i)[0])
      print "\n"
      cmat_train = confusion(y_train,LR_train,types,'training','LogReg',plot=False)

      print "\t Test set"
      for i in range(2):
        print i, t[i], len(np.where(y_test.NumType.values[:,0]==i)[0]), len(np.where(LR_test==i)[0])
      print "\n"
      cmat_test = confusion(y_test,LR_test,types,'test','LogReg',plot=False)

      # Fill the dictionary
      i_com = np.where((y_test.NumType.values.ravel()-LR_test)==0)[0]
      i_lr = np.where(LR_test==0)[0]
      i_ok_class = np.intersect1d(i_com,i_lr) # events classified in the class of interest by the LR and identical to the manual classification

      sub_dic["nb"] = len(i_lr) # total number of events classified in the class of interest
      sub_dic["nb_common"] = len(i_ok_class)
      sub_dic["index_ok"] = otimes[i_ok_class]
      sub_dic["nb_other"],sub_dic["i_other"] = [],[]
      for k in range(len_numt):
        if k != n:
          i_other_man = list(y_test_ref[y_test_ref.NumType==k].index)
          ii = np.intersect1d(i_lr,i_other_man)
          sub_dic["nb_other"].append((types[k],len(ii)))
          sub_dic["i_other"].append((types[k],otimes[ii]))
      sub_dic["rate_%s"%types[n]] = (cmat_train[0,0],cmat_test[0,0])
      sub_dic["rate_rest"] = (cmat_train[1,1],cmat_test[1,1])
      sub_dic["nb_manuals"] = ((types[n],len(y_test[y_test.NumType==0])),('Rest',len(y_test[y_test.NumType==1])))

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
      y_train[y_train_ref.NumType==n] = 0
      y_test[y_test_ref.NumType==n] = 0
      y_train[y_train_ref.NumType!=n] = 1
      y_test[y_test_ref.NumType!=n] = 1

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
        grid.fit(x_train.values, y_train.NumType.values.ravel())
        LR_train = grid.best_estimator_.predict(x_train)
        LR_test = grid.best_estimator_.predict(x_test)

      print "\t Training set"
      for i in range(2):
        print i, print_type[i], len(np.where(y_train.NumType.values[:,0]==i)[0]), len(np.where(LR_train==i)[0])
      print "\n"
      cmat_train = confusion(y_train,LR_train,types,'training','LogReg',plot=True)
      plt.close()

      print "\t Test set"
      for i in range(2):
        print i, print_type[i], len(np.where(y_test.NumType.values[:,0]==i)[0]), len(np.where(LR_test==i)[0])
      print "\n"
      cmat_test = confusion(y_test,LR_test,types,'test','LogReg',plot=True)
      plt.close()

      # Fill the dictionary
      sub_dic={}
      i_com = np.where((y_test.NumType.values.ravel()-LR_test)==0)[0]
      i_lr = np.where(LR_test==0)[0]
      i_ok_class = np.intersect1d(i_com,i_lr) # events classified in the class of interest by the LR and identical to the manual classification
      sub_dic["nb"] = len(i_lr) # total number of events classified in the class of interest
      sub_dic["nb_common"] = len(i_ok_class) # total number of well classified events
      sub_dic["index_ok"] = otimes[i_ok_class] # index of well classified events
      sub_dic["nb_other"],sub_dic["i_other"] = [],[]
      for k in range(len_numt):
        if k != n:
          i_other_man = list(y_test_ref[y_test_ref.NumType==k].index)
          ii = np.intersect1d(i_lr,i_other_man)
          sub_dic["nb_other"].append((types[k],len(ii))) # number of events belonging to another class
          sub_dic["i_other"].append((types[k],otimes[ii])) # index of events belonging to another class
      sub_dic["rate_%s"%types[n]] = (cmat_train[0,0],cmat_test[0,0]) # % success rate of the extracted class
      sub_dic["rate_rest"] = (cmat_train[1,1],cmat_test[1,1]) # % success rate of the rest
      sub_dic["nb_manuals"] = ((types[n],len(y_test[y_test.NumType==0])),('Rest',len(y_test[y_test.NumType==1])))
      dic[types[n]] = sub_dic

    DIC[b] = dic

  file = opt.opdict['result_path']
  print "One-vs-All results stored in %s"%file
  write_binary_file(file,DIC)
