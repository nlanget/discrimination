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
    print opt.opdict['train_file']
    if len(opt.opdict['stations']) == 1 and opt.opdict['boot'] > 1 and 'train_x' not in list_attr:
      if os.path.exists(opt.opdict['train_file']):
        TRAIN_Y = read_binary_file(opt.opdict['train_file'])
      else:
        TRAIN_Y = {}
        for tir in range(opt.opdict['boot']):
          TRAIN_Y[tir] = {}
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

    if opt.opdict['plot_dataset']:
      opt.composition_dataset()

    #K = len(opt.types)

    ### ITERATE OVER TRAINING SET DRAWS ###
    for b in range(opt.opdict['boot']):
      print "\n-------------------- # iter: %d --------------------\n"%(b+1)

      subsubdic = {}
      print "WHOLE SET", x_ref.shape, y_ref.shape

      ### if there is no pre-defined training set ###
      if 'train_x' not in list_attr:
        x_train = x_test.copy()
        if len(opt.opdict['stations']) == 1 and opt.opdict['boot'] > 1:
          if len(TRAIN_Y[b]) > 0:
            y_train = y_ref.reindex(index=TRAIN_Y[b]['training_set'])
            y_train = y_train.dropna(how='any')
            y_cv = y_ref.reindex(index=TRAIN_Y[b]['cv_set'])
            y_cv = y_cv.dropna(how='any')
            y_test = y_ref.reindex(index=TRAIN_Y[b]['test_set'])
            y_test = y_test.dropna(how='any')
          else:
            y_train, y_cv, y_test = generate_datasets(opt.opdict['proportions'],opt.numt,y_ref)
            TRAIN_Y[b]['training_set'] = map(int,list(y_train.index))
            TRAIN_Y[b]['cv_set'] = map(int,list(y_cv.index))
            TRAIN_Y[b]['test_set'] = map(int,list(y_test.index))

        ### multi-stations case ###
        else:
          if marker_sta == 0:
            y_train, y_cv, y_test = generate_datasets(opt.opdict['proportions'],opt.numt,y_ref)
            list_ev_train = y_train.index
            list_ev_cv = y_cv.index
            list_ev_test = y_test.index
          else:
            y_train = y_ref.reindex(index=list_ev_train)
            y_train = y_train.dropna(how='any')
            y_cv = y_ref.reindex(index=list_ev_cv)
            y_cv = y_cv.dropna(how='any')
            y_test = y_ref.reindex(index=list_ev_test)
            y_test = y_test.dropna(how='any')

        x_train = x_ref.reindex(index=y_train.index)

      ### if a training set was pre-defined ###
      else:
        x_train = x_ref_train.copy()
        y_train = y_ref_train.copy()
        y_train, y_cv, y_test = generate_datasets(opt.opdict['proportions'],opt.numt,y_ref,y_train=y_train)

      x_cv = x_ref.reindex(index=y_cv.index)
      x_test = x_ref.reindex(index=y_test.index)

      i_train = y_train.index
      x_train.index = range(x_train.shape[0])
      y_train.index = range(y_train.shape[0])
      print "TRAINING SET", x_train.shape, y_train.shape
      if x_train.shape[0] != y_train.shape[0]:
        print "Training set: Incoherence in x and y dimensions"
        sys.exit()

      i_cv = y_cv.index
      x_cv.index = range(x_cv.shape[0])
      y_cv.index = range(y_cv.shape[0])
      print "CROSS-VALIDATION SET", x_cv.shape, y_cv.shape
      if x_cv.shape[0] != y_cv.shape[0]:
        print "Cross-validation set: Incoherence in x and y dimensions"
        sys.exit()

      subsubdic['list_ev'] = np.array(y_test.index)

      i_test = y_test.index
      x_test.index = range(x_test.shape[0])
      y_test.index = range(y_test.shape[0])
      print "TEST SET", x_test.shape, y_test.shape
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
        one_by_one(opt,x_ref,y_ref,set['Otime'],boot=10,method='svm')
        continue

      elif opt.opdict['method'] == 'ova':
        print "********** EXTRACTION 1-VS-ALL **********"
        one_vs_all(opt,x_ref,y_ref,set['Otime'],boot=10,method='svm')
        continue

      elif opt.opdict['method'] in ['svm','svm_nl']:
        # SVM
        print "********** SVM **********"
        if opt.opdict['method'] == 'svm':
          kern = 'Lin'
        else:
          kern = 'NonLin'

        out = implement_svm(x_train,x_test,y_train,y_test,opt.types,opt.opdict,kern=kern,proba=opt.opdict['probas'])

        if 'map' in sorted(out):
          opt.map = out['map']

        if 'thetas' in sorted(out):
          theta_vec = out['thetas']
          theta,threshold = {},{}
          for it in range(len(theta_vec)):
            theta[it+1] = np.append(theta_vec[it][-1],theta_vec[it][:-1])
            threshold[it+1] = 0.5
          out['thetas'] = theta
          out['threshold'] = threshold

      elif opt.opdict['method'] == 'lrsk':
        # LOGISTIC REGRESSION (scikit learn)
        print "********* Logistic regression (sklearn) **********"
        out = implement_lr_sklearn(x_train,x_test,y_train,y_test)
        threshold, theta = {},{}
        for it in range(len(out['thetas'])):
          threshold[it+1] = 0.5
          theta[it+1] = np.append(out['thetas'][it][-1],out['thetas'][it][:-1])
        out['threshold'] = threshold
        out['thetas'] = theta

      elif opt.opdict['method'] == 'lr':
        # LOGISTIC REGRESSION
        print "********* Logistic regression **********"
        from LR_functions import do_all_logistic_regression
        out = do_all_logistic_regression(x_ref,y_ref,i_train,i_cv,i_test)
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
      cmat_train = confusion_matrix(y_train_np,CLASS_train)
      p_tr = dic_percent(cmat_train,opt.types,verbose=True)
      out['rate_train'] = p_tr
      print "   Global : %.2f%%"%p_tr['global']
      if opt.opdict['plot_confusion'] or opt.opdict['save_confusion']:
        plot_confusion_mat(cmat_train,opt.types,'Training',opt.opdict['method'].upper())
        if opt.opdict['save_confusion']:
          savefig = '%s/training_%s.png'%(opt.opdict['fig_path'],opt.opdict['result_file'])
          print "Confusion matrix saved in %s"%savefig
          plt.savefig(savefig)

      # TEST SET
      print "\t *TEST SET"
      y_test_np = y_test.NumType.values.ravel()
      cmat_test = confusion_matrix(y_test_np,CLASS_test)
      p_test = dic_percent(cmat_test,opt.types,verbose=True)
      out['rate_test'] = p_test
      print "   Global : %.2f%%"%p_test['global']
      if opt.opdict['plot_confusion'] or opt.opdict['save_confusion']:
        plot_confusion_mat(cmat_test,opt.types,'Test',opt.opdict['method'].upper())
        if opt.opdict['save_confusion']:
          savefig = '%s/test_%s.png'%(opt.opdict['fig_path'],opt.opdict['result_file'])
          print "Confusion matrix saved in %s"%savefig
          plt.savefig(savefig)
        if opt.opdict['plot_confusion']:
          plt.show()
        else:
          plt.close()

      # PLOT PRECISION AND RECALL
      if opt.opdict['plot_prec_rec']:
        from LR_functions import normalize,plot_precision_recall
        x_train, x_test = normalize(x_train,x_test)
        plot_precision_recall(x_train,y_train.NumType,x_test,y_test.NumType,theta)

      pourcentages = (p_tr['global'],p_test['global'])
      out['method'] = opt.opdict['method']
      out['types'] = opt.types
      opt.out = out

      # PLOT DECISION BOUNDARIES
      n_feat = x_train.shape[1] # number of features
      if n_feat < 4:
        if opt.opdict['plot_sep'] or opt.opdict['save_sep']:
          print "\nPLOTTING"
          print "Theta values:",theta
          print "Threshold:", threshold

          # COMPARE AND PLOT LR AND SVM RESULTS
          out_svm, out_nl = {},{}
          dir = '%s_SEP'%opt.opdict['method'].upper()
          if opt.opdict['method']=='lr' and opt.opdict['compare']:
            dir = 'LR_SVM_SEP'
            out_svm = implement_svm(x_train,x_test,y_train,y_test,opt.types,opt.opdict,kern='Lin')
            cmat_svm_tr = confusion_matrix(y_train_np,out_svm['label_train'])
            cmat_svm_test = confusion_matrix(y_test_np,out_svm['label_test'])
            svm_ptr = dic_percent(cmat_svm_tr,opt.types)
            svm_pt = dic_percent(cmat_svm_test,opt.types)
            theta_svm,t_svm = {},{}
            for it in range(len(out_svm['thetas'])):
              theta_svm[it+1] = np.append(out_svm['thetas'][it][-1],out_svm['thetas'][it][:-1])
              t_svm[it+1] = 0.5
            out_svm['thetas'] = theta_svm
            out_svm['threshold'] = t_svm
            out_svm['rate_test'] = svm_pt
            out_svm['rate_train'] = svm_ptr
            out_svm['method'] = 'SVM'

          if opt.opdict['method'] in ['lr','svm'] and opt.opdict['compare_nl']:
            dir = '%s_NL_SEP'%opt.opdict['method'].upper()
            out_nl = implement_svm(x_train,x_test,y_train,y_test,opt.types,opt.opdict,kern='NonLin')
            cmat_svm_tr = confusion_matrix(y_train_np,out_nl['label_train'])
            cmat_svm_test = confusion_matrix(y_test_np,out_nl['label_test'])
            svm_ptr = dic_percent(cmat_svm_tr,opt.types)
            svm_pt = dic_percent(cmat_svm_test,opt.types)
            out_nl['rate_test'] = svm_pt
            out_nl['rate_train'] = svm_ptr
            out_nl['method'] = 'SVM_NL'

          save_dir = os.path.join(opt.opdict['fig_path'],dir)
          opt.verify_and_create(save_dir)

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
            from plot_functions import plot_hyp_func_1f, histo_pdfs
            if opt.opdict['method']=='lr' and opt.opdict['compare']:
              plot_hyp_func_1f(x_train,y_train,theta,opt.opdict['method'],threshold=threshold,x_ok=x_test_good,x_bad=x_test_bad,th_comp=theta_svm,cmat_test=cmat_test,cmat_svm=cmat_svm_test,cmat_train=cmat_train)
            else:
              #histo_pdfs(x_test,y_test,x_train=x_train,y_train=y_train)
              plot_hyp_func_1f(x_train,y_train,theta,opt.opdict['method'],threshold=threshold,x_ok=x_test_good,x_bad=x_test_bad,cmat_test=cmat_test,cmat_train=cmat_train)

          # PLOT FOR 2 ATTRIBUTES AND 2 to 3 CLASSES
          elif n_feat == 2:
            name = '%s_%s'%(opt.opdict['feat_list'][0],opt.opdict['feat_list'][1])
            if opt.opdict['method'] in ['lr','svm']:
              from plot_2features import plot_2f_all
              plot_2f_all(out,x_train,y_train,x_test,y_test,x_test_bad,out_comp=out_svm,map_nl=out_nl)
            elif opt.opdict['method'] == 'svm_nl':
              from plot_2features import plot_2f_nonlinear
              plot_2f_nonlinear(out,x_train,y_train,x_test,y_test,y_train=y_train)

          # PLOT FOR 3 ATTRIBUTES
          elif n_feat == 3:
            from plot_functions import plot_db_3d
            plot_db_3d(x_train,y_train.NumType,theta[1],title='Training set')
            plot_db_3d(x_test,y_test.NumType,theta[1],title='Test set')
            name = '%s_%s_%s'%(opt.opdict['feat_list'][0],opt.opdict['feat_list'][1],opt.opdict['feat_list'][2])

          if opt.opdict['save_sep']:
            plt.savefig('%s/CL_sep_%s.png'%(save_dir,name))
          if opt.opdict['plot_sep']:
            plt.show()
          else:
            plt.close()

      # WRITE RESULTS INTO A DICTIONARY
      subsubdic['%'] = pourcentages
      trad_CLASS_test = []
      for i in CLASS_test:
        i = int(i)
        trad_CLASS_test.append(opt.types[i])
      subsubdic['classification'] = trad_CLASS_test
      if opt.opdict['probas']:
        subsubdic['proba'] = out['probas']
      if opt.opdict['plot_var']:
        subsubdic['out'] = out
      subdic[b] = subsubdic

    if opt.opdict['plot_var'] and opt.opdict['method'] in ['lr','svm','lrsk'] and n_feat==2 and len(opt.opdict['types'])==2:
      from plot_2features import plot_2f_variability
      plot_2f_variability(subdic,x_train,y_train,x_test,y_test)
      plt.savefig('%s/%s_variability_pas.png'%(opt.opdict['fig_path'],opt.opdict['method'].upper()))
      plt.show()


    dic_results[opt.trad[isc]] = subdic

  dic_results['header'] = {}
  dic_results['header']['features'] = opt.opdict['feat_list']
  dic_results['header']['types'] = opt.opdict['types']
  dic_results['header']['catalog'] = opt.opdict['label_test']

  if opt.opdict['method'] in ['lr','lrsk','svm','svm_nl']:
    print "Save results in file %s"%opt.opdict['result_path']
    write_binary_file(opt.opdict['result_path'],dic_results)

  if 'train_file' in sorted(opt.opdict):
    if not os.path.exists(opt.opdict['train_file']) and opt.opdict['boot'] > 1:
      write_binary_file(opt.opdict['train_file'],TRAIN_Y)

# ================================================================

def create_training_set(y_ref,numt,prop):

  """
  Generates a training set randomly from the test set.
  The proportions of each class constituting the test set are kept.
  """

  y = y_ref.copy()
  y_train = pd.DataFrame(columns=y.columns)
  list_index = np.array([])
  for i in numt:
    #print i, y.NumType.values[:5]
    a = y[y.NumType==i]
    nb = int(np.floor(prop*len(a)))
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
def generate_datasets(proportions,numtype,y_ref,y_train=None):
  """
  Split the whole dataset into :
    * a training set
    * a cross-validation set
    * a test set
  Behave differently if a training set does already exist.
  """
  y = y_ref.copy()
  prop_train, prop_cv, prop_test = proportions
  if not y_train:
    y_train = create_training_set(y,numtype,prop_train)
    itrain = list(y_train.index)
    ifull = list(y.index)
    new_ifull = np.setxor1d(ifull,itrain)
    y = y.reindex(index=new_ifull)

    y_cv = create_training_set(y,numtype,prop_cv)
    icv = list(y_cv.index)
    ifull = list(y.index)
    new_ifull = np.setxor1d(ifull,icv)
    y = y.reindex(index=new_ifull)
    y_test = y.copy()

  else:
    prop_cv = len(y_train)*1./len(y_ref) # proportion of the training set wrt the whole set
    y_cv = create_training_set(y,numtype,prop_cv)
    icv = list(y_cv.index)
    ifull = list(y.index)
    new_ifull = np.setxor1d(ifull,icv)
    y = y.reindex(index=new_ifull)
    y_test = y.copy()

  return y_train, y_cv, y_test
# ================================================================

def plot_confusion_mat(cmat,l,set,method,ax=None):
  """
  Plots the confusion matrix
  """
  cmat = np.array(cmat,dtype=float)
  for i in range(cmat.shape[0]):
    cmat[i,:] = cmat[i,:]*100./np.sum(cmat[i,:])

  if not ax:
    fig = plt.figure()
    fig.set_facecolor('white')
    ax = fig.add_subplot(111)

  ax.matshow(cmat,cmap=plt.cm.gray_r)
  for i in range(cmat.shape[0]):
    for j in range(cmat.shape[0]):
      if cmat[j,i] >= np.max(cmat)/2. or cmat[j,i] > 50:
        col = 'w'
      else:
        col = 'k'
      if cmat.shape[0] <= 4:
        ax.text(i,j,"%.2f"%cmat[j,i],color=col)
      else:
        ax.text(i,j,"%d"%np.around(cmat[j,i]),color=col)
  ax.set_title('%s set - %s'%(set,method.upper()),size=10)
  ax.set_xlabel('Prediction')
  ax.set_ylabel('Observation')
  if len(l) <= 4:
    ax.set_xticklabels(['']+l)
    ax.set_yticklabels(['']+l)

# ================================================================
def dic_percent(cmat,types,verbose=False):

  NB_class = cmat.shape[0]
  NB_ev = np.sum(cmat)

  p = {}
  p['global'] = np.sum(np.diag(cmat))*1./NB_ev*100
  for i in range(NB_class):
    l_man = np.sum(cmat[i,:])
    l_auto = np.sum(cmat[:,i])
    p_cl = cmat[i,i]*1./l_man*100
    p[('%s'%types[i],i)] = round(p_cl,2)
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

  Returns an output dictionary with keys :  
  - label_test : classification predicted by SVM for the test set
  - label_train : classification predicted by SVM for the training set

  If proba is True, add the key 'probas' containing 
  the probability estimates for each element of the dataset

  If kernel is linear, add the key 'thetas' containing  
  the coefficients of the linear decision boundary

  If kernel is non linear, add the key "map" containing 
  the classification map.
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
    NB_class = len(types)
    for k in range(NB_class):
      output['probas'][types[k]] = probabilities[:,k]
  if kern == 'Lin':
    output['thetas'] = grid.best_estimator_.raw_coef_
  elif len(x_train.columns) == 2:
    pas = .01
    x_vec, y_vec = np.arange(-1,1,pas), np.arange(-1,1,pas)
    x_vec, y_vec = np.meshgrid(x_vec,y_vec)
    vec = np.c_[x_vec.ravel(),y_vec.ravel()]
    print vec.shape
    map = grid.best_estimator_.predict(np.c_[x_vec.ravel(),y_vec.ravel()])
    output['map'] = map.reshape(x_vec.shape)
  return output

# ================================================================

def implement_lr_sklearn(x_train,x_test,y_train,y_test):
  """
  Implements logistic regression from scikit learn package.
  """
  from LR_functions import normalize
  x_train, x_test = normalize(x_train,x_test)

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

  p_train, p_cv, p_test = opt.opdict['proportions']

  for b in range(boot):

    otimes = map(int,list(otimes_ref.values))
    otimes = np.array(otimes)

    x_test_ref = x_test_ref0.copy()
    y_test_ref = y_test_ref0.copy()

    print "\n\tONE BY ONE EXTRACTION ------ iteration %d"%b
    dic = {}

    inum = 0
    for n in range(len_numt):

      sub_dic={}

      ### Splitting of the whole set in training, CV and test sets ###
      y_train_ref, y_cv, y_test_ref = generate_datasets(opt.opdict['proportions'],opt.numt,y_test_ref)
      y_test_ref = pd.concat([y_cv,y_test_ref])
      i_train = y_train_ref.index
      i_cv = y_cv.index
      i_test = y_test_ref.index

      ### Defining the training set ###
      x_train = x_test_ref.reindex(index=y_train_ref.index)
      y_train_ref.index = range(y_train_ref.shape[0])
      x_train.index = range(x_train.shape[0])
      if inum == 0:
        list_i_train = [list(otimes[map(int,list(y_train_ref.index))])]
      else:
        list_i_train.append(list(otimes[map(int,list(y_train_ref.index))]))

      ### Defining the test set ###
      x_test = x_test_ref.reindex(index=y_test_ref.index)
      x_test.index = range(x_test.shape[0])
      y_test_ref.index = range(y_test_ref.shape[0])
      if inum == 0:
        list_i_test = [list(otimes[map(int,list(y_test_ref.index))])]
      else:
        list_i_test.append(list(otimes[map(int,list(y_test_ref.index))]))

      if x_train.shape[0] != y_train_ref.shape[0]:
        print "Training set: Incoherence in x and y dimensions"
        sys.exit()

      if x_test.shape[0] != y_test_ref.shape[0]:
        print "Test set: Incoherence in x and y dimensions"
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
        out = do_all_logistic_regression(x_test_ref0,y_test_ref0,i_train,i_cv,i_test,)
      elif method == 'svm':
        kern = 'NonLin'
        print "SVM\n"
        out = implement_svm(x_train,x_test,y_train,y_test,opt.types,opt.opdict,kern=kern)

      CLASS_test = out['label_test']
      CLASS_train = out['label_train']

      # TRAINING SET
      print "\t *TRAINING SET"
      y_train_np = y_train.NumType.values.ravel()  
      from sklearn.metrics import confusion_matrix
      cmat_train = confusion_matrix(y_train_np,CLASS_train)
      p_tr = dic_percent(cmat_train,[types[n],'Rest'],verbose=True)
      out['rate_train'] = p_tr
      print "   Global : %.2f%%"%p_tr['global']
      if opt.opdict['plot_confusion'] or opt.opdict['save_confusion']:
        plot_confusion_mat(cmat_train,opt.types,'Training',opt.opdict['method'].upper())
        if opt.opdict['save_confusion']:
          savefig = '%s/training_%s.png'%(opt.opdict['fig_path'],opt.opdict['result_file'])
          print "Confusion matrix saved in %s"%savefig
          plt.savefig(savefig)

      # TEST SET
      print "\t *TEST SET"
      y_test_np = y_test.NumType.values.ravel()
      cmat_test = confusion_matrix(y_test_np,CLASS_test)
      p_test = dic_percent(cmat_test,[types[n],'Rest'],verbose=True)
      out['rate_test'] = p_test
      print "   Global : %.2f%%"%p_test['global']
      if opt.opdict['plot_confusion'] or opt.opdict['save_confusion']:
        plot_confusion_mat(cmat_test,opt.types,'Test',opt.opdict['method'].upper())
        if opt.opdict['save_confusion']:
          savefig = '%s/test_%s.png'%(opt.opdict['fig_path'],opt.opdict['result_file'])
          print "Confusion matrix saved in %s"%savefig
          plt.savefig(savefig)
        if opt.opdict['plot_confusion']:
          plt.show()
        else:
          plt.close()

      # Fill the dictionary
      i_com = np.where((y_test.NumType.values.ravel()-CLASS_test)==0)[0]
      i_lr = np.where(CLASS_test==0)[0]
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
      sub_dic["rate_%s"%types[n]] = (out['rate_train'][('%s'%types[n], 0)], out['rate_test'][('%s'%types[n], 0)])
      sub_dic["rate_rest"] = (out['rate_train'][('Rest', 1)], out['rate_test'][('Rest', 1)])
      sub_dic["nb_manuals"] = ((types[n],len(y_test[y_test.NumType==0])),('Rest',len(y_test[y_test.NumType==1])))

      i_ok_test = i_test[np.where(CLASS_test!=0)[0]]
      i_ok_train = i_train[np.where(CLASS_train!=0)[0]]
      i_ok = np.concatenate([i_ok_test,i_ok_train])
      otimes = i_ok
      y_test_ref = y_test_ref0.reindex(index=map(int,list(i_ok)))

      dic[types[n]] = sub_dic
      inum = inum + 1

    dic['i_train'] = list_i_train
    dic['i_test'] = list_i_test
    DIC[b] = dic

  file = opt.opdict['result_path']
  print "One-by-One results stored in %s"%file
  write_binary_file(file,DIC)

  file = '%s/stats_OBO'%os.path.dirname(opt.opdict['result_path'])
  write_binary_file(file,EXT)

# ================================================================

def one_vs_all(opt,x_test_ref,y_test_ref,otimes_ref,boot=1,method='lr'):

  """
  Extract one class among the whole data.
  """

  from LR_functions import do_all_logistic_regression

  types = opt.types
  numt = opt.numt
  len_numt = len(numt)

  DIC = {}
  DIC['features'] = x_test_ref.columns
  for b in range(boot):

    print "\n\tONE VS ALL EXTRACTION ------ iteration %d"%b

    dic = {}
    otimes = map(str,list(otimes_ref.values))
    otimes = np.array(otimes)

    ### Splitting of the whole set in training, CV and test sets ###
    y_train, y_cv, y_test = generate_datasets(opt.opdict['proportions'],opt.numt,y_test_ref)
    i_train = y_train.index
    i_cv = y_cv.index
    i_test = y_test.index

    ### Defining the training set ###
    x_train = x_test_ref.reindex(index=y_train.index)
    y_train.index = range(y_train.shape[0])
    x_train.index = range(x_train.shape[0])
    dic["i_train"] = otimes[map(int,list(y_train.index))]

    ### Defining the test set ###
    x_test = x_test_ref.reindex(index=y_test.index)
    x_test.index = range(x_test.shape[0])
    y_test.index = range(y_test.shape[0])
    dic["i_test"] = otimes[map(int,list(y_test.index))]

    y_train_tir = y_train.copy()
    y_test_tir = y_test.copy()

    for n in range(len_numt):

      y_train[y_train_tir.NumType==n] = 0
      y_test[y_test_tir.NumType==n] = 0
      y_train[y_train_tir.NumType!=n] = 1
      y_test[y_test_tir.NumType!=n] = 1

      print y_train.shape[0], y_test.shape[0]

      print "----------- %s vs all -----------"%types[n]
      print_type = [types[n],'All']

      if method == 'lr':
        print "Logistic Regression\n"
        i_train = y_train.index
        i_cv = y_cv.index
        i_test = y_test.index
        out = do_all_logistic_regression(x_test_ref,y_test_ref,i_train,i_cv,i_test)
      elif method == 'svm':
        kern = 'NonLin'
        print "SVM\n"
        out = implement_svm(x_train,x_test,y_train,y_test,opt.types,opt.opdict,kern=kern)

      CLASS_test = out['label_test']
      CLASS_train = out['label_train']

      # TRAINING SET
      print "\t *TRAINING SET"
      y_train_np = y_train.NumType.values.ravel()  
      from sklearn.metrics import confusion_matrix
      cmat_train = confusion_matrix(y_train_np,CLASS_train)
      p_tr = dic_percent(cmat_train,[types[n],'Rest'],verbose=True)
      out['rate_train'] = p_tr
      print "   Global : %.2f%%"%p_tr['global']
      if opt.opdict['plot_confusion'] or opt.opdict['save_confusion']:
        plot_confusion_mat(cmat_train,opt.types,'Training',opt.opdict['method'].upper())
        if opt.opdict['save_confusion']:
          savefig = '%s/training_%s.png'%(opt.opdict['fig_path'],opt.opdict['result_file'])
          print "Confusion matrix saved in %s"%savefig
          plt.savefig(savefig)

      # TEST SET
      print "\t *TEST SET"
      y_test_np = y_test.NumType.values.ravel()
      cmat_test = confusion_matrix(y_test_np,CLASS_test)
      p_test = dic_percent(cmat_test,[types[n],'Rest'],verbose=True)
      out['rate_test'] = p_test
      print "   Global : %.2f%%"%p_test['global']
      if opt.opdict['plot_confusion'] or opt.opdict['save_confusion']:
        plot_confusion_mat(cmat_test,opt.types,'Test',opt.opdict['method'].upper())
        if opt.opdict['save_confusion']:
          savefig = '%s/test_%s.png'%(opt.opdict['fig_path'],opt.opdict['result_file'])
          print "Confusion matrix saved in %s"%savefig
          plt.savefig(savefig)
        if opt.opdict['plot_confusion']:
          plt.show()
        else:
          plt.close()


      # Fill the dictionary
      sub_dic={}
      i_com = np.where((y_test.NumType.values.ravel()-CLASS_test)==0)[0]
      i_lr = np.where(CLASS_test==0)[0]
      i_ok_class = np.intersect1d(i_com,i_lr) # events classified in the class of interest by the LR and identical to the manual classification
      sub_dic["nb"] = len(i_lr) # total number of events classified in the class of interest
      sub_dic["nb_common"] = len(i_ok_class) # total number of well classified events
      sub_dic["index_ok"] = otimes[i_ok_class] # index of well classified events
      sub_dic["nb_other"],sub_dic["i_other"] = [],[]
      for k in range(len_numt):
        if k != n:
          i_other_man = list(y_test_tir[y_test_tir.NumType==k].index)
          ii = np.intersect1d(i_lr,i_other_man)
          sub_dic["nb_other"].append((types[k],len(ii))) # number of events belonging to another class
          sub_dic["i_other"].append((types[k],otimes[ii])) # index of events belonging to another class
      sub_dic["rate_%s"%types[n]] = (out['rate_train'][('%s'%types[n], 0)], out['rate_test'][('%s'%types[n], 0)]) # % success rate of the extracted class
      sub_dic["rate_rest"] = (out['rate_train'][('Rest', 1)], out['rate_test'][('Rest', 1)]) # % success rate of the rest
      sub_dic["nb_manuals"] = ((types[n],len(y_test[y_test.NumType==0])),('Rest',len(y_test[y_test.NumType==1])))
      dic[types[n]] = sub_dic

    DIC[b] = dic

  file = opt.opdict['result_path']
  print "One-vs-All results stored in %s"%file
  write_binary_file(file,DIC)
