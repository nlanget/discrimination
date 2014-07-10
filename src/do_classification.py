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
        import cPickle
        with open(opt.opdict['train_file'],'rb') as file:
          my_depickler = cPickle.Unpickler(file)
          TRAIN_Y = my_depickler.load()
          file.close()
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
        savefile = '%s/1B1_%s_%s-red'%(opt.opdict['outdir'],opt.opdict['feat_filename'].split('.')[0],opt.trad[isc][0])
        one_by_one(x_test,y_test,opt.types,opt.numt,set['Otime'],savefile,boot=10,method='svm')
        continue

      elif opt.opdict['method'] == 'ova':
        print "********** EXTRACTION 1-VS-ALL **********"
        from extraction import one_vs_all
        savefile = '%s/OVA_%s_%s-red'%(opt.opdict['outdir'],opt.opdict['feat_filename'].split('.')[0],opt.trad[isc][0])
        one_vs_all(x_test,y_test,opt.types,opt.numt,set['Otime'],savefile,boot=10,method='svm')
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

  import cPickle
  with open(opt.opdict['result_path'],'w') as file:
    my_pickler = cPickle.Pickler(file)
    my_pickler.dump(dic_results)
    file.close()

  if not os.path.exists(opt.opdict['train_file']) and opt.opdict['boot'] > 1:
    with open(opt.opdict['train_file'],'w') as file:
      my_pickler = cPickle.Pickler(file)
      my_pickler.dump(TRAIN_Y)
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
    if key == 'features':
      continue
    list_ev = list_ev + list(dic[key][0]['list_ev'])
  list_ev = np.array(list_ev)
  list_ev_all = np.unique(list_ev)

  df = pd.DataFrame(index=list_ev_all,columns=sorted(dic),dtype=str)
  for key in sorted(dic):
    if key == 'features':
      continue
    for iev,event in enumerate(dic[key][0]['list_ev']):
      df[key][event] = dic[key][0]['classification'][iev]

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
  print manual
  print auto

  a = pd.read_csv(auto,index_col=False)
  a = a.reindex(columns=['Type'])
  a.Type[a.Type=='LowFrequency'] = 'LF'

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
  print "%% of well classified events : %.2f"%(len(sim)*100./N)
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
# ================================================================
def plot_confusion_only(opt):
  """
  Reads the result file and plots the corresponding confusion matrix.
  """

  opt.tri()

  manual = opt.opdict['label_filename']
  auto = opt.opdict['class_auto_path']

  print auto
  print manual

  a = pd.read_csv(auto,index_col=False)
  a = a.reindex(columns=['Type'])
  a.Type[a.Type=='LowFrequency'] = 'LF'

  m = pd.read_csv(manual,index_col=False)
  m = m.reindex(columns=['Type'],index=a.index)

  m = m.dropna(how='any')
  a = a.reindex(index=m.index)

  for i in range(len(m.values)):
    m.values[i][0] = m.values[i][0].replace(" ","")

  opt.classname2number()
  for i in opt.numt:
    m['Type'][m.Type==opt.types[i]] = i
    a['Type'][a.Type==opt.types[i]] = i
  a = a[a.Type!='?']
  m = m.reindex(index=a.index)

  confusion(m,a.values[:,0],opt.types,'test',opt.opdict['method'],plot=True)
  if opt.opdict['save_confusion']:
    plt.savefig('%s/figures/test_%s.png'%(opt.opdict['outdir'],opt.opdict['result_file'][8:]))
  plt.show()


# ================================================================
def stats(opt):

  filename = opt.opdict['feat_filepath']
  df = pd.read_csv(filename,index_col=False)

  dic = {}
  for sta in opt.opdict['stations']:
    dic[sta] = 0

  for key in df.index:
    stakey = key.split(',')[1].replace(" ","")
    stakey = stakey.replace("'","")
    if key.split(',')[2] == " 'Z')":
      dic[stakey] = dic[stakey]+1
  print dic
  sys.exit()

# ================================================================
def plot_test_vs_train():
  """
  For multiple training set draws
  """
  import cPickle
  path = '../results/Ijen'
  filenames = ['LR/results_ijen_redac_lr','LR/results_ijen_redac_lr-reclass','LR/results_ijen_redac_lr-2c','LR/results_ijen_redac_lr-red','SVM/results_ijen_redac_svm','SVM/results_ijen_redac_svm-reclass','SVM/results_ijen_redac_svm-2c','SVM/results_ijen_redac_svm-red','SVM/results_ijen_redac_svm-red2','SVM/results_ijen_redac_svm-reclass-red','SVM/results_ijen_redac_svm_lin']
  labels = ['LR','LR reclass','LR VB+Tr','LR 8 feat','SVM','SVM reclass','SVM VB+Tr','SVM 30 feat','SVM 8 feat','SVM reclass feat','SVM lin']

  fig = plt.figure()
  fig.set_facecolor('white')
  colors = ['b','c','m','y','g','r','b','c','m','y','g','r']
  markers = ['*','o','h','v','d','s','o','v','s','*','h','d']
  k = 0
  for filename in filenames:
    filename = os.path.join(path,filename)
    with open(filename,'r') as file:
      my_depickler = cPickle.Unpickler(file)
      dic = my_depickler.load()
      file.close()

    DIC = dic[sorted(dic)[0]]
 
    p_tr,p_test = [],[]
    for i in sorted(DIC):
      p_tr.append(DIC[i]['%'][0])
      p_test.append(DIC[i]['%'][1])
    plt.plot(p_tr,p_test,marker=markers[k],color=colors[k],lw=0,label=labels[k])
    k = k+1
  plt.legend(numpoints=1,loc='upper left')
  plt.plot([0,100],[0,100],'k--')
  plt.xlim([60,100])
  plt.ylim([60,100])
  plt.xlabel('% training set')
  plt.ylabel('% test set')
  plt.savefig('../results/Ijen/figures/svm_vs_lr.png')
  plt.show()


if __name__ == '__main__':
  plot_test_vs_train()
