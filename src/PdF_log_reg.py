#!/usr/bin/env python
# encoding: utf-8

import os,sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io.matlab import mio
from LR_functions import *
from bayes_gaussian import data_to_eigen_projection_dD

# --------------------------------------------

def clean_data(a,x,y=None):
  for i in range(len(a)):
    for key in sorted(x):
      x[key] = np.delete(x[key],a[i])
    if y:
      for k in range(1,len(y)+1):
        y[k] = np.delete(y[k],a[i])
  if y:
    return x,y
  else:
    return x

# --------------------------------------------

def read_training_set(matfile):
  """
  Reads training set
  """
  mat = mio.loadmat(matfile)

  nb_EB = len(mat['AsDecEB'].ravel())
  nb_VT = len(mat['AsDecVT'].ravel())
  print "Training set -- # EB = %d, # VT = %d\n"%(nb_EB,nb_VT)

  train = {}
  train['Kurto'] = np.log(np.concatenate((mat['KurtoEB'].ravel(),mat['KurtoVT'].ravel())))
  train['RappMaxMean'] = np.log(np.concatenate((mat['RappMaxMeanEB'].ravel(),mat['RappMaxMeanVT'].ravel())))
  train['AsDec'] = np.log(np.concatenate((mat['AsDecEB'].ravel(),mat['AsDecVT'].ravel())))
  train['Dur'] = np.concatenate((mat['DurEB'].ravel(),mat['DurVT'].ravel()))
  train['Ene'] = np.log(np.concatenate((mat['EneFFTeB'].ravel(),mat['EneFFTvT'].ravel())))

  # y=0 if EB ; y=1 if VT
  y = {}
  y[1] = np.concatenate((np.zeros(nb_EB,dtype=int),np.ones(nb_VT,dtype=int)))

  # Remove bad data from training set
  a = np.where(train['Dur']<0)
  if a:
    train,y = clean_data(a,train,y)
  b = np.where(train['AsDec']>=50)
  if b:
    train,y = clean_data(b,train,y)

  train = pd.DataFrame(train)
  y = pd.DataFrame(y)

  return train,y

# --------------------------------------------

def read_test_set(matfile):
  """
  Reads test set and OVPF classification
  """
  mat = mio.loadmat(matfile)

  test = {}
  test['Kurto'] = mat['KurtoVal'].ravel()
  test['RappMaxMean'] = mat['RappMaxMeanVal'].ravel()
  test['AsDec'] = mat['AsDecVal'].ravel()
  test['Dur'] = mat['DurationVal'].ravel()
  test['Ene'] = mat['EneHighFreqVal'].ravel()

  # Remove bad data from test set
  ibad = []
  a = np.where(test['Dur']<0)
  if a[0].any():
    x_data = clean_data(a,test)
    ibad.append(a[0])
  b = np.where(test['AsDec']>=50)
  if b[0].any():
    x_data = clean_data(b,test)
    ibad.append(b[0])
  ibad = np.array(ibad).ravel()

  class_obs = mat['OVPFID'].ravel() # 1 for EB / 2 for VT
  valid = mat['CLASSVALID'].ravel() # 1 if same as OVPF; else 0
  # Remove bad data
  if ibad.any():
    class_obs = np.delete(class_obs,ibad)
    valid = np.delete(valid,ibad)
  
  class_obs[np.where(class_obs==1)[0]] = 0 # 0 for EB
  class_obs[np.where(class_obs!=0)[0]] = 1 # 1 for VT

  test = pd.DataFrame(test)
  class_obs = pd.DataFrame(class_obs)
  valid = pd.DataFrame(valid)

  return test, class_obs,valid

# --------------------------------------------

def make_uncorrelated_data(data,testdata,f,new_f):
    """
    Takes the raw data and test data and does required PCA
    Modified from A. Maggi
    """

    # Combine Kurto and RappMaxMean
    f_test = [f[0], f[1]]

    # set up EB and VT matrices (nsamples x nfeatures)
    v = [data[name] for name in f]
    v_test = [testdata[name] for name in f_test]

    x = np.vstack(tuple(v)).T
    x_test = np.vstack(tuple(v_test)).T
    (n,d) = x.shape
    (n_test,d) = x_test.shape

    # stack them together
    x_all = np.vstack((x,x_test))
    e_values, e_vectors, x_new = data_to_eigen_projection_dD(x_all)

    # put the principle component in a new observable
    data[new_f] = x_new[0:n,0]
    testdata[new_f] = x_new[n:,0]

    del data[f[0]]
    del data[f[1]]
    del testdata[f[0]]
    del testdata[f[1]]

    return data, testdata

# -------------------------------------------

def hypothesis_function(mins,maxs,theta):
  x_synth=pd.DataFrame(np.arange(np.min(mins),np.max(maxs),0.01))

  deg = len(theta)-1
  x_synth_deg = poly(deg,x_synth)
  mat = features_mat(x_synth_deg)

  hyp = g(np.dot(theta.reshape(1,len(theta)),mat))[0]

  return x_synth.values[:,0],hyp

# --------------------------------------------

def plot_histo(x,x_data,nb_EB,nb_VT,savefig=False):
  """
  Plots histograms
  """
  for key in sorted(x):
    fig=plt.figure()
    fig.set_facecolor('white')
    plt.hist([x[key][:nb_EB],x[key][nb_EB:],x_data[key]],50,normed=1,align='left',histtype='barstacked',label=['EB','VT','TestSet'])
    plt.title(key)
    plt.legend(loc=2,prop={'size':'medium'})
    if savefig:
      plt.savefig('../results/Piton/histo_%s.png'%key)

# --------------------------------------------

def plot_one_feature(x,y,syn,hyp,x_correct,x_bad,text,savefig=False):
  fig=plt.figure()
  fig.set_facecolor('white')
  x_eb = x[y==0].values[:,0]
  x_vt = x[y==1].values[:,0]
  nn, b, p = plt.hist([x_eb,x_vt],25,normed=True,histtype='stepfilled',alpha=.2,color=('b','g'),label=['EB','VT'])
  norm=np.mean([np.max(nn[0]),np.max(nn[1])])
  if x_correct and x_bad:
    nn, b, p = plt.hist([x_correct,x_bad],25,normed=True,color=('k','r'),histtype='step',fill=False,ls='dashed',lw=2,label=['Test Set'])
  plt.plot(syn,norm*hyp,'y-',lw=2,label='hypothesis')
  plt.legend()
  if text:
    plt.figtext(0.15,0.85,"%.2f %% VT"%text[0],color='g')
    plt.figtext(0.15,0.8,"%.2f %% EB"%text[1],color='b')
    plt.figtext(0.15,0.75,"%.2f %% test set"%text[2])
    plt.figtext(0.15,0.7,"%.2f %% test set"%text[3],color='r')
  plt.xlabel(x.columns[0])
  plt.title('Logistic regression - 1 feature')
  if savefig:
    plt.savefig('../results/Piton/feat1_%s.png'%x.columns[0],format='png')

# --------------------------------------------

def plot_features(x_train,y_train,x_all,y_all,x_bad,text,savefig=False):
  """
  Plots each feature vs each other
  """
  n = x_train.shape[1]
  for ikey,key in enumerate(x_train.columns):
    for k in x_train.columns[ikey+1:]:
      fig = plt.figure()
      fig.set_facecolor('white')
      plt.scatter(x_all[key],x_all[k],c=y_all,cmap=plt.cm.gray,alpha=0.2)
      plt.scatter(x_train[key],x_train[k],c=y_train,cmap=plt.cm.winter,alpha=0.5)
      if key in x_bad.columns:
        plt.scatter(x_bad[key],x_bad[k],c='r',alpha=0.2)
      if text:
        plt.figtext(0.7,0.85,"%.2f %% VT"%text[0],color='g')
        plt.figtext(0.7,0.8,"%.2f %% EB"%text[1],color='b')
        plt.figtext(0.7,0.75,"%.2f %% test set"%text[2])
        plt.figtext(0.7,0.7,"%.2f %% test set"%text[3],color='r')
      plt.xlabel(key)
      plt.ylabel(k)
      plt.title('Logistic regression - %d features'%n)
      if savefig:
        plt.savefig('../results/feat%d_%s%s.png'%(len(x_train),key,k),format='png')
        plt.close()
      #plt.savefig('../../Desktop/features/%s_vs_%s.png'%(key,k))
      #plt.close()

# --------------------------------------------

def do_logistic_regression_PdF(list_features,hist=False,plot=False,save=False,plot_sep=False):
  """
  Implements logistic regression on PdF data
  list_features = list of features we want to work with
  Options:
    - hist=True: plots histograms of each feature for both training and test sets
    - plot=True: plots training and test sets, shows bad classification, displays statistics
    - save=True: saves figures
    - plot_sep=True: plots the decision boundary
  """
  training_set, y = read_training_set('../data/Piton/TrainingSet_2.mat')
  #training_set, ovpf, valid = read_test_set('../DonneeAutoID_PdF/TestDataSet_2.mat')
  
  test_set, ovpf, valid = read_test_set('../data/Piton/TestDataSet_2.mat')

  x = training_set.reindex(columns=list_features)
  x_data = test_set.reindex(columns=list_features)
 
  # Combine Kurto and RappMaxMean into one feature (Krapp)
  if ('Kurto' and 'RappMaxMean') in list_features:
    x, x_data = make_uncorrelated_data(x,x_data,['Kurto','RappMaxMean'],'KRapp')

  #plot_pdf_feat(x_data,ovpf,['EB','VT'])
  #plt.show()
  #sys.exit()

  # Check if it's OK
  if x.shape[0] != y.shape[0] or x_data.shape[0] != ovpf.shape[0]:
    print "Warning !! x and y are not the same length"
    print "Training set: x: %d examples / y: %d examples"%(x.shape[0],y.shape[0])
    print "Test set: x: %d examples / y: %d examples"%(x_data.shape[0],ovpf.shape[0])
    sys.exit()

  LR_class_train,theta,LR_class,wtr = do_all_logistic_regression(x,y,x_data,y_testset=ovpf,output=True)

  # More detailed analysis of the results
  # EB = 0 ------------ VT = 1
  # Training set
  nb_tot_train = y.shape[0]
  nb_eb_train = len(np.where(y[1]==0)[0])
  nb_vt_train = len(np.where(y[1]==1)[0])
  LR_nb_eb_train = len(np.where(LR_class_train==0)[0])
  LR_nb_vt_train = len(np.where(LR_class_train==1)[0])

  if nb_tot_train != nb_eb_train+nb_vt_train or nb_tot_train != LR_nb_eb_train+LR_nb_vt_train :
    print "Something's wrong for the training set!"
    sys.exit()
  else:
    nb_eb_bad,nb_vt_bad,nb_vt_ok = comparison(LR_class_train,y[1])
    nb_ok_train = len(LR_class_train)-(nb_eb_bad+nb_vt_bad)
    print "\nPiton de la Fournaise"
    print "Training set"
    print "Training set recovery: %.2f %%"%(np.float(nb_ok_train)*100/nb_tot_train)
    print "Training set - VT: %.2f %%"%(100-np.float(nb_vt_bad)*100/nb_vt_train)
    print "Training set - EB: %.2f %%"%(100-np.float(nb_eb_bad)*100/nb_eb_train)
    text=[100-np.float(nb_vt_bad)*100/nb_vt_train,100-np.float(nb_eb_bad)*100/nb_eb_train]

  # Test set
  nb_eb_ovpf = len(np.where(ovpf[0]==0)[0])
  nb_vt_ovpf = len(np.where(ovpf[0]==1)[0])
  nb_tot = nb_vt_ovpf+nb_eb_ovpf
  LR_nb_eb = len(np.where(LR_class==0)[0])
  LR_nb_vt = len(np.where(LR_class==1)[0])

  if nb_tot != LR_nb_eb+LR_nb_vt:
    print "Something's wrong for the test set!"
    sys.exit()
  else:
    nb_bad_eb,nb_bad_vt,nb_good_vt = comparison(LR_class,ovpf[0])
    nb_ok = len(LR_class)-(nb_bad_eb+nb_bad_vt)

    x_bad = bad_class(x_data,np.where((LR_class-ovpf[0])!=0)[0])
    x_correct = bad_class(x_data,np.where((LR_class-ovpf[0])==0)[0])

    print "\nTest Set"
    print "Classification OK for %d events out of %d (%.2f %%)"%(nb_ok,nb_tot,np.float(nb_ok)*100/nb_tot)
    print "OVPF classification -- Nb VT : %d, Nb EB : %d"%(nb_vt_ovpf,nb_eb_ovpf)
    print "Logistic classification -- Nb VT : %d, Nb EB : %d"%(LR_nb_vt,LR_nb_eb)
    #print "Correct classification as VT: %.2f %%"%(100-np.float(nb_bad_vt)*100/LR_nb_vt)
    print "Correct classification as EB: %.2f %%"%(100-np.float(nb_bad_eb)*100/LR_nb_eb)
    text.append(np.float(nb_ok)*100/nb_tot)
    text.append(100-np.float(nb_ok)*100/nb_tot)

  if hist:
    plot_histo(x,x_data,nb_eb_train,nb_vt_train,savefig=save)

  if plot:
    n=x.shape[1] # Number of features
    if n == 1 and len(theta[1])-1 == n:
      mins=[x.min(),x_data.min()]
      maxs=[x.max(),x_data.max()]
      syn,hyp = hypothesis_function(mins,maxs,theta[1])
      plot_one_feature(x,nb_eb_train,nb_vt_train,syn,hyp,x_correct,x_bad,text,savefig=save)
    if n >= 2:
      plot_features(x,LR_class_train,x_data,ovpf,x_bad,text,savefig=save)

  if plot_sep and (n == 2 or n == 3):
    if n == 2 and len(theta[1])-1 == n:
      x, x_data = normalize(x,x_data)
      plot_db(x,y[1],theta[1],title='Training set')
      plot_db(x_data,ovpf,theta[1],title='Test set')
    if n == 3 and len(theta[1])-1 == n:
      x,x_data = normalize(x,x_data)
      plot_db_3d(x,y[1],theta[1],title='Training set')
      plot_db_3d(x_data,ovpf,theta[1],title='Test set')

  # Comparison with fuzzy logic results
  print "\nFuzzy logic : %.2f %% (%d)"%(np.float(len(np.where(valid==1)[0]))*100/valid.shape[0],len(np.where(valid==1)[0]))

  ydiff = LR_class - ovpf[0]
  same_class = ydiff.copy()
  same_class[np.where(ydiff==0)[0]]=1
  same_class[np.where(ydiff!=0)[0]]=0
  nb_bad_FL,nb_bad_LR,nb_ok_pos = comparison(same_class,valid[0])
  nb_class_ok = valid.shape[0] - (nb_bad_FL+nb_bad_LR)

  print "%% of events correcly classified by fuzzy logic and logistic regression: %.2f %% (%d)"%(float(nb_class_ok)*100/nb_tot,nb_class_ok)
  print "%% of events correcly classified by fuzzy logic but not by logistic regression: %.2f %% (%d)"%(float(nb_bad_LR)*100/nb_tot,nb_bad_LR)
  print "%% of events correcly classified by logistic regression but not by fuzzy logic: %.2f %% (%d)"%(float(nb_bad_FL)*100/nb_tot,nb_bad_FL)

  plt.show()

# --------------------------------------------

def statistics_LR(filename,Hash=False):

  """
  Plot of % training set, % test set, and % test vs % training for different 
  parameters and for several training set repartition.
  Similar to HT_logreg function in Fingerprint_LR.py
  """ 

  import cPickle
  from Fingerprint_LR import plot_results,read_tables,charge_class
 
  if not os.path.exists(filename):
    training_set, y = read_training_set('../DonneeAutoID_PdF/TrainingSet_2.mat')
  
    test_set, ovpf, valid = read_test_set('../DonneeAutoID_PdF/TestDataSet_2.mat')

    if Hash:
      dirname = '/home/nadege/Fingerprint/Test4'
      x_train = read_tables('%s/train_hash_tables_1'%dirname).transpose()
      y_train = read_tables('%s/train_types'%dirname)

      y_temp_1 = list(y_train[y_train[0]==0].index)
      y_temp_2 = list(y_train[y_train[0]==1].index)
      y_temp = y_temp_1+y_temp_2
      y_temp.remove(103)
      y_train = y_train.reindex(index=y_temp)
      x_train = x_train.reindex(index=y_train.index)
      x_train.index = range(len(x_train))
      y_train.index = range(len(y_train))

      if len(training_set) != len(x_train):
        print "Warning !! Check lengths - training sets", len(training_set), len(x_train)
        sys.exit()
      training_set = pd.merge(training_set,x_train,on=x_train.index)
      training_set = training_set.reindex(columns=training_set.columns[1:])

      x_test = read_tables('%s/hash_tables_1'%dirname).transpose()
      for im in [6385,6153,3059]:
        xindex = x_test.index[x_test.index>im]
        xindex = np.array(xindex)-1
        x_test.index = list(x_test.index[x_test.index<im]) + list(xindex)

      y_test = charge_class()
      y_test = pd.DataFrame(y_test)
      y_test = y_test.reindex(index=x_test.index)
      test_set = test_set.reindex(index=y_test.index)
      ovpf = ovpf.reindex(index=y_test.index)

      if len(test_set) != len(x_test):
        print "Warning !! Check lengths - test sets", len(test_set), len(x_test)
        sys.exit()
      test_set = pd.merge(test_set,x_test,on=test_set.index)
      test_set = test_set.reindex(columns=test_set.columns[1:])
      test_set.index = range(len(test_set))
      ovpf.index = test_set.index

    list_features = [['Kurto','RappMaxMean'],['AsDec'],['Dur'],['Ene'],['Kurto','RappMaxMean','AsDec'],['Kurto','RappMaxMean','Dur'],['Kurto','RappMaxMean','Ene'],['AsDec','Dur'],['AsDec','Ene'],['Dur','Ene'],['Kurto','RappMaxMean','AsDec','Dur'],['Kurto','RappMaxMean','AsDec','Ene'],['AsDec','Dur','Ene'],['Kurto','RappMaxMean','AsDec','Dur','Ene']]

    dic = {}
    marker = 1
    if not os.path.exists('%s/permut_clement'%os.path.dirname(filename)):
      marker = 0
      permut = []
      NbPerm = 10
    else:
      with open('%s/permut_clement'%os.path.dirname(filename),'rb') as file:
        my_depickler = cPickle.Unpickler(file)
        permut = my_depickler.load()
        file.close()
      NbPerm = len(permut)

    for ifeat,features in enumerate(list_features):

      print "********",features
      if Hash:
        features = features + range(x_test.shape[1])

      dic[ifeat] = []
      x = training_set.reindex(columns=features)
      x_data = test_set.reindex(columns=features)
 
      # Combine Kurto and RappMaxMean into one feature (Krapp)
      if ('Kurto' and 'RappMaxMean') in features:
        x, x_data = make_uncorrelated_data(x,x_data,['Kurto','RappMaxMean'],'KRapp')

      # Check if it's OK
      if x.shape[0] != y.shape[0] or x_data.shape[0] != ovpf.shape[0]:
        print "Warning !! x and y are not the same length"
        print "Training set: x: %d examples / y: %d examples"%(x.shape[0],y.shape[0])
        print "Test set: x: %d examples / y: %d examples"%(x_data.shape[0],ovpf.shape[0])
        sys.exit()

      for j in range(NbPerm):

        print "\n\tTirage %d"%j
        if ifeat == 0 and marker == 0:
          LR_class_train,theta,LR_class,p,wtr = do_all_logistic_regression(x,y,x_data,y_testset=ovpf,output=True,perc=True)
          permut.append(wtr)

        else:
          LR_class_train,theta,LR_class,p,wtr = do_all_logistic_regression(x,y,x_data,y_testset=ovpf,output=True,wtr=permut[j],perc=True)

        dic[ifeat].append(p)

    with open(filename,'wb') as file:
       my_pickler = cPickle.Pickler(file)
       my_pickler.dump(dic)
       file.close()

    with open('%s/permut_clement'%os.path.dirname(filename),'wb') as file:
      my_pickler = cPickle.Pickler(file)
      my_pickler.dump(permut)
      file.close()

  plot_results(filename)

# --------------------------------------------

if __name__ == '__main__' :
  
  #list_features=['Kurto','RappMaxMean','AsDec','Dur','Ene']
  #if list_features:
  #  do_logistic_regression_PdF(list_features, hist=False, plot=True, save=False, plot_sep=True)

  statistics_LR('../results/Piton/StatsPdF_hash_results',Hash=True)
