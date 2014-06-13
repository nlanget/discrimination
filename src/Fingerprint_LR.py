import os, sys,glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cPickle


def read_tables(filename):

  with open(filename,'rb') as test:
    my_depickler = cPickle.Unpickler(test)
    dic = my_depickler.load()
    test.close()

  return pd.DataFrame(dic)


def charge_class():
  from scipy.io.matlab import mio
  ovpf = mio.loadmat('../data/TestDataSet_2.mat')['OVPFID'].ravel()
  ovpf[ovpf==1] = 0  # 0 for EB
  ovpf[ovpf!=0] = 1  # 1 for VT
  return ovpf


def HT_histo(x,y):

  list_vt = x.index[np.where(y.values==1)[0]]
  list_eb = x.index[np.where(y.values==0)[0]]
  y_vt = y.reindex(index=list_vt)
  y_eb = y.reindex(index=list_eb)
  for i in x.columns:
    a = x.reindex(columns=[i],index=y_vt.index).index
    df_vt = x.reindex(columns=[i],index=y_vt.index).dropna().values
    b = x.reindex(columns=[i],index=y_vt.index).dropna().index
    c = np.setxor1d(np.array(a),np.array(b))
    df_eb = x.reindex(columns=[i],index=y_eb.index).dropna().values

    fig = plt.figure()
    fig.set_facecolor('white')
    plt.hist([df_vt,df_eb],label=['VT','EB'])
    plt.title('Hash table %d'%i)
    plt.ylabel('Number of items')
    plt.legend()
    plt.show()


def trade_off(filename):
  from mpl_toolkits.mplot3d import Axes3D
  path = os.path.dirname(filename)
  with open(filename,'rb') as test:
    my_depickler = cPickle.Unpickler(test)
    dico = my_depickler.load()
    test.close()

  r = [32,64,128]
  p = [100,500,1000,5000]
  rplot = np.linspace(0,len(r)-1,len(r))
  pplot = np.linspace(0,len(p)-1,len(p))
  rplot,pplot = np.meshgrid(rplot,pplot)

  fig = plt.figure()
  fig.set_facecolor('white')
  ax = fig.add_subplot(111, projection='3d')
  maxi,mini = np.array([]),np.array([])
  for indr,ir in enumerate(r):
    for indp,ip in enumerate(p):
      key = '%d_%d'%(ir,ip)
      if key in sorted(dico):
        a = np.array([])
        for ii,ij in dico[key]:
          a = np.append(a,ii)
        maxi = np.append(maxi,np.max(a))
        mini = np.append(mini,np.min(a))
        ax.scatter([indr]*len(a),[indp]*len(a),a,color='k')
      else:
        maxi = np.append(maxi,80)
        mini = np.append(mini,80)
  maxi = np.reshape(maxi,(len(r),len(p)))
  mini = np.reshape(mini,(len(r),len(p)))
  ax.plot_surface(rplot,pplot,maxi.T,alpha=.2,cstride=1)
  ax.plot_surface(rplot,pplot,mini.T,alpha=.2,cstride=1,color='r')

  ax.set_xlabel('Resolution')
  ax.set_ylabel('Number of permutations')
  ax.set_zlabel('% training set')
  ax.set_xlim([0,len(r)-1])
  ax.set_ylim([0,len(p)-1])
  ax.set_xticks(range(len(r)))
  ax.set_yticks(range(len(p)))
  ax.set_xticklabels(r)
  ax.set_yticklabels(p)
  ax.set_title('Trade-off between the resolution and the number of permutations')
  plt.show()


def plot_results(filename):
  path = os.path.dirname(filename)
  with open(filename,'rb') as test:
    my_depickler = cPickle.Unpickler(test)
    dico = my_depickler.load()
    test.close()

  colors = ['k','b','r','g','y','c','m','c','y','g','r','b','k','m']*2
  fig = plt.figure()
  fig.set_facecolor('white')
  for ind,i in enumerate(sorted(dico)):
    a = []
    for ii,ij in dico[i]:
      a.append(ii)
    plt.plot([i]*len(a),a,'*-',color=colors[ind])
  plt.xlabel("Different hash functions")
  plt.ylabel("% training set")
  #plt.xlim([-.01,1.01])
  plt.xlim([-1,14])
  plt.ylim([0,100])
  plt.savefig('%s/fig1.png'%path)

  fig = plt.figure()
  fig.set_facecolor('white')
  for ind,i in enumerate(sorted(dico)):
    a = []
    for ii,ij in dico[i]:
      a.append(ij)
    plt.plot([i]*len(a),a,'*-',color=colors[ind])
  plt.xlabel("Different hash functions")
  plt.ylabel("% test set")
  #plt.xlim([-.01,1.01])
  plt.xlim([-1,14])
  plt.ylim([0,100])
  plt.savefig('%s/fig2.png'%path)

  fig = plt.figure()
  fig.set_facecolor('white')
  markers = ['*','o','h','v','d','s']*3
  for ind,i in enumerate(sorted(dico)):
    a,b = [],[]
    for ii,ij in dico[i]:
      a.append(ii)
      b.append(ij)
    plt.plot(a,b,marker=markers[ind],color=colors[ind],lw=0)
  plt.plot([0,100],[0,100],'k--')
  plt.legend(sorted(dico),numpoints=1,loc='upper left')
  plt.xlabel("% training set")
  plt.ylabel("% test set")
  plt.xlim([0,100])
  plt.ylim([0,100])
  plt.savefig('%s/fig3.png'%path)
  plt.show()


def HT_logreg(path,new=False,plot=False,save=False):
  """
  if new = False: open the permutation file
  """

  from logistic_reg import *
  dic = {}
  permut = []
  NbPerm = 10

  filename = '%s/results'%path
  if new == False:
    with open('train_permut','rb') as file:
      my_depickler = cPickle.Unpickler(file)
      permut = my_depickler.load()
      file.close()
    NbPerm = len(permut)

  if os.path.exists(filename):
    with open(filename,'rb') as file:
      my_depickler = cPickle.Unpickler(file)
      dic = my_depickler.load()
      file.close()

  param = ['32_100','32_500','32_1000','32_5000','64_100','64_500','64_1000','64_5000','128_100','128_500','128_1000']
  for iq,q in enumerate(param):
    print "***** %s *****"%q
    dic[q] = []

    x_train = read_tables('%s/train_hash_tables_%s'%(path,q)).transpose()
    #x_train = read_tables('%s/train_hash_tables'%path).transpose()
    y_train = read_tables('%s/train_types'%path)
    y_train = y_train.reindex(index=x_train.index)
    #HT_histo(x_train,y_train)

    x_test = read_tables('%s/hash_tables_%s'%(path,q)).transpose()
    #x_test = read_tables('%s/hash_tables'%path).transpose()
    y_test = charge_class()
    y_test = pd.DataFrame(y_test[list(x_test.index)])
    #HT_histo(x_test,y_test)

    if (x_train.shape[0] != y_train.shape[0]) or (x_test.shape[0] != y_test.shape[0]):
      print "Training set", x_train.shape, y_train.shape
      print "Test set", x_test.shape, y_test.shape
      print "Check lengths !!"
      sys.exit()

    for j in range(NbPerm):
      print "\t ----- tirage n %d -----"%j
      if iq == 0 and new == True:
        r = data_sets(x_train,y_train,verbose=False)
        permut.append(r[-1])
      (out_train,theta,out,p,permut[j]) = do_all_logistic_regression(x_train,y_train,x_test,y_test,output=True,wtr=permut[j],perc=True)
      dic[q].append(p)

      if plot:
        from Ijen_extract_features_new import confusion
        confusion(y_train,out_train,['EB','VT'],'training','',plot=True) 
        confusion(y_test,out,['EB','VT'],'test','',plot=True)
        plt.show()

  if new == True:
    with open('train_permut','wb') as file:
      my_pickler = cPickle.Pickler(file)
      my_pickler.dump(permut)
      file.close()

  if save:
    with open(filename,'wb') as file:
      my_pickler = cPickle.Pickler(file)
      my_pickler.dump(dic)
      file.close()

  plot_results(filename)


if __name__ == '__main__':

  filedir = '/home/nadege/Fingerprint/Test8'
  #HT_logreg(filedir,new=False,plot=False,save=True)

  #plot_results('%s/results'%filedir)
  trade_off('%s/results'%filedir)

