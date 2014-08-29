import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def read_binary_file(filename):
  """
  Reads a binary file.
  """
  import cPickle
  with open(filename,'rb') as file:
    my_depickler = cPickle.Unpickler(file)
    dic = my_depickler.load()
    file.close()
  return dic


def write_binary_file(filename,dic):
  """
  Writes in a binary file.
  """
  import cPickle
  with open(filename,'w') as file:
    my_pickler = cPickle.Pickler(file)
    my_pickler.dump(dic)
    file.close()


class Options(object):

  def __init__(self):

    self.opdict = {}

    self.opdict['option'] = 'norm' # could be 'norm' for classical seismic attributes or 'hash' for hash tables

    # Define directories and paths
    self.opdict['dir'] = 'Piton'
    self.opdict['channels'] = ['Z']#,'E','N']

    self.opdict['libdir'] = os.path.join('../lib',self.opdict['dir'])
    self.opdict['outdir'] = os.path.join('../results',self.opdict['dir'])

    self.opdict['boot'] = 10 # number of iterations (a new training set is generated at each 'iteration')

    if self.opdict['dir'] == 'Ijen':
      self.ijen()
    elif self.opdict['dir'] == 'Piton':
      self.piton()

    # Define options for classification functions
    self.opdict['method'] = 'lr' # could be 'lr' (logistic regression),'svm' (Support Vector Machine from scikit.learn package),'ova' (1-vs-all extractor), '1b1' (1-by-1 extractor), 'lrsk' (Logistic regression from scikit.learn package)
    self.opdict['plot_pdf'] = False # display the pdfs of the features
    self.opdict['save_pdf'] = False
    self.opdict['plot_confusion'] = False # display the confusion matrices
    self.opdict['save_confusion'] = False
    self.opdict['plot_sep'] = True # plot decision boundary
    self.opdict['save_sep'] = False
    self.opdict['plot_prec_rec'] = False # plot precision and recall

    import time
    date = time.localtime()
    if self.opdict['option'] == 'norm':
      # Features "normales"
      self.opdict['feat_all'] = ['AsDec','Bandwidth','CentralF','Centroid_time','Dur','Ene','Ene5-10','Ene0-5','F_low','F_up','Growth','IFslope','Kurto','MeanPredF','NbPeaks','PredF','RappMaxMean','RappMaxMeanTF','Skewness','sPredF','TimeMaxSpec','Width','ibw0','ibw1','ibw2','ibw3','ibw4','ibw5','ibw6','ibw7','ibw8','ibw9','if0','if1','if2','if3','if4','if5','if6','if7','if8','if9','v0','v1','v2','v3','v4','v5','v6','v7','v8','v9','Rectilinearity','Planarity','Azimuth','Incidence']
      self.opdict['feat_list'] = ['Kurto']
      #self.opdict['feat_log'] = ['AsDec']
      #self.opdict['feat_log'] = ['AsDec','Dur','Ene0-5','Growth','ibw0','MeanPredF','RappMaxMean','RappMaxMeanTF','TimeMaxSpec','v0','v8','v9'] # list of features to be normalized with np.log (makes data look more gaussians)
      #self.opdict['feat_list'] = ['Centroid_time','Dur','Ene0-5','F_up','Growth','Kurto','RappMaxMean','RappMaxMeanTF','Skewness','TimeMaxSpec','Width']
      #self.opdict['feat_list'] = ['Centroid_time','Dur','Ene0-5','F_up','Kurto','RappMaxMean','Skewness','TimeMaxSpec']
      #self.opdict['feat_list'] = ['Dur','F_up','Growth','Kurto','RappMaxMean','RappMaxMeanTF','TimeMaxSpec','Width']
      #self.opdict['feat_list'] = ['CentralF','Centroid_time','Dur','Ene0-5','F_up','Growth','IFslope','Kurto','MeanPredF','RappMaxMean','RappMaxMeanTF','Skewness','TimeMaxSpec','Width','if1','if2','if3','if4','if5','if6','if7','if8','if9','v0','v1','v2','v3','v4','v5','v6','v7','v8','v9']
      #self.opdict['feat_list'] = ['Centroid_time','Dur','Ene0-5','F_low','F_up','IFslope','Kurto','MeanPredF','RappMaxMean','Skewness','ibw0','if6','if7','if8','v8']

    if self.opdict['option'] == 'hash':
      # Hashing
      #self.opdict['feat_test'] = 'HT_%02d%02d.csv'%(date.tm_mday,date.tm_mon)
      self.opdict['feat_test'] = self.opdict['hash_test']
      if 'hash_train' in sorted(self.opdict):
        self.opdict['feat_train'] = self.opdict['hash_train']
      self.opdict['permut_file'] = '%s/permut_HT'%self.opdict['libdir']
      self.opdict['feat_list'] = map(str,range(50))
      #self.opdict['feat_list'] = ['0','3','5','8','26','29','30','41']
      #self.opdict['feat_log'] = map(str,range(50))


    self.opdict['feat_filepath'] = '%s/features/%s'%(self.opdict['outdir'],self.opdict['feat_test'])
    self.opdict['label_filename'] = '%s/%s'%(self.opdict['libdir'],self.opdict['label_test'])

    if self.opdict['method'] == 'lr' or self.opdict['method'] == 'svm' or self.opdict['method'] == 'lrsk':
      #self.opdict['result_file'] = 'results_%s_%dc_%df'%(self.opdict['method'],len(self.opdict['Types']),len(self.opdict['feat_list']))
      self.opdict['result_file'] = 'results_%s_%s'%(self.opdict['method'],self.opdict['feat_list'][0])
    else:
      self.opdict['result_file'] = '%s_%s_svm'%(self.opdict['method'].upper(),self.opdict['stations'][0])
    self.opdict['result_path'] = '%s/%s/%s'%(self.opdict['outdir'],self.opdict['method'].upper(),self.opdict['result_file'])
    if self.opdict['option'] == 'hash':
      self.opdict['result_path'] = '%s_HASH'%self.opdict['result_path']

    self.opdict['fig_path'] = '%s/figures'%self.opdict['outdir']

    self.opdict['types'] = None


  def ijen(self):
    self.opdict['network'] = 'ID'
    self.opdict['stations'] = ['IJEN']
    #self.opdict['stations'] = ['DAM','IBLW','IGEN','IJEN','IMLB','IPAL','IPLA','IPSW','KWUI','MLLR','POS','POSI','PSG','PUN','RAUN','TRWI']
    #self.opdict['Types'] = ['Hembusan','Hibrid','LF','Longsoran','Tektonik','Tremor','VulkanikA','VulkanikB']
    #self.opdict['Types'] = ['Tremor','VulkanikB','?']
    self.opdict['Types'] = ['Tremor','VulkanikB']
    self.opdict['datadir'] = os.path.join('../data',self.opdict['dir'],self.opdict['network'])
    self.opdict['feat_test'] = 'ijen_3006.csv'
    self.opdict['label_test'] = 'Ijen_reclass_all.csv'
    self.opdict['train_file'] = '%s/train_%d'%(self.opdict['libdir'],self.opdict['boot'])


  def piton(self):
    self.opdict['stations'] = ['BOR']
    self.opdict['Types'] = ['EB','VT']
    self.opdict['datadir'] = os.path.join('../data/%s/full_data'%self.opdict['dir'])
    #self.opdict['feat_train'] = 'clement_train.csv'
    #self.opdict['feat_test'] = 'clement_test.csv'
    self.opdict['feat_train'] = 'Piton_trainset.csv'
    self.opdict['feat_test'] = 'Piton_testset.csv'
    self.opdict['hash_train'] = 'HT_Piton_trainset.csv'
    self.opdict['hash_test'] = 'HT_Piton_testset.csv'
    self.opdict['label_train'] = 'class_train_set.csv'
    self.opdict['label_test'] = 'class_test_set.csv'
    self.opdict['learn_file'] = os.path.join(self.opdict['libdir'],'learning_set')


  def data_for_LR(self):

    self.raw_df = self.read_featfile()

    self.manuals = self.read_classification()
    self.manuals.index = self.manuals['Date'].values
    #self.manuals = self.manuals.reindex(columns=['Type'])

    def _get_x(self):
      return self._x

    def _set_x(self,new_x):
      self._x = new_x

    def _get_y(self):
      return self._y

    def _set_y(self,new_y):
      self._y = new_y


    x = property(_get_x,_set_x)
    y = property(_get_y,_set_y)


  def verify_features(self):
    for feat in self.opdict['feat_list']:
      if not feat in df.columns:
        print "The feature %s is not contained in the file..."%feat
        sys.exit()


  def verify_index(self):
    if list(self.x.index) != list(self.y.index):
      print "WARNING !! x and y do not contain the same list of events"
      print "x contains %d events ; y contains %d events"%(len(self.x),len(self.y))
      sys.exit()


  def read_csvfile(self,filename):
    df = pd.read_csv(filename)
    return df


  def read_featfile(self):
    """
    Reads the file containing event features
    """
    return pd.read_csv(self.opdict['feat_filepath'],index_col=False)


  def read_classification(self):
    """
    Reads the file with manual classification
    """
    return pd.read_csv(self.opdict['label_filename'])


  def write_x_and_y(self):
    self.x = self.raw_df.copy()
    self.y = self.manuals.copy()
    self.y = self.y.reindex(index=self.x.index)
    #self.y = self.x.reindex(columns=['EventType'])
    #self.y['Type'] = self.y.EventType.values
    self.verify_index()

    self.x = self.x.reindex(columns=self.opdict['feat_list'])
    self.x = self.x.dropna(how='any')

    self.y = self.y.reindex(index=self.x.index,columns=['Type'])

    self.verify_index()

    self.x.index = range(len(self.x))
    self.y.index = self.x.index

    self.classname2number()


  def classname2number(self):
    """
    Associates numbers to event types.
    """
    self.types = np.unique(self.y.values)
    self.st = self.opdict['types']
    if not self.st: 
      self.st = self.types

    nbtyp = []
    
    for ty in self.types:
      nb = len(self.y[self.y.Type==ty])
      if nb > 5 and ty in self.st:
        nbtyp.append((nb,ty))
      else:
        self.y = self.y[self.y.Type!=ty]
    nbtyp.sort(reverse=True)
    print nbtyp
    self.types = [ty[1] for ty in nbtyp]
    if 'nan' in self.types:
      self.y = self.y[self.y.Type!='nan']
      self.types.remove('nan')
    for i,ty in enumerate(self.types):
      self.y[self.y.Type==ty] = i
 
    self.numt = [i for i in range(len(self.types)) if str(self.types[i]) in self.st]

    self.x = self.x.reindex(index=self.y.index)


  def composition_dataset(self):
    """
    Plots the diagram with the different classes of the dataset.
    """

    self.types = np.unique(self.y.Type.values)
    nb = []
    print "COMPOSITION OF THE DATASET (%d events)"%len(self.y)
    for t in self.types:
      nb.append(len(self.y[self.y.Type==t]))
      print t, nb[-1]

    colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral','lightgreen','khaki','plum','powderblue']
    fig = plt.figure(figsize=(6,6))
    fig.set_facecolor('white')
    plt.pie(nb,labels=self.types,autopct='%1.1f%%',colors=colors)
    plt.title('Dataset')
    plt.show()


  def compute_pdfs(self):

    """
    Computes the probability density functions (pdfs) for all features and all event types.
    """

    from scipy.stats.kde import gaussian_kde

    self.types = np.unique(self.y.Type.values)

    dic={}
    for t in self.types:
      dic[t] = self.x[self.y.Type==t]

    self.gaussians = {}
    for feat in self.opdict['feat_list']:
      vec = np.linspace(self.x.min()[feat],self.x.max()[feat],200)
      #vec = np.linspace(self.x.min()[feat]+self.x.std()[feat],self.x.max()[feat]-self.x.std()[feat],200)
      #vec = np.linspace(self.x.mean()[feat]-self.x.std()[feat],self.x.mean()[feat]+self.x.std()[feat],200)

      self.gaussians[feat] = {}
      self.gaussians[feat]['vec'] = vec

      for it,t in enumerate(self.types):
        if len(dic[t][feat].values) > 1:
          if feat != 'NbPeaks':
            kde = gaussian_kde(dic[t][feat].values)
            a = np.cumsum(kde(vec))[-1]
            self.gaussians[feat][t] = kde(vec)/a
          else:
            self.gaussians[feat][t] = dic[t][feat].values


  def plot_all_pdfs(self,save=False):
    """
    Plots the pdfs.
    """

    if not hasattr(self,'gaussians'):
      self.compute_pdfs()

    list = []
    for feat in sorted(self.gaussians):
      fig = plt.figure()
      fig.set_facecolor('white')
      for it,t in enumerate(self.types):
        if it >= 7:
          lstyle = '--'
        else:
          lstyle = '-'
        if feat != 'NbPeaks':
          plt.plot(self.gaussians[feat]['vec'],self.gaussians[feat][t],ls=lstyle)
        else:
          list.append(self.gaussians[feat][t])
      if feat == 'NbPeaks':
        plt.hist(list,normed=True,alpha=.2)
      plt.title(feat)
      plt.legend(self.types)
      if save:
        plt.savefig('../results/%s/figures/fig_%s.png'%(self.opdict['dir'],feat))
      plt.show()


  def plot_superposed_pdfs(self,g,save=False):
    """
    Plots two kinds of pdfs (for example, the test set ones with the training set)
    """
    if not hasattr(self,'gaussians'):
      self.compute_pdfs()

    if list(sorted(g)) != list(sorted(self.gaussians)):
      print "WARNING !! Not the same features in gaussians..."
      sys.exit()

    colors = ['r','b','g','m','c','y','k']
    for feat in sorted(self.gaussians):
      if feat == 'NbPeaks':
        continue
      fig = plt.figure()
      fig.set_facecolor('white') 
      for it,t in enumerate(self.types):
        plt.plot(self.gaussians[feat]['vec'],self.gaussians[feat][t],c=colors[it],label=t)
        plt.plot(g[feat]['vec'],g[feat][t],ls='--',c=colors[it])
      plt.title(feat)
      plt.legend()
      if save:
        plt.savefig('../results/%s/figures/fig_%s.png'%(self.opdict['dir'],feat))
      plt.show()


  def plot_one_pdf(self,feat,coord=None):

    """
    Plots only one pdf, for a given feature.
    Possibility to plot a given point on it : coord is a numpy array [manual class,feature values,automatic class]
    """
    if not hasattr(self,'gaussians'):
      self.compute_pdfs()

    labels = list(self.types[:])
    if list(coord):
      labels.append('manual %s'%coord[2,0])
      #labels.append('auto')
    fig = plt.figure()
    fig.set_facecolor('white')
    for it,t in enumerate(self.types):
      if it >= 7:
        lstyle = '--'
      else:
        lstyle = '-'
      plt.plot(self.gaussians[feat]['vec'],self.gaussians[feat][t],ls=lstyle)
    if list(coord):
      ind  = [np.argmin(np.abs(self.gaussians[feat]['vec']-c)) for c in coord[1,:]]
      plt.plot(coord[1,:],self.gaussians[feat][coord[0,0]][ind],'ro')
      #plt.plot(coord[1,:],self.gaussians[feat][coord[2,0]][ind],'ko')
    plt.xlim([np.min(self.gaussians[feat]['vec']),np.max(self.gaussians[feat]['vec'])])
    plt.title(feat)
    plt.legend(labels,numpoints=1)
    #plt.savefig('%s/figures/Indet_%s_%s_brut.png'%(self.opdict['outdir'],coord[2,0],feat))
    plt.show()


class MultiOptions(Options):

  def __init__(self):
    Options.__init__(self)


  def do_tri(self):

    if 'feat_train' in sorted(self.opdict):
      self.opdict['feat_filepath'] = '%s/features/%s'%(self.opdict['outdir'],self.opdict['feat_train'])
      self.opdict['label_filename'] = '%s/%s'%(self.opdict['libdir'],self.opdict['label_train'])
      self.tri()

      self.xs_train, self.ys_train = {},{}
      for key in sorted(self.xs):
        self.xs_train[key] = self.xs[key].copy()
        self.ys_train[key] = self.ys[key].copy()
      self.train_x = self.x.copy()
      self.train_y = self.y.copy()

    self.opdict['feat_filepath'] = '%s/features/%s'%(self.opdict['outdir'],self.opdict['feat_test'])
    self.opdict['label_filename'] = '%s/%s'%(self.opdict['libdir'],self.opdict['label_test'])
    self.tri()


  def tri(self):

    self.data_for_LR()
    self.x = self.raw_df.copy()
    self.y = self.manuals.copy()

    self.x = self.x.reindex(columns=self.opdict['feat_list'])
    self.x = self.x.dropna(how='any')
    self.y = self.y[self.y.Type!='n']
    self.y = self.y.reindex(columns=['Date','Type'])

    # Do not select all classes
    ind = self.y[self.y.Type==self.opdict['Types'][0]].index
    for t in self.opdict['Types'][1:]:
      ind = ind.append(self.y[self.y.Type==t].index)
    self.y = self.y.reindex(index=ind)

    list_keys = self.x.index
    list_ev = [list_keys[i].split(',')[0][1:] for i in range(len(list_keys))]
    list_ev_u = np.unique(list_ev)
    list_ev_uniq = np.array(map(int,[ev for ev in list_ev_u if int(ev) in self.y.Date]))
    self.y = self.y.reindex(index=list_ev_uniq)
    self.y.index = range(len(self.y.index))
    self.xs, self.ys = {},{}
    trad = []
    k = 0

    for sta in self.opdict['stations']:
      for comp in self.opdict['channels']:
        ind = []
        for iev,event in enumerate(list_ev_uniq):
          if "(%s, '%s', '%s')"%(event,sta,comp) in list_keys:
            ind.append(iev)
        if ind:
          self.ys[k] = self.y.reindex(index=ind)
          names = [str(self.ys[k].Type.values[i]).replace(" ","") for i in range(len(self.ys[k]))]
          self.ys[k]['Type'] = names
          self.ys[k] = self.ys[k].reindex(columns=['Type'])
          trad.append((sta,comp))
          keys = ["(%d, '%s', '%s')"%(int(list_ev_uniq[ev]),sta,comp) for ev in ind]
          self.xs[k] = self.x.reindex(index=keys)
          self.xs[k].index = list_ev_uniq[ind]
          self.ys[k].index = self.xs[k].index

          if 'feat_log' in sorted(self.opdict):
            for feat in self.opdict['feat_log']:
              if feat in self.opdict['feat_list']:
                self.xs[k][feat] = np.log(self.xs[k][feat])
            self.xs[k] = self.xs[k].dropna(how='any')
            self.ys[k] = self.ys[k].reindex(index=self.xs[k].index)

          k = k+1

    self.trad = trad


  def count_number_of_events(self):
    """
    Counts and displays the number of events available at each station.
    """
    df = self.read_featfile()

    dic = {}
    for sta in self.opdict['stations']:
      dic[sta] = 0

    for key in df.index:
      stakey = key.split(',')[1].replace(" ","")
      stakey = stakey.replace("'","")
      if key.split(',')[2] == " 'Z')" and stakey in self.opdict['stations']:
        dic[stakey] = dic[stakey]+1
    print dic


  def features_onesta(self,sta,comp):
    """
    Returns the features of all events for a given station and a given component.
    """
    feats = self.read_featfile()
    feats = feats.reindex(columns=self.opdict['feat_list'])
    types = self.read_classification()
    types.index = types.Date

    list_index, list_event = [],[]
    for key in feats.index:
      stakey = key.split(',')[1].replace(" ","")
      stakey = stakey.replace("'","")
      if stakey == sta:
        compkey = key.split(',')[2].replace(" ","")
        compkey = compkey.replace("'","")
        compkey = compkey[:-1]
        if compkey == comp:
          list_index.append(key)
          event = key.split(',')[0][1:]
          list_event.append(event)
    feats = feats.reindex(index=list_index)
    types = types.reindex(index=map(int,list_event))
    feats.index = types.index
    feats = feats.dropna(how='any')
    types = types.reindex(index=feats.index)
    return feats,types


class TestOptions(MultiOptions):

  def __init__(self):

    self.opdict = {}
    self.opdict['dir'] = 'Test'
    self.opdict['stations'] = ['IJEN','KWUI']
    self.opdict['channels'] = ['Z']
    self.opdict['Types'] = ['VulkanikB','Tremor']

    self.opdict['libdir'] = os.path.join('../lib',self.opdict['dir'])
    self.opdict['outdir'] = os.path.join('../results',self.opdict['dir'])

    if len(self.opdict['stations']) == 1:
      self.opdict['feat_test'] = 'test_onesta.csv'
    elif len(self.opdict['stations']) > 1:
      self.opdict['feat_test'] = 'test_multista.csv'
    self.opdict['feat_filepath'] = '../results/%s/features/%s'%(self.opdict['dir'],self.opdict['feat_test'])
    self.opdict['feat_list'] = ['RappMaxMean','Kurto']


    self.opdict['label_filename'] = '%s/test_classification.csv'%self.opdict['libdir']

    self.opdict['result_file'] = 'results_%s'%self.opdict['feat_test'].split('.')[0]
    self.opdict['result_path'] = '../results/%s/%s'%(self.opdict['dir'],self.opdict['result_file'])

    self.opdict['class_auto_file'] = 'auto_class_%s.csv'%self.opdict['result_file'].split('_')[2]
    self.opdict['class_auto_path'] = '../results/%s/%s'%(self.opdict['dir'],self.opdict['class_auto_file'])

    self.opdict['method'] = 'lr'
    self.opdict['boot'] = 1
    self.opdict['train_file'] = '%s/train_%d'%(self.opdict['libdir'],self.opdict['boot'])
    self.opdict['plot_pdf'] = False
    self.opdict['plot_confusion'] = False

    self.opdict['types'] = None
