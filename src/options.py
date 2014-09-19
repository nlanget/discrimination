import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


### ====================================================
### FUNCTIONS ###
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


def name2num(df_str,col,names,keep_names=None):
  """
  Associates a number to a string.
  df is a pandas DataFrame.
  col is the name of the column of df for which we want the conversion to be done.
  names is the list of all the existing and different names 
  (Optional) keep_names is the list of names we really want to keep.
  """
  df = df_str.copy()
  
  if not keep_names:
    keep_names = names

  nbtyp = []
  for name in names:
    nb = len(df[df[col]==name])
    if nb > 5 and name in keep_names:
      nbtyp.append((nb,name))
    else:
      df = df[df[col]!=name]
  nbtyp.sort(reverse=True)
  print nbtyp
  names = [nbt[1] for nbt in nbtyp]
  if 'nan' in names:
    df = df[df[col]!='nan']
    names.remove('nan')
  df = df.reindex(columns=[col,'Num%s'%col])
  for i,name in enumerate(names):
    df['Num%s'%col][df[col]==name] = i

  numt = [i for i in range(len(names)) if str(names[i]) in keep_names]

  return df, names, numt


def conversion(df1,df2,col):
  """
  Adds a column Num to df2 in accordance with df1
  where the column col is a list of str
  """
  str = np.unique(df1[col].values)
  df2 = df2.reindex(columns=[col,'Num%s'%col])

  for s in str:
    nb = df1['Num%s'%col][df1[col]==s].values[0]
    df2['Num%s'%col][df2[col]==s] = nb

  return df2

### ====================================================
### HEADER FOR IJEN DATA ###
def ijen():
  opdict = {}
  optdict['dir'] = 'Ijen'
  opdict['channels'] = ['Z']#,'E','N']
  opdict['network'] = 'ID'
  opdict['stations'] = ['IJEN']
  #opdict['stations'] = ['DAM','IBLW','IGEN','IJEN','IMLB','IPAL','IPLA','IPSW','KWUI','MLLR','POS','POSI','PSG','PUN','RAUN','TRWI']
  #opdict['types'] = ['Hembusan','Hibrid','LF','Longsoran','Tektonik','Tremor','VulkanikA','VulkanikB']
  opdict['types'] = ['Tremor','VulkanikB','?']
  #opdict['types'] = ['Tremor','VulkanikB']
  opdict['datadir'] = os.path.join('../data',opdict['dir'],opdict['network'])
  ### FEATURES FILE
  opdict['feat_test'] = 'ijen_3006.csv'
  ### LABEL FILE
  opdict['label_test'] = 'Ijen_3class_all.csv'
  ### FEATURES LIST
  opdict['feat_all'] = ['AsDec','Bandwidth','CentralF','Centroid_time','Dur','Ene20-30','Ene5-10','Ene0-5','F_low','F_up','Growth','IFslope','Kurto','MeanPredF','NbPeaks','PredF','RappMaxMean','RappMaxMeanTF','Skewness','sPredF','TimeMaxSpec','Width','ibw0','ibw1','ibw2','ibw3','ibw4','ibw5','ibw6','ibw7','ibw8','ibw9','if0','if1','if2','if3','if4','if5','if6','if7','if8','if9','v0','v1','v2','v3','v4','v5','v6','v7','v8','v9']
  return opdict

### HEADER FOR PITON DE LA FOURNAISE DATA ###
def piton():
  opdict = {}
  opdict['dir'] = 'Piton'
  opdict['channels'] = ['Z']#,'E','N']
  opdict['stations'] = ['BOR']
  opdict['types'] = ['EB','VT']
  opdict['datadir'] = os.path.join('../data/%s/full_data'%opdict['dir'])
  ### FEATURES FILES
  #opdict['feat_train'] = 'clement_train.csv'
  #opdict['feat_test'] = 'clement_test.csv'
  opdict['feat_train'] = 'Piton_trainset.csv'
  opdict['feat_test'] = 'Piton_testset.csv'
  ### HASH TABLE FEATURES FILES
  opdict['hash_train'] = 'HT_Piton_trainset.csv'
  opdict['hash_test'] = 'HT_Piton_testset.csv'
  ### LABEL FILES
  opdict['label_train'] = 'class_train_set.csv'
  opdict['label_test'] = 'class_test_set.csv'
  ### DECOMPOSITION OF THE TRAINING SET
  opdict['learn_file'] = 'learning_set'
  ### FEATURES LIST
  opdict['feat_all'] = ['AsDec','Bandwidth','CentralF','Centroid_time','Dur','Ene','Ene5-10','Ene0-5','F_low','F_up','Growth','IFslope','Kurto','MeanPredF','NbPeaks','PredF','RappMaxMean','RappMaxMeanTF','Skewness','sPredF','TimeMaxSpec','Width','ibw0','ibw1','ibw2','ibw3','ibw4','ibw5','ibw6','ibw7','ibw8','ibw9','if0','if1','if2','if3','if4','if5','if6','if7','if8','if9','v0','v1','v2','v3','v4','v5','v6','v7','v8','v9','Rectilinearity','Planarity','Azimuth','Incidence'] 
  return opdict

### ====================================================
### CLASS Options() ###
class Options(object):

  def __init__(self):

    self.opdict = {}
    #self.opdict = ijen()
    self.opdict = piton()

    ### Import classification options ###
    self.set_classi_options()
    ### Complete directory/file names ###
    self.fill_opdict()


  def set_classi_options(self):
    """
    Define options for classification functions
    """
    ### Type of features ### 
    # could be 'norm' for classical seismic attributes or 'hash' for hash tables
    self.opdict['option'] = 'norm'

    ### Number of iterations ###
    # a new training set is generated at each 'iteration'
    self.opdict['boot'] = 1

    ### Choice of the classification algorithm ###
    # could be 'lr' (logistic regression)
    # or 'svm' (Support Vector Machine from scikit.learn package)
    # or 'ova' (1-vs-all extractor), 
    # or '1b1' (1-by-1 extractor)
    # or 'lrsk' (Logistic regression from scikit.learn package)
    self.opdict['method'] = 'svm'

    ### Also compute the probabilities for each class ###
    self.opdict['probas'] = False

    ### Display and save the PDFs of the features ###
    self.opdict['plot_pdf'] = False
    self.opdict['save_pdf'] = False

    ### Display and save the confusion matrices ###
    self.opdict['plot_confusion'] = False
    self.opdict['save_confusion'] = False

    ### Plot and save the decision boundaries ###
    self.opdict['plot_sep'] = False
    self.opdict['save_sep'] = False
    self.opdict['compare'] = True # plot SVM and LR decision boundaries on the same plot

    ### Plot precision and recall ###
    self.opdict['plot_prec_rec'] = False # plot precision and recall


  def fill_opdict(self):
    """
    Check and create directories/files paths.
    """
    ### Check the existence of directories and create if necessary ###
    ### Data directory
    self.verify_dir(self.opdict['datadir'])
    ### Library directory
    self.opdict['libdir'] = os.path.join('../lib',self.opdict['dir'])
    self.verify_dir(self.opdict['libdir'])
    ### Output directory
    self.opdict['outdir'] = os.path.join('../results',self.opdict['dir'])
    self.verify_and_create(self.opdict['outdir'])
    ### Figures directory
    self.opdict['fig_path'] = '%s/figures'%self.opdict['outdir']
    self.verify_and_create(self.opdict['fig_path'])
    ### Result directory
    self.opdict['res_dir'] = '%s/%s'%(self.opdict['outdir'],self.opdict['method'].upper())
    self.verify_and_create(self.opdict['res_dir'])

    ### if there is an independent training set
    if 'feat_train' in sorted(self.opdict):
      if not 'label_train' in sorted(self.opdict):
        print "WARNING !!! check training set features and label files...."
 
    ### Check the existence of files ###
    ### Features file
    self.opdict['feat_filename'] = '%s/features/%s'%(self.opdict['outdir'],self.opdict['feat_test'])
    self.verify_file(self.opdict['feat_filename'])
    ### Label file
    self.opdict['label_filename'] = '%s/%s'%(self.opdict['libdir'],self.opdict['label_test'])
    self.verify_file(self.opdict['label_filename'])


    ### DIFFERENT TRAINING SETS (CREATED FROM THE TEST SET)
    if 'train_file' in sorted(self.opdict):
      self.opdict['train_file'] = '%s/train_%d'%(self.opdict['libdir'],self.opdict['boot'])
    ### DECOMPOSITION OF THE TRAINING SET (training, CV, test)
    if 'learn_file' in sorted(self.opdict):
      self.opdict['learn_file'] = os.path.join(self.opdict['libdir'],self.opdict['learn_file'])

    import time
    date = time.localtime()
    if self.opdict['option'] == 'norm':
      # Features "normales"
      #self.opdict['feat_list'] = self.opdict['feat_all']
      self.opdict['feat_list'] = ['RappMaxMean']
      self.opdict['feat_log'] = ['RappMaxMean']
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

    ### RESULT FILE ###
    NB_feat = len(self.opdict['feat_list'])
    if self.opdict['method'] in ['lr','svm','lrsk']:
      if NB_feat == 1:
        self.opdict['result_file'] = 'results_%s_%s'%(self.opdict['method'],self.opdict['feat_list'][0])
      else:
        self.opdict['result_file'] = 'results_%s_%dc_%df'%(self.opdict['method'],len(self.opdict['types']),len(self.opdict['feat_list']))
 
    else:
      self.opdict['result_file'] = '%s_%s_svm'%(self.opdict['method'].upper(),self.opdict['stations'][0])
    self.opdict['result_path'] = '%s/%s'%(self.opdict['res_dir'],self.opdict['result_file'])

    if self.opdict['option'] == 'hash':
      self.opdict['result_path'] = '%s_HASH'%self.opdict['result_path']


  def synthetics(self):
    """
    Options for synthetic tests
    """
    self.opdict = {}
    self.opdict['dir'] = 'Test'
    self.opdict['datadir'] = '../data/%s'%self.opdict['dir']
    self.opdict['stations'] = ['STA']
    self.opdict['channels'] = ['Z']
    self.opdict['types'] = ['A','B']

    self.sep = 'well'
    self.opdict['feat_train'] = '%s_%dc_train.csv'%(self.sep,len(self.opdict['types']))
    self.opdict['feat_test'] = '%s_%dc_test.csv'%(self.sep,len(self.opdict['types']))
    self.opdict['label_train'] = '%s_%dc_train.csv'%(self.sep,len(self.opdict['types']))
    self.opdict['label_test'] = '%s_%dc_test.csv'%(self.sep,len(self.opdict['types']))
    self.opdict['learn_file'] = 'learning_set'
    self.opdict['feat_all'] = ['x1','x2']

    self.opdict['option'] = 'norm'
    self.opdict['method'] = 'lr'
    self.opdict['probas'] = False
    self.opdict['boot'] = 1
    self.opdict['plot_pdf'] = False # display the pdfs of the features
    self.opdict['save_pdf'] = False
    self.opdict['plot_confusion'] = False # display the confusion matrices
    self.opdict['save_confusion'] = False
    self.opdict['plot_sep'] = False # plot decision boundary
    self.opdict['save_sep'] = False
    self.opdict['plot_prec_rec'] = False # plot precision and recall
    self.opdict['compare'] = False


  def read_csvfile(self,filename):
    """
    Read a *.csv file and store it into a pandas DataFrame structure.
    """
    df = pd.read_csv(filename)
    return df


  def read_featfile(self):
    """
    Read the file containing event features
    """
    return pd.read_csv(self.opdict['feat_filename'],index_col=False)


  def read_classification(self):
    """
    Read the file with manual classification
    """
    return pd.read_csv(self.opdict['label_filename'])


  def verify_dir(self,dirname):
    """
    Check the existence of a directory.
    """
    if not os.path.isdir(dirname):
      print "WARNING !! Directory %s does not exist !!"%dirname
      sys.exit()


  def verify_and_create(self,dirname):
    """
    Check the existence of a directory and creates 
    it if it does not exist.
    """
    if not os.path.isdir(dirname):
      print "Create directory %s..."%dirname
      os.makedirs(dirname)


  def verify_file(self,filename):
    """
    Check the existence of a file. 
    """
    if not os.path.isfile(filename):
      print "WARNING !! File %s does not exist !!"%filename
      sys.exit()


  def verify_features(self):
    """
    Check if the features list and the features file are coherent.
    """
    for feat in self.opdict['feat_list']:
      if not feat in self.raw_df.columns:
        print "The feature %s is not contained in the file..."%feat
        sys.exit()


  def verify_index(self):
    """
    Check the indices of the features and the labels.
    """
    if list(self.x.index) != list(self.y.index):
      print "WARNING !! x and y do not contain the same list of events"
      print "x contains %d events ; y contains %d events"%(len(self.x),len(self.y))
      sys.exit()


  def data_for_LR(self):
    """
    Prepare data for the classification process.
    """

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
    Associate numbers to event types.
    """
    self.st = self.opdict['types']
    self.types = np.unique(self.y.Type.values)
    self.y, self.types, self.numt = name2num(self.y,'Type',self.types,keep_names=self.st)
    self.x = self.x.reindex(index=self.y.index)


  def composition_dataset(self):
    """
    Plot the diagram with the different classes of the dataset.
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
    Compute the Probability Density Functions (PDFs) for all features and all event types.
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
    Plot the PDFs.
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
          plt.plot(self.gaussians[feat]['vec'],self.gaussians[feat][t],ls=lstyle,lw=2.)
        else:
          list.append(self.gaussians[feat][t])
      if feat == 'NbPeaks':
        plt.hist(list,normed=True,alpha=.2)
      plt.title(feat)
      plt.legend(self.types)
      if save:
        savename = '%s/fig_%s.png'%(self.opdict['fig_path'],feat)
        print "Save PDF for feature %s in %s"%(feat,savename)
        plt.savefig(savename)
      plt.show()


  def plot_superposed_pdfs(self,g,save=False):
    """
    Plot two kinds of PDFs (for example, the test set and the training set PDFs)
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
        plt.plot(self.gaussians[feat]['vec'],self.gaussians[feat][t],c=colors[it],label=t,lw=2.)
        plt.plot(g[feat]['vec'],g[feat][t],ls='--',c=colors[it],lw=2.)
      plt.title(feat)
      plt.legend()
      if save:
        savename = '%s/CLEMENT/compCL_%s.png'%(self.opdict['fig_path'],feat)
        print "Save PDF for feature %s in %s"%(feat,savename)
        plt.savefig(savename)
      plt.show()


  def plot_one_pdf(self,feat,coord=None):

    """
    Plot only one pdf, for a given feature.
    Possibility to plot a given point on it : coord is a numpy array [manual class,feature values,automatic class(,probabilities)]
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
      if len(coord) == 3:
        plt.plot(coord[1,:],self.gaussians[feat][coord[0,0]][ind],'ro')
      elif len(coord) == 4:
        plt.scatter(coord[1,:],self.gaussians[feat][coord[0,0]][ind],s=40,c=list(coord[3]),cmap=plt.cm.hot)
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
      self.opdict['feat_filename'] = '%s/features/%s'%(self.opdict['outdir'],self.opdict['feat_train'])
      self.verify_file(self.opdict['feat_filename'])
      self.opdict['label_filename'] = '%s/%s'%(self.opdict['libdir'],self.opdict['label_train'])
      self.verify_file(self.opdict['label_filename'])
      self.tri()

      self.xs_train, self.ys_train = {},{}
      for key in sorted(self.xs):
        self.xs_train[key] = self.xs[key].copy()
        self.ys_train[key] = self.ys[key].copy()
      self.train_x = self.x.copy()
      self.train_y = self.y.copy()

    self.opdict['feat_filename'] = '%s/features/%s'%(self.opdict['outdir'],self.opdict['feat_test'])
    self.opdict['label_filename'] = '%s/%s'%(self.opdict['libdir'],self.opdict['label_test'])
    self.tri()


  def tri(self):

    self.data_for_LR()
    self.verify_features()
    self.x = self.raw_df.copy()
    self.y = self.manuals.copy()

    self.x = self.x.reindex(columns=self.opdict['feat_list'])
    self.x = self.x.dropna(how='any')
    self.y = self.y[self.y.Type!='n']
    self.y = self.y.reindex(columns=['Date','Type'])

    # Do not select all classes
    self.y.Type = map(str,list(self.y.Type))
    ind = self.y[self.y.Type==self.opdict['types'][0]].index
    for t in self.opdict['types'][1:]:
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
    Count and display the number of events available at each station.
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
    self.opdict['types'] = ['VulkanikB','Tremor']

    self.opdict['libdir'] = os.path.join('../lib',self.opdict['dir'])
    self.opdict['outdir'] = os.path.join('../results',self.opdict['dir'])

    if len(self.opdict['stations']) == 1:
      self.opdict['feat_test'] = 'test_onesta.csv'
    elif len(self.opdict['stations']) > 1:
      self.opdict['feat_test'] = 'test_multista.csv'
    self.opdict['feat_filename'] = '../results/%s/features/%s'%(self.opdict['dir'],self.opdict['feat_test'])
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
