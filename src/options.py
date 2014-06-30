import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Options(object):

  def __init__(self,opt='norm'):

    self.opdict = {}

    # Define directories and paths
    self.opdict['dir'] = 'Ijen'
    self.opdict['network'] = 'ID'
    self.opdict['stations'] = ['IJEN']
    #self.opdict['stations'] = ['DAM','IBLW','IGEN','IJEN','IMLB','IPAL','IPLA','KWUI','MLLR','POS','POSI','PSG','PUN','RAUN','TRWI']
    self.opdict['channels'] = ['HHZ','HHE','HHN','EHZ','EHE','EHN','BHZ','BHE','BHN']

    self.opdict['datadir'] = os.path.join('../data',self.opdict['dir'],self.opdict['network'])
    self.opdict['libdir'] = os.path.join('../lib',self.opdict['dir'])
    self.opdict['outdir'] = os.path.join('../results',self.opdict['dir'])

    # Define options for classification functions
    self.opdict['method'] = 'lr' # could be 'lr' (logistic regression),'svm' (Support Vector Machine from scikit.learn package),'ova' (1-vs-all extractor), '1b1' (1-by-1 extractor)
    self.opdict['boot'] = 1 # number of iterations (a new training set is generated at each 'iteration')
    self.opdict['plot'] = False # displays the pdfs of the features

    self.opdict['option'] = opt

    import time
    date = time.localtime()
    if opt == 'norm':
      # Features "normales"
      #self.opdict['feat_filename'] = 'ijen_%02d%02d.csv'%(date.tm_mday,date.tm_mon)
      self.opdict['feat_filename'] = 'ijen_2706.csv'
      self.opdict['feat_list'] = ['AsDec','Bandwidth','CentralF','Centroid_time','Dur','F_low','F_up','Growth','IFslope','Kurto','MeanPredF','NbPeaks','Norm_envelope','PredF','RappMaxMean','RappMaxMeanTF','Skewness','sPredF','TimeMaxSpec','Width','ibw0','ibw1','ibw2','ibw3','ibw4','ibw5','ibw6','ibw7','ibw8','ibw9','if0','if1','if2','if3','if4','if5','if6','if7','if8','if9','v0','v1','v2','v3','v4','v5','v6','v7','v8','v9']
      #self.opdict['feat_list'] = ['RappMaxMean','Kurto']

    if opt == 'hash':
      # Hashing
      self.opdict['feat_filename'] = 'HT_ijen_%02d%02d.csv'%(date.tm_mday,date.tm_mon)
      self.opdict['feat_list'] = map(str,range(50))

    self.opdict['feat_filepath'] = '../results/%s/features/%s'%(self.opdict['dir'],self.opdict['feat_filename'])
    self.opdict['label_filename'] = '%s/Ijen_class_all.csv'%self.opdict['libdir']

    self.opdict['result_file'] = 'results_%s_%s'%(self.opdict['feat_filename'].split('.')[0],self.opdict['method'])
    self.opdict['result_path'] = '../results/%s/%s'%(self.opdict['dir'],self.opdict['result_file'])

    self.opdict['class_auto_file'] = 'auto_class_%s.csv'%self.opdict['result_file'].split('_')[2]
    self.opdict['class_auto_path'] = '../results/%s/%s'%(self.opdict['dir'],self.opdict['class_auto_file'])

    self.opdict['types'] = None


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
    return pd.read_csv(self.opdict['feat_filepath'],index_col=False)


  def read_classification(self):
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


  def compute_pdfs(self):

    """
    Computes the probability density functions (pdfs) for all features and all event types.
    """

    from scipy.stats.kde import gaussian_kde

    dic={}
    for i,t in enumerate(self.types):
      if type(self.y.values[0,0]) == int:
        dic[t] = self.x[self.y.values[:,0] == i]
      elif type(self.y.values[0,0]) == str:
        dic[t] = self.x[self.y.values[:,0] == t]

    self.gaussians = {}
    for feat in self.opdict['feat_list']:
      print feat
      vec = np.linspace(self.x.min()[feat],self.x.max()[feat],200)

      self.gaussians[feat] = {}
      self.gaussians[feat]['vec'] = vec

      for it,t in enumerate(self.types):
        if len(dic[t][feat].values) > 1:
          if feat != 'NbPeaks':
            kde = gaussian_kde(dic[t][feat].values)      
            a=np.cumsum(kde(vec))[-1]
            self.gaussians[feat][t] = kde(vec)/a
          else:
            self.gaussians[feat][t] = dic[t][feat].values


  def plot_all_pdfs(self,save=False):

    """
    Plots the pdfs.
    """

    if not hasattr(self,'gaussians'):
      self.compute_pdfs()

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
      if feat == 'NbPeaks':
        plt.hist(list,normed=True,alpha=.2)
      plt.title(feat)
      plt.legend(self.types)
      if save:
        plt.savefig('../results/%s/figures/fig_%s.png'%(self.opdict['dir'],feat))
      plt.show()


  def plot_one_pdf(self,feat,coord=None):

    """
    Plots only one pdf, for a given feature.
    """

    if not hasattr(self,'gaussians'):
      self.compute_pdfs()

    labels = self.types[:]
    if coord:
      labels.append('manual')
      labels.append('auto')
    fig = plt.figure()
    fig.set_facecolor('white')
    for it,t in enumerate(self.types):
      if it >= 7:
        lstyle = '--'
      else:
        lstyle = '-'
      plt.plot(self.gaussians[feat]['vec'],self.gaussians[feat][t],ls=lstyle)
    if coord:
      ind  = np.argmin(np.abs(self.gaussians[feat]['vec']-coord[1][0]))
      plt.plot(coord[1],self.gaussians[feat][coord[0]][ind],'ro')
      plt.plot(coord[1],self.gaussians[feat][coord[2]][ind],'ko')
    plt.title(feat)
    plt.legend(labels,numpoints=1)
    plt.show()


class MultiOptions(Options):

  def __init__(self,opt):
    Options.__init__(self,opt)

  def tri(self):

    self.data_for_LR()

    self.x = self.raw_df.copy()
    self.y = self.manuals.copy()

    self.x = self.x.reindex(columns=self.opdict['feat_list'])
    self.x = self.x.dropna(how='any')
    self.y = self.y.reindex(columns=['Date','Type'])

    list_keys = self.x.index
    list_ev = [list_keys[i].split(',')[0][1:] for i in range(len(list_keys))]
    list_ev_uniq = np.unique(list_ev)
    self.y.index = range(len(self.y.index))
    self.xs, self.ys = {},{}
    trad = []
    k = 0

    for sta in self.opdict['stations']:
      for comp in ['Z','N','E']:
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

          if list(self.ys[k]['Type'].values):
            self.ys[k][self.ys[k].Type=='HarmonikTremor'] = 'Tremor'
            self.ys[k][self.ys[k].Type=='TremorHarmonik'] = 'Tremor'
            self.ys[k][self.ys[k].Type=='Tremorharmonik'] = 'Tremor'
            self.ys[k][self.ys[k].Type=='TremorLow'] = 'Tremor'
            self.ys[k][self.ys[k].Type=='HembusanTremor'] = 'Hembusan'
            self.ys[k][self.ys[k].Type=='LF'] = 'LowFrequency'
            self.ys[k][self.ys[k].Type=='Longoran'] = 'Longsoran'
            self.ys[k][self.ys[k].Type=='Longsoran/Ggr'] = 'Longsoran'
            self.ys[k][self.ys[k].Type=='TektonikJauh/MMIII'] = 'TektonikJauh'
            self.ys[k][self.ys[k].Type=='TektonikJauhMMIII'] = 'TektonikJauh'
            self.ys[k][self.ys[k].Type=='TektonikJauhMMIIV'] = 'TektonikJauh'
            self.ys[k][self.ys[k].Type=='TektonikLokalMMIIII'] = 'TektonikLokal'
            self.ys[k][self.ys[k].Type=='TektonilLokal'] = 'TektonikLokal'

          k = k+1

    self.trad = trad


class TestOptions(MultiOptions):

  def __init__(self):

    self.opdict = {}
    self.opdict['dir'] = 'Test'
    self.opdict['stations'] = ['IJEN','KWUI']
    self.opdict['channels'] = ['HHZ','HHE','HHN','EHZ','EHE','EHN','BHZ','BHE','BHN']

    self.opdict['libdir'] = os.path.join('../lib',self.opdict['dir'])
    self.opdict['outdir'] = os.path.join('../results',self.opdict['dir'])

    if len(self.opdict['stations']) == 1:
      self.opdict['feat_filename'] = 'test_onesta.csv'
    elif len(self.opdict['stations']) > 1:
      self.opdict['feat_filename'] = 'test_multista.csv'
    self.opdict['feat_filepath'] = '../results/%s/features/%s'%(self.opdict['dir'],self.opdict['feat_filename'])
    self.opdict['feat_list'] = ['RappMaxMean','Kurto']


    self.opdict['label_filename'] = '%s/test_classification.csv'%self.opdict['libdir']

    self.opdict['result_file'] = 'results_%s'%self.opdict['feat_filename'].split('.')[0]
    self.opdict['result_path'] = '../results/%s/%s'%(self.opdict['dir'],self.opdict['result_file'])

    self.opdict['class_auto_file'] = 'auto_class_%s.csv'%self.opdict['result_file'].split('_')[2]
    self.opdict['class_auto_path'] = '../results/%s/%s'%(self.opdict['dir'],self.opdict['class_auto_file'])

    self.opdict['method'] = 'log_reg'
    self.opdict['boot'] = 1
    self.opdict['plot'] = False

    self.opdict['types'] = None
