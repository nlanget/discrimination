import numpy as np
import pandas as pd
import os,sys,glob
from options import MultiOptions

class AnalyseResults(MultiOptions):

  def __init__(self,opt):
    MultiOptions.__init__(self,opt)

    self.opdict['class_auto_file'] = 'auto_class.csv'
    self.opdict['class_auto_path'] = '%s/%s/%s'%(self.opdict['outdir'],self.opdict['method'].upper(),self.opdict['class_auto_file'])

    self.concatenate_results()
    self.display_results()


  def read_result_file(self):
    """
    Reads the file containing the results
    """
    import cPickle
    with open(self.opdict['result_path'],'r') as file:
      my_depickler = cPickle.Unpickler(file)
      dic = my_depickler.load()
      file.close()
    self.opdict['feat_list'] = dic['features']
    del dic['features']
    self.results = dic
    self.opdict['stations'] = [key[0] for key in sorted(dic)]
    self.opdict['channels'] = [key[1] for key in sorted(dic)]


  def concatenate_results(self):
    """
    Does a synthesis of all classifications
    Stores the automatic classification into a .csv file.
    The index of the DataFrame structure contains the list of events.
    The columns of the DataFrame structure contain, for each event : 
      - Type : automatic class
      - Nb : number of stations implied in the classification process
      - NbDiff : number of different classes found
      - % : proportion of the final class among all classes. If the proportion are equal (for ex., 50-50), write ?
    """
    self.read_result_file()

    list_ev = []
    for key in sorted(self.results):
      list_ev = list_ev + list(self.results[key][0]['list_ev'])
    list_ev = np.array(list_ev)
    list_ev_all = np.unique(list_ev)

    df = pd.DataFrame(index=list_ev_all,columns=sorted(self.results),dtype=str)
    for key in sorted(self.results):
      for iev,event in enumerate(self.results[key][0]['list_ev']):
        df[key][event] = self.results[key][0]['classification'][iev]

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

    struct.to_csv(self.opdict['class_auto_path'])


  def read_manual_auto_files(self):
    """
    Reads the file with manual classification.
    Reads the file with automatic classification.
    """
    print self.opdict['result_path']
    print self.opdict['label_filename']
    print self.opdict['class_auto_path']

    self.auto = pd.read_csv(self.opdict['class_auto_path'],index_col=False)
    self.auto = self.auto.reindex(columns=['Type'])
    self.auto.Type[self.auto.Type=='LowFrequency'] = 'LF'

    self.man = self.read_classification()
    self.man.index = self.man.Date
    self.man = self.man.reindex(columns=['Type'],index=self.auto.index)

    self.man = self.man.dropna(how='any')
    self.auto = self.auto.reindex(index=self.man.index)

    for i in range(len(self.man.values)):
      self.man.values[i][0] = self.man.values[i][0].replace(" ","")


  def display_results(self):
    """
    Displays the success rate of the classification process.
    """
    self.read_manual_auto_files()

    N = len(self.man)
    list_man = self.man.values.ravel()
    list_auto = self.auto.values.ravel()
    sim = np.where(list_man==list_auto)[0]
    list_auto_sim = list_auto[sim]
    print "%% of well classified events : %.2f"%(len(sim)*100./N)
    print "\n"

    types = np.unique(list_auto)
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
    print "\n"


  def plot_confusion(self):
    """
    Plots the confusion matrix (test set).
    """
    from do_classification import confusion
    import matplotlib.pyplot as plt

    self.tri()
    self.classname2number()

    m = self.man
    a = self.auto
    for i in self.numt:
      m['Type'][m.Type==self.types[i]] = i
      a['Type'][a.Type==self.types[i]] = i
    a = a[a.Type!='?']
    m = m.reindex(index=a.index)

    confusion(m,a.values[:,0],self.types,'test',self.opdict['method'],plot=True)
    if self.opdict['save_confusion']:
      plt.savefig('%s/figures/test_%s.png'%(self.opdict['outdir'],self.opdict['result_file'][8:]))
    plt.show()
