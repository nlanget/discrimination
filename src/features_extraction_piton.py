#!/usr/bin/env python
# encoding: utf-8

import os,glob,sys
from obspy.core import read,utcdatetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
Functions adapted to Piton de la Fournaise volcano dataset.
*.mat files
"""

class SeismicTraces():

  """
  Class which reads the seismic traces associated with an event.
  Also computes some basic features such as the spectrum and the 
  envelope and determines the P-onset.
  """

  def __init__(self,mat,comp,train=None):
    self.read_file(mat,comp,train)
    self.dt = .01
    self.station = 'BOR'
    self.t0 = 0
    self.starttime = 0
    self.i1, self.i2 = 0,len(self.tr)-1

    # Compute the spectrum
    self.spectrum(plot=False)


  def read_file(self,mat,comp,train):
    if train:
      i,type = train[0],train[1]
      kname = 'Sig%s'%type
      if comp == 'Z':
        self.tr = mat[kname][0][i][0][0][0]
      elif comp == 'E':
        self.tr = mat[kname][0][i][1][0][0]
      elif comp == 'N':
        self.tr = mat[kname][0][i][2][0][0]
      self.tr_env = mat['%s_ENV'%kname][i]
      self.tr_grad = mat['%s_GRAD'%kname][i]
    else:
      df = pd.DataFrame(mat['SIG2'])
      if comp == 'Z':
        self.tr = mat['SIG2'][:,0][0][0]
      elif comp == 'E':
        self.tr = mat['SIG2'][:,1][0][0]
      elif comp == 'N':
        self.tr = mat['SIG2'][:,2][0][0]
      self.tr_env = mat['SIG_ENV'].T[0]
      self.tr_grad = mat['SIG_GRAD'].T[0]


  def process_envelope(self):
    """
    Runs envelope processing on a waveform.
    """
    from obspy.signal import filter
    self.tr_env = filter.envelope(self.tr)


  def spectrum(self,plot=False):
    """
    Computes the Fourier transform of the signal
    """
    data = self.tr[self.i1:self.i2]
    self.TF = np.fft.fft(data)
    self.freqs = np.fft.fftfreq(len(self.TF),d=self.dt)

    if plot:
      time = np.arange(len(data))*self.dt
      fig = plt.figure()
      fig.set_facecolor('white')
      ax1 = fig.add_subplot(211)
      ax1.plot(time,data,'k')
      ax1.set_xlabel('Time (s)')
      ax2 = fig.add_subplot(212,title='Amplitude spectrum')
      ax2.plot(self.freqs[:len(self.freqs)/2],np.abs(self.TF[:len(self.TF)/2]),'k')
      ax2.set_xlabel('Frequency (Hz)')
      ax2.set_ylabel('Amplitude')
      ax2.set_xlim([0,20])
      fig.suptitle(self.tr.stats.station)
      plt.show()

  def duration(self):
    from waveform_features import signal_duration
    it0 = (self.t0 - self.starttime) * 1./self.dt
    self.ponset = signal_duration(self,it0=it0,plot=False)


  def display_traces(self):
    """
    Displays the seismic traces: raw signal, envelope and kurtosis gradient
    """

    time = np.arange(len(self.tr)) * self.dt

    fig = plt.figure()
    fig.set_facecolor('white')

    ax1 = fig.add_subplot(311,title='Raw data')
    ax1.plot(time,self.tr,'k')
    if 'ponset' in self.__dict__.keys():
      ax1.plot([time[self.ponset],time[self.ponset]],[np.min(self.tr),np.max(self.tr)],'r',lw=2)
      if 'dur' in self.__dict__.keys():
        ax1.plot([time[self.ponset]+self.dur,time[self.ponset]+self.dur],[np.min(self.tr),np.max(self.tr)],'b',lw=2)
    ax1.set_xticklabels('')

    ax2 = fig.add_subplot(312,title='Smoothed envelope')
    ax2.plot(time,self.tr_env,'k')
    if 'ponset' in self.__dict__.keys():
      ax2.plot([time[self.ponset],time[self.ponset]],[np.min(self.tr_env),np.max(self.tr_env)],'r',lw=2)
      emax = np.argmax(self.tr_env)
      ax2.plot([time[emax],time[emax]],[np.min(self.tr_env),np.max(self.tr_env)],'g',lw=2)
      if 'dur' in self.__dict__.keys():
        ax2.plot([time[self.ponset]+self.dur,time[self.ponset]+self.dur],[np.min(self.tr_env),np.max(self.tr_env)],'b',lw=2)
    
    ax3 = fig.add_subplot(313,title='Kurtosis gradient')
    ax3.plot(self.tr_grad,'k')
    plt.show()


  def amplitude_distribution(self):
    """
    Displays the seismic trace and its distribution of amplitude
    """
    import matplotlib.mlab as mlab

    time = np.arange(len(self.tr)) * self.dt

    fig = plt.figure()
    fig.set_facecolor('white')
    ax1 = fig.add_subplot(211,title='Raw data')
    ax1.plot(time,self.tr,'k')
    ax1.plot(time,self.tr_env,'r')

    data = self.tr_env
    ax2 = fig.add_subplot(212,title='Distribution of the amplitudes')
    n, bins, patches = ax2.hist(data,50)
    plt.show()


# ================================================================

def read_data_for_features_extraction(set='test',save=False):
  """
  Extracts the features from all seismic files
  If option 'save' is set, then save the pandas DataFrame as a .csv file
  """
  from scipy.io.matlab import mio
  from options import MultiOptions
  opt = MultiOptions()

  if save:
    if os.path.exists(opt.opdict['feat_filepath']):
      print "WARNING !! File %s already exists"%opt.opdict['feat_filepath']
      print "Check if you really want to replace it..." 
      sys.exit()

  list_features = opt.opdict['feat_list']
  df = pd.DataFrame(columns=list_features)

  if set == 'test':
    datafiles = glob.glob(os.path.join(opt.opdict['datadir'],'TestSet/SigEve_*'))
    datafiles.sort()
    liste = [os.path.basename(datafiles[i]).split('_')[1].split('.mat')[0] for i in range(len(datafiles))]
    liste = map(int,liste) # sort the list of file following the event number
    liste.sort()

    tsort = opt.read_csvfile(opt.opdict['label_filename'])
    tsort.index = tsort.Date

    for ifile,numfile in enumerate(liste):
      file = os.path.join(opt.opdict['datadir'],'TestSet/SigEve_%d.mat'%numfile)
      print ifile,file
      mat = mio.loadmat(file)

      for comp in opt.opdict['channels']:
        ind = (numfile,'BOR',comp)
        dic = pd.DataFrame(columns=list_features,index=[ind])
        dic['EventType'] = tsort[tsort.Date==numfile].Type.values[0]
        dic['Ponset'] = 0

        s = SeismicTraces(mat,comp)
        list_attr = s.__dict__.keys()

        if len(list_attr) > 2:
          if opt.opdict['option'] == 'norm':
            dic = extract_norm_features(s,list_features,dic,plot=False)
          elif opt.opdict['option'] == 'hash':
            permut_file = '%s/permut_%s'%(opt.opdict['libdir'],opt.opdict['feat_test'].split('.')[0])
            dic = extract_hash_features(s,list_features,dic,permut_file,plot=False)
          df = df.append(dic)

  elif set == 'train':
    datafile = os.path.join(opt.opdict['datadir'],'TrainingSetPlusSig_2.mat')
    mat = mio.loadmat(datafile)
    hob_all_EB = {}
    for i in range(mat['KurtoEB'].shape[1]):
      print "EB", i
      for comp in opt.opdict['channels']:
        dic = pd.DataFrame(columns=list_features,index=[(i,'BOR',comp)])
        dic['EventType'] = 'EB'
        dic['Ponset'] = 0
        
        s = SeismicTraces(mat,comp,train=[i,'EB'])
        list_attr = s.__dict__.keys()
        if len(list_attr) > 2:
          if opt.opdict['option'] == 'norm':
            dic = extract_norm_features(s,list_features,dic,plot=False)
          elif opt.opdict['option'] == 'hash':
            permut_file = '%s/permut_%s'%(opt.opdict['libdir'],opt.opdict['feat_train'].split('.')[0])
            dic = extract_hash_features(s,list_features,dic,permut_file,plot=False)
          df = df.append(dic)
      neb = i+1

    for i in range(mat['KurtoVT'].shape[1]):
      print "VT", i+neb
      for comp in opt.opdict['channels']:
        dic = pd.DataFrame(columns=list_features,index=[(i+neb,'BOR',comp)])
        dic['EventType'] = 'VT'
        dic['Ponset'] = 0

        s = SeismicTraces(mat,comp,train=[i,'VT'])
        list_attr = s.__dict__.keys()
        if len(list_attr) > 2:
          if opt.opdict['option'] == 'norm':
            dic = extract_norm_features(s,list_features,dic,plot=False)
          elif opt.opdict['option'] == 'hash':
            permut_file = '%s/permut_%s'%(opt.opdict['libdir'],opt.opdict['feat_train'].split('.')[0])
            dic = extract_hash_features(s,list_features,dic,permut_file,plot=False)
          df = df.append(dic)

  if save:
    print "Features written in %s"%opt.opdict['feat_filepath']
    df.to_csv(opt.opdict['feat_filepath'])

# ================================================================

def extract_norm_features(s,list_features,dic,plot=False):

    """
    Extraction of all features given by list_features, except hash 
    table values.
    """

    list_attr = s.__dict__.keys()
    if 'tr_grad' not in list_attr:
      for feat in list_features:
        dic[feat] = np.nan
      return dic
    # Determine P-onset
    s.duration()
    #s.display_traces()
    #s.amplitude_distribution()

    if len(list_attr) > 7:

      if 'Dur' in list_features:
        # Mean of the predominant frequency
        from waveform_features import spectrogram
        s, dic['MeanPredF'], dic['TimeMaxSpec'], dic['NbPeaks'], dic['Width'], hob, vals, dic['sPredF'] = spectrogram(s,plot=False)
        for i in range(len(vals)):
          dic['v%d'%i] = vals[i]
        dic['Dur'] = s.dur

      if 'Acorr' in list_features:
        from waveform_features import autocorrelation,filt_ratio
        vals = autocorrelation(s,plot=False)
        for i in range(len(vals)):
          dic['acorr%d'%i] = vals[i]

        vals = filt_ratio(s,plot=False)
        for i in range(len(vals)):
          dic['fratio%d'%i] = vals[i]

      if 'Ene20-30' in list_features:
       # Energy between 10 and 30 Hz
        from waveform_features import energy_between_10Hz_and_30Hz
        f1, f2 = 20,30
        dic['Ene%d-%d'%(f1,f2)] = energy_between_10Hz_and_30Hz(s.tr[s.i1:s.i2]/np.max(s.tr[s.i1:s.i2]),s.dt,wd=f1,wf=f2,ponset=s.ponset)

      if 'Ene5-10' in list_features:
        f1, f2 = 5,10
        dic['Ene%d-%d'%(f1,f2)] = energy_between_10Hz_and_30Hz(s.tr[s.i1:s.i2]/np.max(s.tr[s.i1:s.i2]),s.dt,wd=f1,wf=f2,ponset=s.ponset)

      if 'Ene0-5' in list_features:
        f1, f2 = .5,5
        dic['Ene%d-%d'%(f1,f2)] = energy_between_10Hz_and_30Hz(s.tr[s.i1:s.i2]/np.max(s.tr[s.i1:s.i2]),s.dt,wd=f1,wf=f2,ponset=s.ponset)

      if 'RappMaxMean' in list_features:
        # Max over mean ratio of the envelope
        from waveform_features import max_over_mean
        dic['RappMaxMean'] = max_over_mean(s.tr[s.i1:s.i2])

      if 'AsDec' in list_features:
        # Ascendant phase duration over descendant phase duration
        from waveform_features import growth_over_decay
        p = growth_over_decay(s)
        if p > 0:
          dic['AsDec'] = p

      if 'Growth' in list_features:
        from waveform_features import desc_and_asc
        dic['Growth'] = desc_and_asc(s)

      if 'Skewness' in list_features:
        # Skewness
        from waveform_features import skewness
        sk = skewness(s.tr_env[s.i1:s.i2])
        dic['Skewness'] = sk
        #s.amplitude_distribution()

      if 'Kurto' in list_features: 
        # Kurtosis
        from waveform_features import kurtosis_envelope
        k = kurtosis_envelope(s.tr_env[s.i1:s.i2])
        dic['Kurto'] = k

      if ('F_low' in list_features) or ('F_up' in list_features):
        # Lowest and highest frequency for kurtogram analysis
        from waveform_features import kurto_bandpass
        dic['F_low'],dic['F_up'] = kurto_bandpass(s,plot=False)

      if 'Centroid_time' in list_features:
        # Centroid time
        from waveform_features import centroid_time
        dic['Centroid_time'] = centroid_time(s.tr[s.i1:s.i2],s.dt,s.TF,s.ponset,tend=s.tend,plot=False)

      if 'RappMaxMeanTF' in list_features:
        # Max over mean ratio of the amplitude spectrum
        from waveform_features import max_over_mean
        dic['RappMaxMeanTF'] = max_over_mean(np.abs(s.TF[:len(s.TF)/2]))

      if 'IFslope' in list_features:
        # Average of the instantaneous frequency and slope of the unwrapped instantaneous phase
        from waveform_features import instant_freq
        vals, dic['IFslope'] = instant_freq(s.tr[s.i1:s.i2],s.dt,s.TF,s.ponset,s.tend,plot=False)
        for i in range(len(vals)):
          dic['if%d'%i] = vals[i]

      if ('ibw0' in list_features) or ('Norm_envelope' in list_features):
        # Average of the instantaneous frequency and normalized envelope
        from waveform_features import instant_bw
        vals, Nenv = instant_bw(s.tr[s.i1:s.i2],s.tr_env[s.i1:s.i2],s.dt,s.TF,s.ponset,s.tend,plot=False)
        dic['Norm_envelope'] = Nenv
        for i in range(len(vals)):
          dic['ibw%d'%i] = vals[i]

      if ('PredF' in list_features) or ('CentralF' in list_features) or ('Bandwidth' in list_features):
        # Spectral attributes
        from waveform_features import around_freq
        dic['PredF'], dic['Bandwidth'], dic['CentralF'] = around_freq(s.TF,s.freqs,plot=False)

      dic['Ponset'] = s.ponset
      return dic

# ================================================================

def extract_hash_features(s,list_features,dic,permut_file,plot=False):
  """
  Extracts hash table values.
  """
  from fingerprint_functions import FuncFingerprint, spectrogram, ponset_stack, vec_compute_signature, LSH

  list_attr = s.__dict__.keys()
  if 'tr_grad' not in list_attr:
    for feat in list_features:
      dic[feat] = np.nan
    return dic

  full_tr = s.tr
  grad = s.tr_grad
  q = [100.,.8,1]
  (full_spectro,f,full_time,end) = spectrogram(full_tr,param=q)
  ponset = ponset_stack(full_tr,full_spectro,full_time,plot=plot)
  dic['Ponset'] = ponset
 
  idebut = ponset-int(2*full_spectro.shape[1]/full_time[-1])
  print ponset, idebut
  if idebut < 0:
    idebut = 0
  ifin = idebut+full_spectro.shape[0]
  time = full_time[idebut:ifin]
  spectro = full_spectro[:,idebut:ifin]
  if spectro.shape[1] < spectro.shape[0]:
    spectro = full_spectro[:,-spectro.shape[0]:]
  haar_bin = FuncFingerprint(np.log10(spectro),time,full_tr,f,end,plot=plot,error=plot)

  m = haar_bin.shape[1]  # number of columns
  n = haar_bin.shape[0]  # number of rows
  vec_bin = np.zeros(m*n)
  for i in range(n):
    for j in range(m):
      vec_bin[m*i+j] = haar_bin[i,j]
  
  # HASH TABLES
  pp = 500
  p_file = '%s_%d'%(permut_file,full_spectro.shape[0])
  if not os.path.isfile(p_file):
    print "Permutation"
    from fingerprint_functions import define_permutation
    import cPickle
    permut = define_permutation(len(vec_bin),pp)
    with open(p_file,'wb') as file:
      my_pickler = cPickle.Pickler(file)
      my_pickler.dump(permut)
      file.close()
  else:
    import cPickle
    with open(p_file,'rb') as file:
      my_depickler = cPickle.Unpickler(file)
      permut = my_depickler.load()
      file.close()

  MH_sign = vec_compute_signature(vec_bin,permut)
  HashTab = LSH(MH_sign,l=50)
  for iht,ht in enumerate(HashTab):
    dic['%d'%iht] = ht

  if plot:
    plt.show()

  return dic

# ================================================================
if __name__ == '__main__':
  read_data_for_features_extraction(set='train',save=True)
