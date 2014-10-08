#!/usr/bin/env python
# encoding: utf-8

import os,glob,sys
from obspy.core import read,utcdatetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class SeismicTraces():

  """
  Class which reads the seismic traces associated with an event.
  Also computes some basic features such as the spectrum and the 
  envelope and determines the P-onset.
  """

  def __init__(self,filename,origin):
    self.t0 = origin
    self.tr = self.read_file(filename)

    if len(self.tr) > 0:

      datadir = os.path.dirname(filename)
      fname = os.path.basename(filename)

      # Kurtosis gradient
      filename = os.path.join(datadir,'GRAD/%s_GRAD'%fname)
      if os.path.exists(filename):
        self.tr_grad = self.read_file(filename)
      else:
        filename = os.path.join(datadir,"grad/%s_GRAD"%fname)
        if os.path.exists(filename):
          self.tr_grad = self.read_file(filename)
        else:
          print "Warning !! File %s_GRAD does not exist"%fname

      # Envelope
      filename = os.path.join(datadir,"ENV/%s_env.*"%fname)
      if os.path.exists(filename):
        self.tr_env = self.read_file(filename)
      else:
        self.process_envelope()

  def read_file(self,filename):
    #tb,ta = 15, 135
    tb,ta = 20, 100
    st = read(filename)
    if len(st) > 0:
      if len(st) > 1:
        st.merge(fill_value=0)
        st.write(filename,'MSEED')
      st.filter('bandpass',freqmin=1,freqmax=10)
      tr = st[0]
      self.dt = tr.stats.delta
      self.starttime = self.t0 - tb 
      #self.i1, self.i2 = 0,len(tr)-1
      if len(tr.data) != int((ta+tb)*1./self.dt)+1:
        print len(tr.data),int((ta+tb)*1./self.dt)+1
        return [] 
      return tr.data
    else:
      return []


  def process_envelope(self):
    """
    Runs envelope processing on a waveform.
    """
    from obspy.signal import filter
    env = filter.envelope(self.tr)
    # Smooth the envelope
    w = 51 # length of the sliding window
    s = np.r_[env[w-1:0:-1],env,env[-1:-w:-1]]
    window = np.ones(w,'d')
    self.tr_env = np.convolve(window/window.sum(),s,mode='valid')[w/2:-w/2]


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

def read_data_for_features_extraction(save=False):
  """
  Extracts the features from all seismic files
  If option 'save' is set, then save the pandas DataFrame as a .csv file
  """
  from options import MultiOptions
  opt = MultiOptions()

  if save:
    if os.path.exists(opt.opdict['feat_filename']):
      print "WARNING !! File %s already exists"%opt.opdict['feat_filename']
      print "Check if you really want to replace it..." 
      sys.exit()

  list_features = opt.opdict['feat_list']
  df = pd.DataFrame(columns=list_features)

  hob_all = {}

  # Classification
  tsort = opt.read_classification()
  tsort.index = tsort.Date
  tsort = tsort.reindex(columns=['Date','Type'])

  list_sta = opt.opdict['stations']
  for ifile in range(tsort.shape[0]):
    date = tsort.values[ifile,0]
    type = tsort.values[ifile,1]

    for sta in list_sta:
      print "#####",sta
      counter = 0
      for comp in opt.opdict['channels']:
        ind = (date,sta,comp)
        dic = pd.DataFrame(columns=list_features,index=[ind])
        dic['EventType'] = type
        dic['Ponset'] = 0
        list_files = glob.glob(os.path.join(opt.opdict['datadir'],sta,'*%s.D'%comp,'*%s.D*%s_%s*'%(comp,str(date)[:8],str(date)[8:])))
        list_files.sort()
        if len(list_files) > 0:
          file =  list_files[0]
          print ifile, file
          if opt.opdict['option'] == 'norm':
            counter = counter + 1
            dic = extract_norm_features(list_features,date,file,dic)
          elif opt.opdict['option'] == 'hash':
            permut_file = '%s/permut_%s'%(opt.opdict['libdir'],opt.opdict['feat_test'].split('.')[0])
            dic = extract_hash_features(list_features,date,file,dic,permut_file,plot=True)
          df = df.append(dic)

      if counter == 3 and ('Rectilinearity' in list_features or 'Planarity' in list_features or 'Azimuth' in list_features or 'Incidence' in list_features):
        from waveform_features import polarization_analysis
        d_mean = (df.Dur[(date,sta,comp)] + df.Dur[(date,sta,'E')] + df.Dur[(date,sta,'Z')])/3.
        po_mean = int((df.Ponset[(date,sta,comp)] + df.Ponset[(date,sta,'E')] + df.Ponset[(date,sta,'Z')])/3)
        list_files = [file,file.replace("N.D","E.D"),file.replace("N.D","Z.D")]
        rect, plan, az, iang = polarization_analysis(list_files,d_mean,po_mean,plot=False)
        if 'Rectilinearity' in list_features:
          df.Rectilinearity[(date,sta,'Z')], df.Rectilinearity[(date,sta,'N')], df.Rectilinearity[(date,sta,'E')] = rect, rect, rect
        if 'Planarity' in list_features:
          df.Planarity[(date,sta,'Z')], df.Planarity[(date,sta,'N')], df.Planarity[(date,sta,'E')] = plan, plan, plan
        if list_features or 'Azimuth':
          df.Azimuth[(date,sta,'Z')], df.Azimuth[(date,sta,'N')], df.Azimuth[(date,sta,'E')] = az, az, az
        if 'Incidence' in list_features:
          df.Incidence[(date,sta,'Z')], df.Incidence[(date,sta,'N')], df.Incidence[(date,sta,'E')] = iang, iang, iang

  if save:
    print "Features written in %s"%opt.opdict['feat_filename']
    df.to_csv(opt.opdict['feat_filename'])

# ================================================================

def extract_norm_features(list_features,date,file,dic):

    """
    Extraction of all features given by list_features, except hash 
    table values.
    """

    s = SeismicTraces(file,utcdatetime.UTCDateTime(str(date)))
    list_attr = s.__dict__.keys()
    if 'tr_grad' not in list_attr:
      for feat in list_features:
        dic[feat] = np.nan
      return dic
    # Determine P-onset
    s.duration()
    #s.display_traces()
    #s.amplitude_distribution()

    if len(list_attr) > 5:

      from waveform_features import spectrogram
      s, dic['MeanPredF'], dic['TimeMaxSpec'], dic['NbPeaks'], dic['Width'], hob, vals, dic['sPredF'] = spectrogram(s,plot=False)
      for i in range(len(vals)):
        dic['v%d'%i] = vals[i]
      dic['Dur'] = s.dur

      # Compute the spectrum
      s.spectrum(plot=False)

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
        dic['Ene%d-%d'%(f1,f2)] = energy_between_10Hz_and_30Hz(s.tr[s.i1:s.i2]/np.max(s.tr[s.i1:s.i2]),s.dt,wd=f1,wf=f2,ponset=s.ponset-s.i1,tend=s.tend-s.i1)

      if 'Ene5-10' in list_features:
        f1, f2 = 5,10
        dic['Ene%d-%d'%(f1,f2)] = energy_between_10Hz_and_30Hz(s.tr[s.i1:s.i2]/np.max(s.tr[s.i1:s.i2]),s.dt,wd=f1,wf=f2,ponset=s.ponset-s.i1,tend=s.tend-s.i1)

      if 'Ene0-5' in list_features:
        f1, f2 = .5,5
        dic['Ene%d-%d'%(f1,f2)] = energy_between_10Hz_and_30Hz(s.tr[s.i1:s.i2]/np.max(s.tr[s.i1:s.i2]),s.dt,wd=f1,wf=f2,ponset=s.ponset-s.i1,tend=s.tend-s.i1)

      if 'RappMaxMean' in list_features:
        # Max over mean ratio of the envelope
        from waveform_features import max_over_mean
        dic['RappMaxMean'] = max_over_mean(s.tr[s.ponset:s.tend])

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
        sk = skewness(s.tr_env[s.ponset:s.tend])
        dic['Skewness'] = sk
        #s.amplitude_distribution()

      if 'Kurto' in list_features: 
        # Kurtosis
        from waveform_features import kurtosis_envelope
        k = kurtosis_envelope(s.tr_env[s.ponset:s.tend])
        dic['Kurto'] = k

      if ('F_low' in list_features) or ('F_up' in list_features):
        # Lowest and highest frequency for kurtogram analysis
        from waveform_features import kurto_bandpass
        dic['F_low'],dic['F_up'] = kurto_bandpass(s,plot=True)

      if 'Centroid_time' in list_features:
        # Centroid time
        from waveform_features import centroid_time
        dic['Centroid_time'] = centroid_time(s.tr[s.i1:s.i2],s.dt,s.TF,s.ponset-s.i1,tend=s.tend-s.i1,plot=False)

      if 'RappMaxMeanTF' in list_features:
        # Max over mean ratio of the amplitude spectrum
        from waveform_features import max_over_mean
        dic['RappMaxMeanTF'] = max_over_mean(np.abs(s.TF[:len(s.TF)/2]))

      if 'IFslope' in list_features:
        # Average of the instantaneous frequency and slope of the unwrapped instantaneous phase
        from waveform_features import instant_freq
        #p,pf = instant_freq(s.tr[s.i1:s.i2],s.dt,s.TF,plot=False)
        #dic['IFslope'] = np.mean((p,pf[len(pf)-1]))
        vals, dic['IFslope'] = instant_freq(s.tr[s.i1:s.i2],s.dt,s.TF,s.ponset-s.i1,s.tend-s.i1,plot=False)
        for i in range(len(vals)):
          dic['if%d'%i] = vals[i]

      if ('ibw0' in list_features) or ('Norm_envelope' in list_features):
        # Average of the instantaneous frequency and normalized envelope
        from waveform_features import instant_bw
        vals, Nenv = instant_bw(s.tr[s.i1:s.i2],s.tr_env[s.i1:s.i2],s.dt,s.TF,s.ponset-s.i1,s.tend-s.i1,plot=False)
        dic['Norm_envelope'] = Nenv
        for i in range(len(vals)):
          dic['ibw%d'%i] = vals[i]

      if ('PredF' in list_features) or ('CentralF' in list_features) or ('Bandwidth' in list_features):
        # Spectral attributes
        from waveform_features import around_freq
        dic['PredF'], dic['Bandwidth'], dic['CentralF'] = around_freq(s.TF,s.freqs,plot=False)

      if 'Cepstrum' in list_features:
        # Cepstrum
        from waveform_features import cepstrum
        #dic['Cepstrum'] = cepstrum(s.TF,s.freqs,plot=True)
        cep = cepstrum(s.TF,s.freqs,plot=False)

      dic['Ponset'] = s.ponset
      return dic

# ================================================================

def hob(hobs,y_train,y_test,x_train,x_test):

  med = {}
  for i in range(len(np.unique(y_test))):
    a = y_train[y_train.EventType==i].index
    for j in range(len(hobs.keys())):
      print type(hobs[j])
      c = np.array(hobs[j])
      med[(i,j)] = np.median(c[a],axis=0)

  for i in range(len(np.unique(y_test))):
    corr = []
    for j in range(len(hobs.keys())):
      for k in range(len(np.unique(y_test))):
        a = y_test[y_test.EventType==k].index
        corr.append(np.corrcoef(med[i,j],hobs[j][a]))
      print corr
      raw_input("Pause")

# ================================================================

def extract_hash_features(list_features,date,file,dic,permut_file,plot=False):
  """
  Extracts hash table values.
  """
  from fingerprint_functions import FuncFingerprint, spectrogram, ponset_grad, vec_compute_signature, LSH

  s = SeismicTraces(file,utcdatetime.UTCDateTime(str(date)))
  list_attr = s.__dict__.keys()
  if 'tr_grad' not in list_attr:
    for feat in list_features:
      dic[feat] = np.nan
    return dic

  full_tr = s.tr
  grad = s.tr_grad
  q = [100.,.8,1]
  (full_spectro,f,full_time,end) = spectrogram(full_tr,param=q)
  ponset = ponset_grad(full_tr,grad,full_time,plot=plot)
  
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
  read_data_for_features_extraction(save=False)
