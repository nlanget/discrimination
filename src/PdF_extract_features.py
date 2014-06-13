#!/usr/bin/env python
# encoding: utf-8

import os,glob,sys
from obspy.core import read, utcdatetime
import numpy as np
import pandas as pd
from waveform_features import *
import matplotlib.pyplot as plt
from scipy.io import savemat
from math import isnan
from logistic_reg import comparison
from waveloc.locations_trigger import read_locs_from_file

# ================================================================

class SeismicTraces():

  def __init__(self,filename,origin=None):
    self.t0 = origin
    self.tr = self.read_file(filename)

    if self.tr:
      self.sta = self.tr.stats.station

      datadir = os.path.dirname(filename)

      filename = os.path.join(datadir,"*%s*.filt.*"%self.sta)
      self.tr_filt = self.read_file(filename)

      filename = os.path.join(datadir,"*%s*kurt_grad.*"%self.sta)
      self.tr_grad = self.read_file(filename)

      filename = os.path.join(datadir,"*%s*env.*"%self.sta)
      self.tr_env = self.read_file(filename)

      self.duration()
      self.spectro = self.spectrogram(plot=False)

      if os.path.isdir(os.path.join(datadir,'HHE')) and os.path.isdir(os.path.join(datadir,'HHN')):
        filename = os.path.join(datadir,"HHE/*%s*.filt.sac"%self.sta)
        self.tr_e = self.read_file(filename)
        filename = os.path.join(datadir,"HHN/*%s*.filt.sac"%self.sta)
        self.tr_n = self.read_file(filename)
        marker = 1
      else:
        marker = 0
    
      # Compute the spectrum
      self.spectrum(plot=False)
  
      # Compute the features 
      from waveform_features import max_over_mean, energy_between_10Hz_and_30Hz, skewness
      self.rapp = max_over_mean(self.tr_filt.data[self.i1:self.i2])
      self.energy = energy_between_10Hz_and_30Hz(self.tr.data[self.i1:self.i2],self.tr.stats.delta)
      self.sk = skewness(self.tr_env.data[self.i1:self.i2])

      self.spectral_attributes()
      self.complex_attributes()
      self.kurto()
 
      if marker:
        self.polarization_attributes()

      #self.display_traces()


  def read_file(self,filename):
    if self.t0:
      st = read(filename,starttime=self.t0-15,endtime=self.t0+135)
      self.i1,self.i2 = 1400,2400
    else:
      st = read(filename)
      self.i1,self.i2 = 0,len(st[0])-1
    if len(st) > 0:
      tr = st[0]
      return tr
    else:
      return None


  def spectrum(self,plot=False):
    data = self.tr_filt.data[self.i1:self.i2]
    self.TF = np.fft.fft(data)
    self.freqs = np.fft.fftfreq(len(self.TF),d=self.tr.stats.delta)
    if plot:
      fig = plt.figure()
      fig.set_facecolor('white')
      ax1 = fig.add_subplot(211)
      ax1.plot(data,'k')
      ax1.set_xlabel('Time')
      ax2 = fig.add_subplot(212,title='Amplitude spectrum')
      ax2.plot(self.freqs[:len(self.freqs)/2],np.abs(self.TF[:len(self.TF)/2]),'k')
      ax2.set_xlabel('Frequency (Hz)')
      ax2.set_ylabel('Amplitude')
      fig.suptitle(self.tr.stats.station)
      plt.show()


  def spectral_attributes(self):
    from waveform_features import around_freq, cepstrum
    self.predf, self.bandw, self.centralf, self.rmsf = around_freq(self.TF,self.freqs,plot=False)

    # Cepstral coefficients
    data = self.tr.data[self.i1:self.i2]
    TF = np.fft.fft(data)
    f = np.fft.fftfreq(len(TF),d=self.tr.stats.delta)
    self.cepst = cepstrum(TF,f,plot=False)
    #self.cepst = cepstrum(self.TF,self.freqs,plot=False)

    # Cepstral coefficients of noise
    from obspy.signal.filter import bandpass
    data = bandpass(self.tr.data[self.i1:self.i2],0,5,1./self.tr.stats.delta)
    TF_n = np.fft.fft(data)
    f_n = np.fft.fftfreq(len(TF_n),d=self.tr.stats.delta)
    self.cepstr_n = cepstrum(TF_n,f_n,plot=False)

#    fig = plt.figure()
#    fig.set_facecolor('white')
#    ax1 = fig.add_subplot(311)
#    ax1.plot(self.tr.data[self.i1:self.i2])
#    ax2 = fig.add_subplot(312) 
#    ax2.plot(self.tr_filt.data[self.i1:self.i2])
#    ax3 = fig.add_subplot(313)
#    ax3.plot(data)
#    plt.show()


  def spectrogram(self,plot=False):
    """
    From obspy.imaging.spectrogram.spectrogram.py
    """
    from matplotlib import mlab
    Fs = 1./self.tr.stats.delta
    wlen = Fs / 100.
    istart = self.ponset-100
    iend = self.ponset+1500
    data = self.tr.data[istart:iend] - self.tr.data[istart:iend].mean()
    npts = len(data)
    nfft = int(nearest_pow2(wlen*Fs))
    if nfft > npts:
      nfft = int(nearest_pow2(npts/8.))
    nlap = int(nfft*.9)
    end = npts/Fs

    mult = int(nearest_pow2(8))
    mult = mult*nfft

    specgram, f, time = mlab.specgram(data,Fs=Fs,NFFT=nfft,pad_to=mult,noverlap=nlap)
    specgram = np.flipud(specgram)


    # Plot
    if plot and self.tr.stats.station == 'UV05':
      halfbin_time = (time[1] - time[0]) / 2.0
      halfbin_freq = (f[1] - f[0]) / 2.0
      extent = (time[0] - halfbin_time, time[-1] + halfbin_time, f[0] - halfbin_freq, f[-1] + halfbin_freq)
      fig = plt.figure()
      fig.set_facecolor('white')
      ax1 = fig.add_subplot(211)
      ax1.plot(data,'k')
      ax1.plot([self.ponset-istart,self.ponset-istart],[np.min(data),np.max(data)],'r',lw=2.)
      ax1.set_xticklabels('')
      ax2 = fig.add_subplot(212)
      ax2.imshow(specgram,interpolation="nearest",cmap=plt.cm.jet,extent=extent)
      ax2.set_xlim([0,end])
      ax2.axis('tight')
      ax2.set_xlabel('Time (s)')
      ax2.set_ylabel('Frequency (Hz)')
      fig.suptitle(self.tr.stats.station)
      plt.close()

      if self.tr.stats.station == 'UV05':
        fig = plt.figure()
        fig.set_facecolor('white')
        plt.imshow(specgram,interpolation="nearest",cmap=plt.cm.jet,extent=extent,vmin=0,vmax=10**6)      
        plt.xlim([0,end])
        #plt.colorbar()
        plt.axis('tight')
        #plt.show()

    return specgram


  def complex_attributes(self):
    from waveform_features import instant_freq, instant_bw, centroid_time
    data = self.tr_filt.data[self.i1:self.i2] 
    self.ifreq = instant_freq(data,self.tr.stats.delta,self.TF,plot=False)
    self.ibw, self.normenv = instant_bw(data,self.tr.stats.delta,self.TF,plot=False)
    self.c = centroid_time(data,self.tr.stats.delta,self.TF,plot=False)


  def duration(self):
    from waveform_features import signal_duration, growth_over_decay
    self.ponset, self.tend, self.dur = signal_duration(self.tr_filt,self.tr_grad,self.tr_env,plot_one=False)
    self.p = growth_over_decay(self.tr_env,self.ponset,self.tend)


  def kurto(self):
    from waveform_features import kurto_bandpass, kurtosis_envelope
    self.fl, self.fu = kurto_bandpass(self.tr,plot=False)
    self.k = kurtosis_envelope(self.tr_filt.data[self.i1:self.i2],self.tr_env[self.i1:self.i2])


  def polarization_attributes(self):
    from waveform_features import polarization_analysis
    istart,iend = self.ponset-10, self.ponset+30
    x = np.array([self.tr_filt.data[istart:iend],self.tr_n.data[istart:iend],self.tr_e.data[istart:iend]])
    cov = np.cov(x)
    self.rect, self.plan, self.lambda_max = polarization_analysis(cov,plot=True)
    self.plot_particle_motion(istart,iend)


  def display_traces(self):
    fig = plt.figure()
    fig.set_facecolor('white')

    ax1 = fig.add_subplot(411,title='Unfiltered')
    ax1.plot(self.tr,'k')
    ax1.set_xticklabels('')

    ax2 = fig.add_subplot(412,title='Filtered')
    ax2.plot(self.tr_filt,'k')
    ax2.plot([self.ponset,self.ponset],[np.min(self.tr_filt.data),np.max(self.tr_filt.data)],'g',lw=2)
    ax2.plot([self.tend,self.tend],[np.min(self.tr_filt.data),np.max(self.tr_filt.data)],'r',lw=2)
    ax2.set_xticklabels('')

    ax3 = fig.add_subplot(413,title='Envelope')
    ax3.plot(self.tr_env,'k')
    ax3.set_xticklabels('')

    ax4 = fig.add_subplot(414,title='Kurtosis gradient')
    ax4.plot(self.tr_grad,'k')
    fig.suptitle(self.sta)
    plt.show()


  def plot_particle_motion(self,istart,iend):
    fig = plt.figure()
    fig.set_facecolor('white')

    ax1 = fig.add_subplot(311,title='Comp Z')
    ax1.plot(self.tr_z.data,'k')
    ax1.plot(self.tr_z.data[istart:iend],'r')
    ax1.set_xticklabels('')

    ax2 = fig.add_subplot(312,title='Comp N')
    ax2.plot(self.tr_n.data,'k')
    ax2.plot(self.tr_n.data[istart:iend],'r')
    ax2.set_xticklabels('')

    ax3 = fig.add_subplot(313,title='Comp E')
    ax3.plot(self.tr_e.data,'k')
    ax3.plot(self.tr_e.data[istart:iend],'r')

    fig = plt.figure()
    fig.set_facecolor('white')
    ax1 = fig.add_subplot(221)
    ax1.plot(tr_n.data[istart:iend],tr_z.data[istart:iend],'k')
    ax1.set_xlabel('N')
    ax1.set_ylabel('Z')

    ax2 = fig.add_subplot(222)
    ax2.plot(tr_n.data[istart:iend],tr_e.data[istart:iend],'k')
    ax2.set_xlabel('N')
    ax2.set_ylabel('E')

    ax3 = fig.add_subplot(223)
    ax3.plot(tr_z.data[istart:iend],tr_e.data[istart:iend],'k')
    ax3.set_xlabel('Z')
    ax3.set_ylabel('E')
  
    fig.suptitle(tr_z.stats.station)
    plt.show()


def nearest_pow2(x):
  """
  From obspy.imaging.spectrogram.spectrogram.py
  """
  from math import pow,ceil,floor
  a = pow(2,ceil(np.log2(x)))
  b = pow(2,floor(np.log2(x)))
  if abs(a-x) < abs(b-x):
    return a
  else:
    return b
# ================================================================
def read_mat_file(matfile):
  from scipy.io.matlab import mio
  mat=mio.loadmat(matfile)
  for key in sorted(mat):
    if key[0] == '_':
      del mat[key]
    else:
      mat[key]=mat[key].ravel()
  return mat
# ================================================================
def plot_histo_feat(dic,save=False):
  for feat in sorted(dic):
    vec=dic[feat]
    print feat, np.min(vec), np.max(vec)

    fig=plt.figure()
    fig.set_facecolor('white')
    plt.hist(vec,25)
    plt.title(feat)
    if save:
      plt.savefig('../results/Piton/%s.png'%feat)
  plt.show()
# ================================================================
def extract_features(plot=False,plot_histo=False):

  waveloc_path = os.getenv('WAVELOC_PATH')

  # m is True if use of coda exponential fit for signal duration computation
  m = True 

  outdir = os.path.join(waveloc_path,'out')
  locfile = os.path.join(outdir,'Piton/2010-10-14/loc/locations.dat')
  #locfile = os.path.join(outdir,'2010-10-14/loc/list_EB_ovpf')
  datadir = os.path.join(waveloc_path,'data/Piton/2010-10-14')

  datafiles = glob.glob(os.path.join(datadir,'*.mseed_1day'))
  datafiles.sort()

  if os.path.basename(locfile) == 'locations.dat':
    locs = read_locs_from_file(locfile)
  else:
    with open(locfile,'r') as test:
      lines=test.readlines()
      test.close()
    locs=[]
    for line in lines:
      loc={}
      loc['o_time']=utcdatetime.UTCDateTime(line.split(',')[0])
      locs.append(loc)

  dic = {}
  list_features = ['AsDec','Dur','Ene','Kurto','RappMaxMean','F_low','F_up','Skewness','PredF','Bandwidth','CentralF','Centroid_time','Norm_envelope','Spectrogram']
  for key in list_features:
    dic[key] = np.array([])
  for iloc,loc in enumerate(locs):
    origin = loc['o_time']
    print origin

    asdec_list, dur_list, en_list, klist, rlist, sk_list = [],[],[],[],[],[]
    predofreqs,bw_list,centrfreqs = [],[],[]
    fl_list, fu_list = [],[]
    clist, cov_list = [],[]
    nenv_list = []
    cepstr_noise, cepstr_sig = np.array([]),np.array([])
    spec = np.array([])

    iceps,ispec = 0,0
    for file in datafiles:
      s = SeismicTraces(file,origin)
      list_attr = s.__dict__.keys()

      if len(list_attr) > 2:

        if 'energy' in list_attr:
          en_list.append(s.energy)  # Energy between 10 and 30 Hz

        if 'rapp' in list_attr:
          rlist.append(s.rapp)  # Max over mean ratio

        if 'dur' in list_attr:
          if s.dur > 0:
            dur_list.append(s.dur)  # Duration
            if s.p > 0:
              asdec_list.append(s.p)  # AsDec

        if 'sk' in list_attr:
          sk_list.append(s.sk)  # Skewness

        if 'cov' in list_attr:
          cov_list.append(s.cov)  # Covariance matrices

        if 'predf' in list_attr:
          predofreqs.append(s.predf)  # Predominant frequency

        if 'bandw' in list_attr:
          bw_list.append(s.bandw)  # Frequency bandwidth

        if 'centralf' in list_attr:
          centrfreqs.append(s.centralf) # Central frequency

        if 'k' in list_attr:
          klist.append(s.k)  # Kurtosis

        if 'fl' in list_attr:
          fl_list.append(s.fl)  # Lowest frequency for kurtogram analysis
          fu_list.append(s.fu)  # Highest frequency for kurtogram analysis

        if 'c' in list_attr:
          clist.append(s.c)  # Centroid time

        if 'normenv' in list_attr:
          nenv_list.append(s.normenv)  # Normalized envelope

        if 'cepst' in list_attr:
          if cepstr_sig.all() and cepstr_sig.any():
            if iceps == 0:
              cepstr_noise = np.append(cepstr_noise,s.cepstr_n)
              cepstr_sig = np.append(cepstr_sig,s.cepst)
            else:
              cepstr_noise = np.vstack((cepstr_noise,s.cepstr_n))
              cepstr_sig = np.vstack((cepstr_sig,s.cepst))
            iceps += 1

        if 'spectro' in list_attr:
          if s.spectro.any() and s.spectro.all():
            if ispec == 0:
              if iloc == 0:
                dic['__dim_spectro__'] = s.spectro.shape
                spec = s.spectro
                ispec = 1
              elif s.spectro.shape == dic['__dim_spectro__']:
                spec = s.spectro
                ispec = 1
            else:
              if s.spectro.shape == dic['__dim_spectro__']:
                spec = spec + s.spectro
                ispec += 1

    asdec = np.mean(asdec_list)
    duration = np.mean(dur_list)
    energy = np.mean(en_list)
    kurto = np.mean(klist)
    rapp = np.mean(rlist)
    skew = np.mean(sk_list)
    f_low = np.mean(fl_list)
    f_up = np.mean(fu_list)
    predominant_f = np.mean(predofreqs)
    bandwidth = np.mean(bw_list)
    central_f = np.mean(centrfreqs)
    centroid_time = np.mean(clist)
    norm_env = np.mean(nenv_list)
    if spec.any() and spec.all():
      spectrogram = spec / ispec
    if cov_list:
      cov = np.mean(cov_list)
      polarization_analysis(cov,plot=True)
    sce = np.sum(np.abs(cepstr_noise - 1./len(datafiles)*np.sum(cepstr_sig,axis=0))**2,axis=0)
    scb = 1./len(datafiles) * np.abs(np.sum(cepstr_noise,axis=0))**2
    F2_2 = (len(datafiles)-1) * scb/sce

    #plot=True
    if plot:
      fig = plt.figure()
      fig.set_facecolor('white')
      plt.plot(F2_2,'k')
      plt.xlabel('Quefrency')
      plt.ylabel('Amplitude of F(2,2)')
      plt.show()

    if not isnan(asdec) and not isnan(duration) and not isnan(energy) and not isnan(kurto) and not isnan(rapp) and not isnan(skew) and not isnan(f_low) and not isnan(f_up) and not isnan(predominant_f) and not isnan(bandwidth) and not isnan(central_f) and not isnan(centroid_time) and not isnan(norm_env) and duration > 0 and spectrogram.any() and spectrogram.all():
      dic['AsDec'] = np.append(dic['AsDec'],asdec)
      dic['Dur'] = np.append(dic['Dur'],duration)
      dic['Ene'] = np.append(dic['Ene'],energy)
      dic['Kurto'] = np.append(dic['Kurto'],kurto)
      dic['RappMaxMean'] = np.append(dic['RappMaxMean'],rapp)
      dic['Skewness'] = np.append(dic['Skewness'],skew)
      dic['F_low'] = np.append(dic['F_low'],f_low)
      dic['F_up'] = np.append(dic['F_up'],f_up)
      dic['PredF'] = np.append(dic['PredF'],predominant_f) 
      dic['Bandwidth'] = np.append(dic['Bandwidth'],bandwidth)
      dic['CentralF'] = np.append(dic['CentralF'],central_f)
      dic['Centroid_time'] = np.append(dic['Centroid_time'],centroid_time)
      dic['Norm_envelope'] = np.append(dic['Norm_envelope'],norm_env)
      if iloc == 0:
        dic['Spectrogram'] = np.append(dic['Spectrogram'],spectrogram)
      else:
        dic['Spectrogram'] = np.vstack((dic['Spectrogram'],spectrogram.ravel()))
      print dic['Kurto'], type(dic['Kurto']), dic['Kurto'].shape
      raw_input("P")

  savemat('set_14oct_test23may.mat',dic,oned_as='row')

  if plot_histo:
    plot_histo_feat(dic,save=True)
# ================================================================
def exp_fit(tr_env, tmax, stop, noise, plot=False):
  coda = tr_env.data[tmax+1400:]
  N = len(coda)
  if stop !=0 and stop-(tmax+1400) > 200:
    coda = tr_env.data[tmax+1400:stop]
    #coda = np.append(coda,noise*np.ones(N-len(coda)))
  #print len(coda)
  from scipy.optimize import curve_fit
  vec = np.linspace(0,1,(len(coda)))
  try:
    popt, pcov = curve_fit(func,vec,coda,maxfev=5000)
    #vec = np.linspace(0,1,(len(tr_env.data[tmax+1400:])))
    if plot:
      vec = np.linspace(0,1,(len(tr_env.data[tmax+1400:])))
      plt.plot(range(tmax-1300,len(tr_filt.data)),func(vec, *popt), 'r-', label="Fitted Curve")
      plt.legend()
    return popt, vec

  except RuntimeError:
    print "Number of calls exceeded (station %s). Continue anyway..."%tr_env.stats.station
    return None, None
# ================================================================
def func(x, a, b, c):
  return a * np.exp(-b * x) + c
# ================================================================
def reg_log():

  from PdF_log_reg import *
  training_set = read_mat_file('training_set_EB.mat')
  test_set = read_mat_file('test_set_14oct.mat') 
  del training_set['Spectrogram']
  del test_set['Spectrogram']

  list_features = ['AsDec','Dur','Kurto','RappMaxMean','Skewness','F_low','F_up','PredF','Bandwidth','CentralF']
  #list_features = ['Kurto','RappMaxMean']
  x = pd.DataFrame(training_set).reindex(columns=list_features)
  x_data = pd.DataFrame(test_set).reindex(columns=list_features)

  nb_eb_train = x.shape[0]
  nb_vt_train = 50
 
  x = x.append(x_data[:nb_vt_train],ignore_index=True)
  print x.shape
  print x_data.shape
  # EB = 1 -------------------- VT = 0
  y=pd.DataFrame(np.concatenate((np.zeros(nb_eb_train,dtype=int),np.ones(nb_vt_train,dtype=int))))

  print "********* Original data **********"
  LR_train,theta,LR_test = do_all_logistic_regression(x,y,x_data,output=True)
  print_results(y,LR_train,LR_test)
  plot_result(x,x_data,y[1],LR_test,theta,norm=True)
  print y.values[:,0]
  print LR_train

  # PCA
  # Identifies the combination of features with highest variance
  print "********** PCA **********"
  x_pca, x_data_pca = implement_pca(x,x_data)
  LR_train,theta,LR_test = do_all_logistic_regression(x_pca,y,x_data_pca,norm=False,output=True)
  print_results(y,LR_train,LR_test)
  plot_result(x_pca,x_data_pca,y[1],LR_test,theta)

  nbc = x_pca.shape[1]

  # SparsePCA
  # Identifies the combination of sparse features that best reconstructs the data
  print "********** Sparse PCA **********"
  x_spca, x_data_spca = implement_sparse_pca(x,x_data,nbc)
  LR_train,theta,LR_test = do_all_logistic_regression(x_spca,y,x_data_spca,norm=False,output=True)
  print_results(y,LR_train,LR_test)
  plot_result(x_spca,x_data_spca,y[1],LR_test,theta)
 
  # ICA
  # Identifies the combination of features with highest non-gaussianity
  print "********** ICA **********"
  x_ica, x_data_ica = implement_ica(x,x_data,nbc)
  LR_train,theta,LR_test = do_all_logistic_regression(x_ica,y,x_data_ica,norm=False,output=True)
  print_results(y,LR_train,LR_test)
  plot_result(x_ica,x_data_ica,y[1],LR_test,theta)

  # KMeans
  # Unsupervised learning by clustering
  print "*********** KMean ***********"
  print "	** On original data"
  y_kmean_training, y_kmean = implement_kmean(x,x_data)
  print_results(y,y_kmean_training,y_kmean)
  plot_clustering(x,x_data,y_kmean_training,y_kmean)

  print "\n	** On data after PCA"
  y_kmean_train_pca, y_kmean_pca = implement_kmean(x_pca,x_data_pca)
  print_results(y,y_kmean_train_pca,y_kmean_pca)

  # MeanShift
  # Unsupervised learning by clustering
  print "*********** MeanShift ***********"
  y_ms_train, y_ms = implement_mean_shift(x,x_data)
  print_results(y,y_ms_train,y_ms)
  plot_clustering(x,x_data,y_ms_train,y_ms)

  from PdF_log_reg import plot_histo 
  #plot_histo(x,x_data,nb_eb_train,nb_vt_train,savefig=False)

  plt.show()

  # Combine Kurto and RappMaxMean into one feature (Krapp)
  #if x.has_key('Kurto') and x.has_key('RappMaxMean'):
  #  x, x_data = make_uncorrelated_data(x,x_data,['Kurto','RappMaxMean'],'KRapp')
  #if x.has_key('F_low') and x.has_key('F_up'):
  #  x, x_data = make_uncorrelated_data(x,x_data,['F_low','F_up'],'F_band')
# ================================================================
def plot_result(x,x_data,y,result,theta,norm=False):
  nb_eb = len(np.where(y==0)[0])
  nb_vt = len(np.where(y==1)[0])

  if norm:
    from logistic_reg import normalize
    x,x_data = normalize(x,x_data)

  if x.shape[1] == 1:
    from PdF_log_reg import hypothesis_function, plot_one_feature
    VT = x_data.reindex(index=np.where(result==1)[0])
    EB = x_data.reindex(index=np.where(result==0)[0])
    syn, hyp = hypothesis_function(x.min(),x.max(),theta[1])
    plot_one_feature(x,nb_eb,nb_vt,syn,hyp,VT,EB,[])

  if x.shape[1] == 2:
    from plot_func import plot_db
    plot_db(x,y,theta[1])
    plot_db(x_data,result,theta[1])

  if x.shape[1] == 3:
    from plot_func import plot_db_3d
    plot_db_3d(x,y,theta[1])
# ================================================================
def plot_clustering(x,x_data,y_train,y_clus):

  if x.shape[1] >= 2:
    fig = plt.figure()
    fig.set_facecolor('white')
    plt.scatter(x.values[:,0],x.values[:,1],c=y_train,cmap=plt.cm.gray)
    plt.title('Training set')
    plt.xlabel(x.columns[0])
    plt.ylabel(x.columns[1])

    fig = plt.figure()
    fig.set_facecolor('white')
    plt.scatter(x_data.values[:,0],x_data.values[:,1],c=y_clus,cmap=plt.cm.gray)
    plt.title('Test set')
    plt.xlabel(x_data.columns[0])
    plt.ylabel(x_data.columns[1])
# ================================================================
def print_results(y,LR_class_train, LR_class):
  nb_eb_train = len(np.where(y==0)[0])
  nb_vt_train = len(np.where(y==1)[0])
  print "Classification of the training set:"
  print "True #VT:", nb_vt_train
  print "True #EB:", nb_eb_train
  nb_eb_bad, nb_vt_bad,nb_vt_good = comparison(LR_class_train,y[1])
  perc_vt = float(nb_vt_good)/nb_vt_train*100
  perc_eb = 100-float(nb_eb_bad)/nb_eb_train*100
  text = [perc_vt,perc_eb,0,0]
  print "#VT:%d, success rate = %.2f%%"%(len(np.where(LR_class_train==1)[0]),perc_vt)
  print "#EB:%d, success rate = %.2f%%"%(len(np.where(LR_class_train==0)[0]),perc_eb)
 
  print "\nClassification of the test set:"
  print "# VT:",len(np.where(LR_class==1)[0])
  print "# EB:",len(np.where(LR_class==0)[0])
# ================================================================
def implement_pca(x,x_data):
  from sklearn.decomposition import PCA
  pca = PCA(whiten=True)
  pca.fit(x.values)
  print pca.explained_variance_ratio_
  nbc = len(np.where(pca.explained_variance_ratio_ >= 0.05)[0])

  pca = PCA(n_components=nbc,whiten=True)
  x_pca = pca.fit(x.values).transform(x.values)
  x_data_pca = pca.transform(x_data.values)
  print x_pca.shape

  return pd.DataFrame(x_pca), pd.DataFrame(x_data_pca)
# ================================================================
def implement_sparse_pca(x,x_data,nbc):
  from logistic_reg import normalize
  x, x_data = normalize(x,x_data)
  from sklearn.decomposition import SparsePCA
  Spca = SparsePCA(n_components=nbc)
  Spca.fit(x.values)
  x_spca = Spca.transform(x.values)
  x_data_spca = Spca.transform(x_data.values)
  print x_spca.shape
  return pd.DataFrame(x_spca), pd.DataFrame(x_data_spca)
# ================================================================
def implement_ica(x,x_data,nbc):
  from sklearn.decomposition import FastICA
  ica = FastICA(n_components=nbc,whiten=True)
  ica.fit(x.values)
  x_ica = ica.transform(x.values)
  x_data_ica = ica.transform(x_data.values)
  print x_ica.shape
  return pd.DataFrame(x_ica), pd.DataFrame(x_data_ica)
# ================================================================
def implement_kmean(x,x_data):
  from sklearn.cluster import KMeans
  kmean = KMeans(k=2)
  y_kmean_training = kmean.fit(x.values).predict(x.values)
  y_kmean = kmean.fit(x_data.values).predict(x_data.values)
  return y_kmean_training, y_kmean
# ================================================================
def implement_mean_shift(x,x_data):
  from sklearn.cluster import MeanShift,estimate_bandwidth
  bandwidth = estimate_bandwidth(x.values)
  ms = MeanShift(cluster_all=True)
  ms.fit(x.values)
  y_ms_training = ms.labels_
  ms = MeanShift(cluster_all=True)
  ms.fit(x_data.values)
  y_ms = ms.labels_
  return y_ms_training, y_ms
# ================================================================
if __name__ == '__main__' :

  extract_features(plot=False,plot_histo=False)
  reg_log()
