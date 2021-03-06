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

def process_envelope(tr,w=51):
  """
  Runs envelope processing on a waveform.
  w is the length of the sliding window
  """
  from obspy.signal import filter
  env = filter.envelope(tr)
  s = np.r_[env[w-1:0:-1],env,env[-1:-w:-1]]
  window = np.ones(w,'d')
  return np.convolve(window/window.sum(),s,mode='valid')[w/2:-w/2]


class SeismicTraces():

  """
  Class which reads the seismic traces associated with an event.
  Also computes some basic features such as the spectrum and the 
  envelope and determines the P-onset.
  """

  def __init__(self,mat,comp,train=None):
    self.read_file(mat,comp,train)
    if not 'tr_env' in self.__dict__.keys():
      self.tr_env = process_envelope(self.tr,w=101)
    self.dt = .01
    self.station = 'BOR'
    self.t0 = 0
    self.starttime = 0

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
      #self.tr_env = mat['%s_ENV'%kname][i]
      self.tr_grad = mat['%s_GRAD'%kname][i]
    else:
      df = pd.DataFrame(mat['SIG2'])
      if comp == 'Z':
        self.tr = mat['SIG2'][:,0][0][0]
      elif comp == 'E':
        self.tr = mat['SIG2'][:,1][0][0]
      elif comp == 'N':
        self.tr = mat['SIG2'][:,2][0][0]
      #self.tr_env = mat['SIG_ENV'].T[0]
      self.tr_grad = mat['SIG_GRAD'].T[0]


  def read_all_files(self,mat,train):
    if train:
      i,type = train[0],train[1]
      kname = 'Sig%s'%type
      self.tr_z = mat[kname][0][i][0][0][0]
      self.tr_e = mat[kname][0][i][1][0][0]
      self.tr_n = mat[kname][0][i][2][0][0]
    else:
      df = pd.DataFrame(mat['SIG2'])
      self.tr_z = mat['SIG2'][:,0][0][0]
      self.tr_e = mat['SIG2'][:,1][0][0]
      self.tr_n = mat['SIG2'][:,2][0][0]


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
      fig.suptitle(self.station)
      plt.show()


  def display_traces(self):
    """
    Displays the seismic traces: raw signal, envelope and kurtosis gradient
    """

    time = np.arange(len(self.tr)) * self.dt

    print len(time), len(self.tr), len(self.tr_env), len(self.tr_grad)

    fig = plt.figure()
    fig.set_facecolor('white')

    ax1 = fig.add_subplot(311,title='Raw data')
    ax1.plot(time,self.tr,'k')
    if 'ponset' in self.__dict__.keys():
      ax1.plot([time[self.ponset],time[self.ponset]],[np.min(self.tr),np.max(self.tr)],'r',lw=2)
      if 'dur' in self.__dict__.keys():
        ax1.plot([time[self.ponset]+self.dur,time[self.ponset]+self.dur],[np.min(self.tr),np.max(self.tr)],'b',lw=2)
    ax1.set_xlim([0,time[-1]])
    ax1.set_xticklabels('')

    ax2 = fig.add_subplot(312,title='Smoothed envelope')
    ax2.plot(time[:-1],self.tr_env,'k')
    if 'ponset' in self.__dict__.keys():
      ax2.plot([time[self.ponset],time[self.ponset]],[np.min(self.tr_env),np.max(self.tr_env)],'r',lw=2)
      emax = np.argmax(self.tr_env)
      ax2.plot([time[emax],time[emax]],[np.min(self.tr_env),np.max(self.tr_env)],'g',lw=2)
      if 'dur' in self.__dict__.keys():
        ax2.plot([time[self.ponset]+self.dur,time[self.ponset]+self.dur],[np.min(self.tr_env),np.max(self.tr_env)],'b',lw=2)
    ax2.set_xlim([0,time[-1]])
    ax2.set_xticklabels('')
 
    ax3 = fig.add_subplot(313,title='Kurtosis gradient')
    ax3.plot(time,self.tr_grad,'k')
    ax3.set_xlabel('Time (s)')
    ax3.set_xlim([0,time[-1]])
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

  if set == 'train':
    opt.opdict['feat_filename'] = '%s/features/%s'%(opt.opdict['outdir'],opt.opdict['feat_train'])
  print opt.opdict['feat_filename']

  if save:
    if os.path.exists(opt.opdict['feat_filename']):
      print "WARNING !! File %s already exists"%opt.opdict['feat_filename']
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

      counter = 0
      for comp in opt.opdict['channels']:
        counter = counter + 1
        ind = (numfile,'BOR',comp)
        dic = pd.DataFrame(columns=list_features,index=[ind])
        dic['EventType'] = tsort[tsort.Date==numfile].Type.values[0]
        dic['Ponset'] = 0

        s = SeismicTraces(mat,comp)
        list_attr = s.__dict__.keys()

        if len(list_attr) > 2:
          if opt.opdict['option'] == 'norm':
            dic = extract_norm_features(s,list_features,dic)
          elif opt.opdict['option'] == 'hash':
            if ifile in [409,1026,1027,1028,1993,2121,2122,2123,2424,2441,3029,3058,3735,3785,3852,3930,4200,4463,4464,4746,6150,6382,6672,6733]:
              continue
            dic = extract_hash_features(s,list_features,dic,opt.opdict['permut_file'],plot=False)
          df = df.append(dic)

        if counter == 3 and ('Rectilinearity' in list_features or 'Planarity' in list_features or 'Azimuth' in list_features or 'Incidence' in list_features):
          d_mean = (df.Dur[(numfile,'BOR',comp)] + df.Dur[(numfile,'BOR','E')] + df.Dur[(numfile,'BOR','Z')])/3.
          po_mean = int((df.Ponset[(numfile,'BOR',comp)] + df.Ponset[(numfile,'BOR','E')] + df.Ponset[(numfile,'BOR','Z')])/3)
          s.read_all_files(mat,False)
          rect, plan, az, iang = polarization_analysis(s,d_mean,po_mean,plot=False)
          if 'Rectilinearity' in list_features:
            df.Rectilinearity[(numfile,'BOR','Z')], df.Rectilinearity[(numfile,'BOR','N')], df.Rectilinearity[(numfile,'BOR','E')] = rect, rect, rect
          if 'Planarity' in list_features:
            df.Planarity[(numfile,'BOR','Z')], df.Planarity[(numfile,'BOR','N')], df.Planarity[(numfile,'BOR','E')] = plan, plan, plan
          if list_features or 'Azimuth':
            df.Azimuth[(numfile,'BOR','Z')], df.Azimuth[(numfile,'BOR','N')], df.Azimuth[(numfile,'BOR','E')] = az, az, az
          if 'Incidence' in list_features:
            df.Incidence[(numfile,'BOR','Z')], df.Incidence[(numfile,'BOR','N')], df.Incidence[(numfile,'BOR','E')] = iang, iang, iang

  elif set == 'train':
    datafile = os.path.join(opt.opdict['datadir'],'TrainingSetPlusSig_2.mat')
    mat = mio.loadmat(datafile)
    hob_all_EB = {}
    for i in range(mat['KurtoEB'].shape[1]):
      print "EB", i
      if i!=10 and i!=61:
        continue
      counter = 0
      for comp in opt.opdict['channels']:
        counter = counter + 1
        dic = pd.DataFrame(columns=list_features,index=[(i,'BOR',comp)])
        dic['EventType'] = 'EB'
        dic['Ponset'] = 0
        
        s = SeismicTraces(mat,comp,train=[i,'EB'])
        list_attr = s.__dict__.keys()
        if len(list_attr) > 2:
          if opt.opdict['option'] == 'norm':
            dic = extract_norm_features(s,list_features,dic)
          elif opt.opdict['option'] == 'hash':
            dic = extract_hash_features(s,list_features,dic,opt.opdict['permut_file'],plot=False)
          df = df.append(dic)

      neb = i+1
      if counter == 3 and ('Rectilinearity' in list_features or 'Planarity' in list_features or 'Azimuth' in list_features or 'Incidence' in list_features):
        d_mean = (df.Dur[(i,'BOR',comp)] + df.Dur[(i,'BOR','E')] + df.Dur[(i,'BOR','Z')])/3.
        po_mean = int((df.Ponset[(i,'BOR',comp)] + df.Ponset[(i,'BOR','E')] + df.Ponset[(i,'BOR','Z')])/3)
        s.read_all_files(mat,train=[i,'EB'])
        rect, plan, az, iang = polarization_analysis(s,d_mean,po_mean,plot=False)
        if 'Rectilinearity' in list_features:
          df.Rectilinearity[(i,'BOR','Z')], df.Rectilinearity[(i,'BOR','N')], df.Rectilinearity[(i,'BOR','E')] = rect, rect, rect
        if 'Planarity' in list_features:
          df.Planarity[(i,'BOR','Z')], df.Planarity[(i,'BOR','N')], df.Planarity[(i,'BOR','E')] = plan, plan, plan
        if 'Azimuth' in list_features:
          df.Azimuth[(i,'BOR','Z')], df.Azimuth[(i,'BOR','N')], df.Azimuth[(i,'BOR','E')] = az, az, az
        if 'Incidence' in list_features:
          df.Incidence[(i,'BOR','Z')], df.Incidence[(i,'BOR','N')], df.Incidence[(i,'BOR','E')] = iang, iang, iang


    for i in range(mat['KurtoVT'].shape[1]):
      print "VT", i+neb
      if i != 5:
        continue
      counter = 0
      for comp in opt.opdict['channels']:
        counter = counter + 1
        dic = pd.DataFrame(columns=list_features,index=[(i+neb,'BOR',comp)])
        dic['EventType'] = 'VT'
        dic['Ponset'] = 0

        s = SeismicTraces(mat,comp,train=[i,'VT'])

        list_attr = s.__dict__.keys()
        if len(list_attr) > 2:
          if opt.opdict['option'] == 'norm':
            dic = extract_norm_features(s,list_features,dic)
          elif opt.opdict['option'] == 'hash':
            dic = extract_hash_features(s,list_features,dic,opt.opdict['permut_file'],plot=False)
          df = df.append(dic)
      if counter == 3 and ('Rectilinearity' in list_features or 'Planarity' in list_features or 'Azimuth' in list_features or 'Incidence' in list_features):
        d_mean = (df.Dur[(i+neb,'BOR',comp)] + df.Dur[(i+neb,'BOR','E')] + df.Dur[(i+neb,'BOR','Z')])/3.
        po_mean = int((df.Ponset[(i+neb,'BOR',comp)] + df.Ponset[(i+neb,'BOR','E')] + df.Ponset[(i+neb,'BOR','Z')])/3)
        s.read_all_files(mat,train=[i,'VT'])
        rect, plan, az, iang = polarization_analysis(s,d_mean,po_mean,plot=False)
        if 'Rectilinearity' in list_features:
          df.Rectilinearity[(i+neb,'BOR','Z')], df.Rectilinearity[(i+neb,'BOR','N')], df.Rectilinearity[(i+neb,'BOR','E')] = rect, rect, rect
        if 'Planarity' in list_features:
          df.Planarity[(i+neb,'BOR','Z')], df.Planarity[(i+neb,'BOR','N')], df.Planarity[(i+neb,'BOR','E')] = plan, plan, plan
        if 'Azimuth' in list_features:
          df.Azimuth[(i+neb,'BOR','Z')], df.Azimuth[(i+neb,'BOR','N')], df.Azimuth[(i+neb,'BOR','E')] = az, az, az
        if 'Incidence' in list_features:
          df.Incidence[(i+neb,'BOR','Z')], df.Incidence[(i+neb,'BOR','N')], df.Incidence[(i+neb,'BOR','E')] = iang, iang, iang

  if save:
    print "Features written in %s"%opt.opdict['feat_filename']
    df.to_csv(opt.opdict['feat_filename'])

# ================================================================

def extract_norm_features(s,list_features,dic):

    """
    Extraction of all features given by list_features, except hash 
    table values.
    """

    list_attr = s.__dict__.keys()
    if 'tr_grad' not in list_attr:
      for feat in list_features:
        dic[feat] = np.nan
      return dic

    #s.display_traces()
    #s.amplitude_distribution()

    if len(list_attr) > 6:

      # Mean of the predominant frequency
      from waveform_features import spectrogram
      s, dic['MeanPredF'], dic['TimeMaxSpec'], dic['NbPeaks'], dic['Width'], hob, vals, dic['sPredF'] = spectrogram(s,plot=False)
      for i in range(len(vals)):
        dic['v%d'%i] = vals[i]
      dic['Dur'] = s.dur

      s.spectrum(plot=False)

      if 'Acorr' in list_features:
        from waveform_features import autocorrelation,filt_ratio
        vals = autocorrelation(s,plot=False)
        for i in range(len(vals)):
          dic['acorr%d'%i] = vals[i]

        vals = filt_ratio(s,plot=False)
        for i in range(len(vals)):
          dic['fratio%d'%i] = vals[i]

      if 'Ene10-30' in list_features:
       # Energy between 10 and 30 Hz
        from waveform_features import energy_between_10Hz_and_30Hz
        f1, f2 = 10,30
        dic['Ene%d-%d'%(f1,f2)] = energy_between_10Hz_and_30Hz(s.tr[s.i1:s.i2]/np.max(s.tr[s.i1:s.i2]),s.dt,wd=f1,wf=f2,ponset=s.ponset-s.i1,tend=s.tend-s.i1)

      if 'Ene5-10' in list_features:
        from waveform_features import energy_between_10Hz_and_30Hz
        f1, f2 = 5,10
        dic['Ene%d-%d'%(f1,f2)] = energy_between_10Hz_and_30Hz(s.tr[s.i1:s.i2]/np.max(s.tr[s.i1:s.i2]),s.dt,wd=f1,wf=f2,ponset=s.ponset-s.i1,tend=s.tend-s.i1)

      if 'Ene0-5' in list_features:
        from waveform_features import energy_between_10Hz_and_30Hz
        f1, f2 = .5,5
        dic['Ene%d-%d'%(f1,f2)] = energy_between_10Hz_and_30Hz(s.tr[s.i1:s.i2]/np.max(s.tr[s.i1:s.i2]),s.dt,wd=f1,wf=f2,ponset=s.ponset-s.i1,tend=s.tend-s.i1)

      if 'RappMaxMean' in list_features:
        # Max over mean ratio of the envelope
        from waveform_features import max_over_mean
        dic['RappMaxMean'] = max_over_mean(s.tr_env[s.ponset:s.tend])

      if 'AsDec' in list_features:
        # Ascendant phase duration over descendant phase duration
        from waveform_features import growth_over_decay
        p = growth_over_decay(s)
        if p > 0:
          dic['AsDec'] = p
        s.display_traces()

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
        dic['F_low'],dic['F_up'] = kurto_bandpass(s,plot=False)

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

      dic['Ponset'] = s.ponset
      return dic

# ================================================================

def polarization_analysis(S,dur,ponset,plot=False):
  """
  Returns:
  - the planarity
  - the rectilinearity
  - the azimuth
  - the incidence angle
  from Hammer et al. 2012
  """

  i1, i2 = ponset-50, ponset+200
  zz = S.tr_z[i1:i2]
  nn = S.tr_n[i1:i2]
  ee = S.tr_e[i1:i2]

  cov_mat = np.cov((zz,nn,ee))
  #cov_mat = np.cov((S.tr_z,S.tr_n,S.tr_e))
  vals, vecs = np.linalg.eig(cov_mat)

  vals_sort,vecs_sort = np.sort(vals)[::-1], np.array([])
  for i in range(len(vals)):
    ind = np.where(vals == vals_sort[i])[0]
    vecs_sort = np.append(vecs_sort,vecs[ind])
  vecs_sort = np.reshape(vecs_sort,vecs.shape)

  rect = 1 - (vals_sort[1]+vals_sort[2])/(2*vals_sort[0])
  plan = 1 - 2*vals_sort[2]/(vals_sort[1]+vals_sort[0])

  # Azimuth
  az = np.arctan(vecs_sort[1][0]*np.sign(vecs_sort[0][0])/(vecs_sort[2][0]*np.sign(vecs_sort[0][0])))
  # Angle of incidence
  iang = np.arccos(vecs_sort[0][0])
  # Conversion from radians to degrees
  from math import pi
  az = az*180/pi
  iang = iang*180/pi

  if plot:
    print rect, plan, az, iang

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    center = [0,0,0]
    # find the rotation matrix and radii of the axes
    U, s, rotation = np.linalg.svd(cov_mat)
    radii = 1.0/np.sqrt(s)

    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    for i in range(len(x)):
      for j in range(len(x)):
        [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], rotation) + center

    # Plot particle motion
    import matplotlib.gridspec as gridspec
    from mpl_toolkits.mplot3d import Axes3D
    G = gridspec.GridSpec(3,2)

    fig = plt.figure(figsize=(12,7))
    fig.set_facecolor('white')

    time = np.arange(0,len(S.tr_z)*S.dt,S.dt)

    gs1 = gridspec.GridSpec(3,1)
    ax1 = fig.add_subplot(gs1[0])
    ax1.plot(time,S.tr_z,'k')
    ax1.plot(time[range(i1,i2)],S.tr_z[i1:i2],'r')
    ax1.set_xlim([0,time[len(S.tr_z)-1]])
    ax1.set_title('Vertical component')
    ax1.set_xticklabels('')

    ax2 = fig.add_subplot(gs1[1])
    ax2.plot(time,S.tr_n,'k')
    ax2.plot(time[range(i1,i2)],S.tr_n[i1:i2],'r')
    ax2.set_xlim([0,time[len(S.tr_n)-1]])
    ax2.set_title('North component')
    ax2.set_xticklabels('')

    ax3 = fig.add_subplot(gs1[2])
    ax3.plot(time,S.tr_e,'k')
    ax3.plot(time[range(i1,i2)],S.tr_e[i1:i2],'r')
    ax3.set_xlim([0,time[len(S.tr_e)-1]])
    ax3.set_title('East component')
    ax3.set_xlabel('Time (s)')

    gs1.tight_layout(fig, rect=[None, None, 0.45, None])

    gs2 = gridspec.GridSpec(1,2)
    ax4 = fig.add_subplot(gs2[0,0])
    ax4.plot(S.tr_n[i1:i2],S.tr_z[i1:i2],'k')
    ax4.set_xlabel('N')
    ax4.set_ylabel('Z')
 
    ax5 = fig.add_subplot(gs2[0,1])
    ax5.plot(S.tr_n[i1:i2],S.tr_e[i1:i2],'k')
    ax5.set_xlabel('N')
    ax5.set_ylabel('E')

    gs2.tight_layout(fig, rect=[0.45, 0.50, None, None])

    gs3 = gridspec.GridSpec(1,2)
    ax6 = fig.add_subplot(gs3[0,0])
    ax6.plot(S.tr_z[i1:i2],S.tr_e[i1:i2],'k')
    ax6.set_xlabel('Z')
    ax6.set_ylabel('E')

    ax7 = fig.add_subplot(gs3[0,1],projection='3d')
    ax7.plot_wireframe(x, y, z, rstride=4, cstride=4, color='k', alpha=0.2)
    ax7.set_xlabel('Z')
    ax7.set_ylabel('N')
    ax7.set_zlabel('E')
    ax7.set_xticklabels('')
    ax7.set_yticklabels('')
    ax7.set_zticklabels('')

    gs3.tight_layout(fig, rect=[0.45, None, None, 0.50])

    top = min(gs1.top, gs2.top)
    bottom = max(gs1.bottom, gs3.bottom)

    gs1.update(top=top, bottom=bottom)
    gs2.update(top=top)
    gs3.update(bottom=bottom)

    plt.figtext(.06,.96,'(a)',fontsize=16)
    plt.figtext(.06,.65,'(b)',fontsize=16)
    plt.figtext(.06,.35,'(c)',fontsize=16)
    plt.figtext(.52,.96,'(d)',fontsize=16)
    plt.figtext(.78,.96,'(e)',fontsize=16)
    plt.figtext(.52,.48,'(f)',fontsize=16)
    plt.figtext(.78,.46,'(g) Ellipsoid',fontsize=16)

    plt.show()


  return rect, plan, az, iang


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
  dic['Ponset'] = int(full_time[ponset]*1./s.dt)
 
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
    plt.savefig('../results/Piton/figures/Examples/hash.png')
    plt.show()

  return dic

# ================================================================

def compare_ponsets(set='test'):
  """
  Compare the Ponsets determined either with the kurtosis gradient, either with the 
  frequency stack of the spectrogram.
  """
  from scipy.io.matlab import mio
  from options import MultiOptions
  opt = MultiOptions()

  if set == 'test':
    datafiles = glob.glob(os.path.join(opt.opdict['datadir'],'TestSet/SigEve_*'))
    datafiles.sort()
    liste = [os.path.basename(datafiles[i]).split('_')[1].split('.mat')[0] for i in range(len(datafiles))]
    liste = map(int,liste) # sort the list of file following the event number
    liste.sort()

    df = pd.read_csv('%s/features/Piton_testset.csv'%opt.opdict['outdir'],index_col=False)
    df = df.reindex(columns=['Dur_freq','Ponset_freq','Dur_grad','Ponset_grad'])
    for ifile,numfile in enumerate(liste):
      file = os.path.join(opt.opdict['datadir'],'TestSet/SigEve_%d.mat'%numfile)
      print ifile,file
      mat = mio.loadmat(file)
      for comp in opt.opdict['channels']:
        ind = (numfile, 'BOR', comp)
        df_one = df.reindex(index=[str(ind)])
        pfr = df_one.Ponset_freq
        pgr = df_one.Ponset_grad
        dfr = df_one.Dur_freq
        dgr = df_one.Dur_grad

        s = SeismicTraces(mat,comp)
        fig = plt.figure(figsize=(9,4))
        fig.set_facecolor('white')
        plt.plot(s.tr,'k')
        plt.plot([pfr,pfr],[np.min(s.tr),np.max(s.tr)],'r',lw=2.,label='freq')
        plt.plot([pgr,pgr],[np.min(s.tr),np.max(s.tr)],'r--',lw=2.,label='grad')
        plt.plot([pfr+dfr*1./s.dt,pfr+dfr*1./s.dt],[np.min(s.tr),np.max(s.tr)],'y',lw=2.)
        plt.plot([pgr+dgr*1./s.dt,pgr+dgr*1./s.dt],[np.min(s.tr),np.max(s.tr)],'y--',lw=2.)
        plt.legend()
        plt.show()

# ================================================================
if __name__ == '__main__':
  read_data_for_features_extraction(set='train',save=False)
  #compare_ponsets()
