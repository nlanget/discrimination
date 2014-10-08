#!/usr/bin/env python
# encoding: utf-8

import pandas as pd
import sys,os,glob
import numpy as np
from obspy.core import read,utcdatetime
import matplotlib.pyplot as plt


def common_events_catalogs():
  dir = '/home/nadege/Desktop/IJEN_catalogues'
  list_cat = ['DB_moi_detect_volc_common_POS.csv','DB_moi_detect_volc_common_POSbis.csv']#,'DB_detect_volc_common_new7.csv']

  list = []
  for cat in list_cat:
    df = pd.read_csv('%s/%s'%(dir,cat))
    list.append(np.array(df[df.columns[0]].values))
    print cat, len(df)

  for i in range(len(list)):
    for j in range(len(list[i])):
      list[i][j] = utcdatetime.UTCDateTime(list[i][j])

  diff = np.array([np.min(np.abs(list[0]-utcdatetime.UTCDateTime(date))) for date in list[1]])
  id = np.where(diff==0.)[0]
  print "\nDates"
  dates =  np.array(list[1])[id]
  print dates

  df = pd.read_csv('%S/DB_moi_detect_volc_common_POS.csv'%dir)
  pos_spec = df[df.columns[2]].values[id]
  print "\nClassification spectre"
  print pos_spec
  pos_wf = df[df.columns[3]].values[id]
  print "\nClassification forme d'onde"
  print pos_wf
  df = pd.read_csv('%s/DB_moi_detect_volc_common_POSbis.csv'%dir)
  pos_bis_indo = df[df.columns[2]].values[id]
  print "\nClassification spectre"
  print pos_bis_indo


  datadir = '/home/nadege/Desktop/NEW_CLASS/Cat_POS/POS'
  for idate,date in enumerate(dates):
    # Affichage, dans l'ordre, de classification spectrale, forme d'onde, indonesienne
    print date, pos_spec[idate],pos_wf[idate],pos_bis_indo[idate]
    st = read('%s/*%d%02d%02d_%02d%02d%02d*'%(datadir,date.year,date.month,date.day,date.hour,date.minute,date.second))
    st.filter('bandpass',freqmin=1,freqmax=10)
    st.plot()


# *******************************************************************************

def plot_catalogs():
  dir = '/home/nadege/Desktop/IJEN_catalogues'
  cat = 'DB_moi_detect_volc_common_POSbis.csv'
  df = pd.read_csv('%s/%s'(dir,cat))
  dates = [utcdatetime.UTCDateTime(date) for date in df[df.columns[0]].values]

  print len(dates), df.shape
  datadir = '/home/nadege/Desktop/NEW_CLASS/Cat_POS_bis/POS'
  for idate,date in enumerate(dates):
    etfile = '%s/*%d%02d%02d_%02d%02d%02d*'%(datadir,date.year,date.month,date.day,date.hour,date.minute,date.second)
    files = glob.glob(etfile)
    files.sort()
    if files:
      print date, df[df.columns[2]].values[idate],df[df.columns[3]].values[idate]
      st = read(etfile)
      #st.plot()
      st.filter('bandpass',freqmin=1,freqmax=10)
      st.plot(color='red')


# *******************************************************************************

def classif_auto():
  datadir = '/media/disk/CLASSIFICATION/ijen'
  catname = '../lib/Ijen/Ijen_reclass_all.csv'
  df = pd.read_csv(catname)
  df = df.dropna(how='any')
  df = df.reindex(index=df[df.Type=='?'].index)
  df.index = range(len(df))
  df_reclass = pd.DataFrame(index=df.Date,columns=['Type','Date_UTC'],dtype=str)
  permut = np.random.permutation(len(df))
  j = 0
  for i in permut:
    date = utcdatetime.UTCDateTime(df.Date_UTC[i])
    files = glob.glob('%s/*%d%02d%02d_%02d%02d%02d*'%(datadir,date.year,date.month,date.day,date.hour,date.minute,date.second))
    files.sort()
    if len(files) > 0:
      j = j+1
      print j,df.Date[i]
      file = files[0]
      st = read(file)
      st.filter('bandpass',freqmin=1,freqmax=10)
      spectrogram(st[0],plot=True)
      cl = str(raw_input("Classe ?\n"))
      if cl == 'v':
        cl = 'VulkanikB'
      elif cl == 't':
        cl = 'Tremor'
      elif cl == 'h':
        cl = 'Hibrid'
      elif cl == ' ':
        cl = '?'
      else:
        cl = 'n' 
      df_reclass.Type[df.Date[i]] = cl
      df_reclass.Date_UTC[df.Date[i]] = date
  print df_reclass
  df_reclass.to_csv('../lib/Ijen/Ijen_reclass_all_indet.csv')


# *******************************************************************************

def common_events_myClass():
  path = '/home/nadege/Desktop/NEW_CLASS'
  list_cat = ['Cat_POS','Cat_POS_bis']

  list = []
  for cat in list_cat:
    df = pd.read_csv(os.path.join(os.path.join(path,cat),'MaClassification.csv'))
    list.append(df.Date.values)
    print cat,len(df)

  for i in range(len(list)):
    for j in range(len(list[i])):
      list[i][j] = utcdatetime.UTCDateTime(list[i][j])

  # Search common events
  dates,idcom = [],[]
  for iev,ev in enumerate(list[0]):
    if ev in list[1]:
      dates.append(ev)
      idcom.append(iev)

  print "\nDates"
  print dates

  df1 = pd.read_csv('%s/%s/MaClassification.csv'%(list_cat[0],path))
  df1.Type[df1.Type=='LP'] = 'Tremor'
  df1.Type[df1.Type=='TektonikJauh'] = 'Tektonik'
  df1.Type[df1.Type=='TektonikLokal'] = 'Tektonik'
  df1 = df1.reindex(index=idcom)
  df1.index = range(len(df1))
  pos = df1.Type.values
  print "\nClassification POS", len(pos)
  print pos

  df2 = pd.read_csv('%s/%s/MaClassification.csv'%(list_cat[1],path))
  id2 = [i for i,date in enumerate(df2.Date.values) if utcdatetime.UTCDateTime(date) in dates]
  df2.Type[df2.Type=='LP'] = 'Tremor'
  df2.Type[df2.Type=='TektonikJauh'] = 'Tektonik'
  df2.Type[df2.Type=='TektonikLokal'] = 'Tektonik'
  df2 = df2.reindex(index=id2)
  df2.index = range(len(df2))
  pos_bis = df2.Type.values
  print "\nClassification POS_bis", len(pos_bis)
  print pos_bis

  idcom2 = []
  for i in range(len(idcom)):
    if df1.Type.values[i] == df2.Type.values[i]:
        if df1.Type.values[i] != '?':
          idcom2.append(i)

  df = df1.reindex(index=idcom2)
  df['Date_UTC'] = df.Date
  for id,d in enumerate(df.Date_UTC.values):
    d = utcdatetime.UTCDateTime(d)
    df['Date'].values[id] = '%d%02d%02d%02d%02d%02d'%(d.year,d.month,d.day,d.hour,d.minute,d.second)
  df.to_csv('test_extraction.csv')

  datadir = '/home/nadege/Desktop/NEW_CLASS/Cat_POS/POS'
  plot = False
  dic = {}
  for idate,date in enumerate(dates):
    print date, pos[idate],pos_bis[idate]
    if plot:
      st = read('%s/*%d%02d%02d_%02d%02d%02d*'%(datadir,date.year,date.month,date.day,date.hour,date.minute,date.second))
      st.filter('bandpass',freqmin=1,freqmax=10)
      st.plot()


# *******************************************************************************

def spectral_content(filename,datadir,list_dates=None):
  df = pd.read_csv(filename)
  df.index = df.Date 

  if not list_dates:
    list_dates = df.index
  for date in list_dates:
    file = glob.glob('%s/*Z*%s_%s*'%(datadir,str(date)[:8],str(date)[8:]))
    if len(file) > 0:
      st = read(file[0])
      st.filter('bandpass',freqmin=1,freqmax=50)
      tr = st[0][1500:4000]
      spectrogram(tr,plot=True) 


def spectrogram(full_trace,param=[100.,.9,8],plot=False):
  """
  Adapted from obspy.imaging.spectrogram.spectrogram.py
  param[0] = influence taille de la fenetre glissante (plus la fenetre est petite, plus NFFT diminue)
  param[1] = % de recouvrement entre fenetres glissantes
  param[2] = inluence le nb de points en frequence du spectrogramme (plus c'est grand, plus le nb de pts augmente)
  """
  from matplotlib import mlab
  trace = full_trace[1500:4000]
  Fs = 1./.01
  wlen = Fs / param[0]
  data = trace - np.mean(trace)
  npts = len(data)
  nfft = int(nearest_pow2(wlen*Fs))
  if nfft > npts:
    nfft = int(nearest_pow2(npts/4.))
  nlap = int(nfft*param[1])
  end = npts/Fs

  mult = int(nearest_pow2(param[2]))
  mult = mult*nfft

  # Normalize data
  data = data/np.max(np.abs(data))
  """
  NFFT = nb of points used for computing the FFT in each window.
  Fs = sampling frequency.
  noverlap = nb of points of overlap between windows.
  pad_to = nb of points to which the data segment is padded.
  """
  specgram, f, time = mlab.specgram(data,Fs=Fs,NFFT=nfft,pad_to=mult,noverlap=nlap)
  specgram = np.flipud(specgram)

  f = f[:-1]
  #lim = int(smallest_pow2(len(time)))
  #time = time[:lim]
  specgram = specgram[:-1,:]#lim]

  TF,freqs = spectrum(trace)

  if plot:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    G = gridspec.GridSpec(2,2)
    halfbin_time = (time[1] - time[0]) / 2.0
    halfbin_freq = (f[1] - f[0]) / 2.0
    #extent = (time[0] - halfbin_time, time[-1] + halfbin_time, f[0] - halfbin_freq, f[-1] + halfbin_freq)
    extent = (time[0] - halfbin_time, time[-1] + halfbin_time, f[0] - halfbin_freq, f[52])
    fig = plt.figure(figsize=(10,4))
    fig.set_facecolor('white')
    ax1 = fig.add_subplot(G[0,0])
    ax1.plot(full_trace,'k')
    ax2 = fig.add_subplot(G[1,0])
    #ax2.imshow(np.log(specgram[52:,:]),interpolation="nearest",cmap=plt.cm.jet,extent=extent)
    ax2.imshow(specgram[52:,:],interpolation="nearest",cmap=plt.cm.jet,extent=extent)
    ax2.set_xlim([0,end])
    ax2.set_ylim([0,20])
    ax2.axis('tight')
    ax3 = fig.add_subplot(G[:,1])
    ax3.plot(freqs[:len(freqs)/2],np.abs(TF[:len(TF)/2]),'k')
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_xlim([0,20])
    plt.show()


def spectrum(trace):
    """
    Computes the Fourier transform of the signal
    """
    data = trace
    TF = np.fft.fft(data)
    freqs = np.fft.fftfreq(len(TF),d=.01)
    return TF,freqs


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

# *******************************************************************************

def plotSampleWF():
  """
  Tire au hasard 10 événements de chaque classe manuelle et affiche 
  les formes d'ondes.
  1 figure par classe.
  """
  datadir = '../data/Ijen/ID/IJEN/EHZ.D'
  catname = '../lib/Ijen/Ijen_reclass_all.csv'
  df = pd.read_csv(catname)
  df = df.dropna(how='any')
  df = df.reindex(index=df[df.Type!='?'].index)
  df = df.reindex(index=df[df.Type!='n'].index)

  tuniq = np.unique(df.Type.values)
  print tuniq
  for i in range(len(tuniq)):
    df_type = df[df.Type==tuniq[i]]
    permut = np.random.permutation(df_type.index)
    fig = plt.figure(figsize=(10,12))
    fig.set_facecolor('white')
    num = 0
    for ij,j in enumerate(permut):
      if num >= 10:
        break
      date = utcdatetime.UTCDateTime(str(df.Date[j]))
      files = glob.glob('%s/*%d%02d%02d_%02d%02d%02d*'%(datadir,date.year,date.month,date.day,date.hour,date.minute,date.second))
      files.sort()
      if len(files) > 0:
        print date
        file = files[0]
        st = read(file)
        st.filter('bandpass',freqmin=1,freqmax=10)

        ax = fig.add_subplot(10,1,num+1)
        ax.plot(st[0],'k')
        ax.set_axis_off()
        num = num + 1
    fig.suptitle(tuniq[i])
    #plt.savefig('/home/nadege/Desktop/reclass_%s.png'%tuniq[i])
  plt.show()

# *******************************************************************************

def plotOneSampleWF():
  """
  Tire au hasard un événement de chaque classe et affiche les formes d'ondes.
  1 figure avec toutes les formes d'ondes.
  """
  datadir = '../data/Ijen/ID/IJEN/EHZ.D'
  catname = '../lib/Ijen/Ijen_class_all.csv'
  df = pd.read_csv(catname)
  df = df.dropna(how='any')
  df = df.reindex(index=df[df.Type!='?'].index)
  df = df.reindex(index=df[df.Type!='n'].index)

  tuniq = np.unique(df.Type.values)
  print tuniq
  fig = plt.figure(figsize=(12,12))
  fig.set_facecolor('white')
  for i in range(len(tuniq)):
    df_type = df[df.Type==tuniq[i]]
    permut = np.random.permutation(df_type.index)
    marker = 1
    j = 0
    while marker:
      p = permut[j]
      j = j+1
      date = utcdatetime.UTCDateTime(str(df.Date[p]))
      files = glob.glob('%s/*%d%02d%02d_%02d%02d%02d*'%(datadir,date.year,date.month,date.day,date.hour,date.minute,date.second))
      files.sort()
      if len(files) > 0:
        marker = 0
        file = files[0]
        st = read(file)
        st.filter('bandpass',freqmin=1,freqmax=10)

        ax = fig.add_subplot(len(tuniq),1,i+1)
        ax.plot(st[0],'k')
        ax.text(.05,.9,tuniq[i],transform=ax.transAxes)
        ax.set_axis_off()
  plt.savefig('/home/nadege/Desktop/sample_waveforms.png')
  plt.show()


# *******************************************************************************

def plotOneSampleWF_redac():
  """
  Tire au hasard un événement de chaque classe et affiche les formes d'ondes.
  1 figure avec toutes les formes d'ondes.
  """
  from matplotlib.gridspec import GridSpec
  datadir = '../data/Ijen/ID/IJEN/EHZ.D'
  catname = '../lib/Ijen/Ijen_reclass_all.csv'
  df = pd.read_csv(catname)
  df = df.dropna(how='any')
  #df = df.reindex(index=df[df.Type!='?'].index)
  df = df.reindex(index=df[df.Type!='n'].index)

  tuniq = np.unique(df.Type.values)
  print tuniq
  fig = plt.figure(figsize=(12,12))
  fig.set_facecolor('white')
  grid = GridSpec(16,3)
  for i in range(len(tuniq)):
    df_type = df[df.Type==tuniq[i]]
    permut = np.random.permutation(df_type.index)
    marker = 1
    j = 0
    compteur = 0
    while marker:
      p = permut[j]
      j = j+1
      date = utcdatetime.UTCDateTime(str(df.Date[p]))
      files = glob.glob('%s/*%d%02d%02d_%02d%02d%02d*'%(datadir,date.year,date.month,date.day,date.hour,date.minute,date.second))
      files.sort()
      if len(files) > 0:
        compteur = compteur+1
        if compteur == 3:
          marker = 0
        file = files[0]
        st = read(file)
        st.filter('bandpass',freqmin=1,freqmax=10)

        if compteur == 1:
          ax = fig.add_subplot(grid[2*i+1,:])
          ax.plot(st[0],'k')
        else:
          ax = fig.add_subplot(grid[2*i,compteur-1])
          ax.plot(st[0],'k')
        ax.set_axis_off()
    ax = fig.add_subplot(grid[2*i,0])
    ax.text(.2,.5,tuniq[i],transform=ax.transAxes)
    ax.set_axis_off()
  plt.savefig('/home/nadege/Desktop/sample_waveforms.png')
  plt.show()


if __name__ == '__main__':
  #common_events_catalogs()
  #plot_catalogs()
  #common_events_myClass()
  #classif_auto()
  #spectral_content('/home/nadege/Desktop/IJEN_catalogues/test_extraction.csv','/home/nadege/Desktop/NEW_CLASS/Cat_POS/POS')
  #plotSampleWF()
  #plotOneSampleWF()
  plotOneSampleWF_redac()

