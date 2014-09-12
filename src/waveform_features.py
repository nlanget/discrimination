import numpy as np
from obspy.signal import filter
from obspy.core import read
from scipy.stats import kurtosis
from scipy.integrate import trapz


# ***********************************************************


def max_over_mean(trace):

  """
  Returns the maximum of the envelope of a trace over its mean.
  from Hibert 2012
  """

  e = filter.envelope(trace)
  emax = np.max(e)
  emean = np.mean(e)

  return emax/emean


# ***********************************************************


def kurtosis_envelope(trace):

  """
  Returns the kurtosis envelope of a trace
  from Hibert 2012
  """

  k = kurtosis(trace, axis=0, fisher=False, bias=True) # Pearson's definition: k(fisher)=k(pearson)-3

  return k


# ***********************************************************


def skewness(trace):
  """
  Returns the skewness of the envelope of a trace
  """
  from scipy.stats import skew

  sk = skew(trace)

  return sk


# ***********************************************************


def autocorrelation(trace,plot=False):
  """
  Computes the autocorrelation function
  from Langer et al. 2006
  """
  a = np.correlate(trace.tr,trace.tr,'full')
  a = a[len(a)/2:]
  a = a/np.max(a)

  t = np.arange(len(trace.tr))*trace.dt
  val, tval = window_p(np.abs(a),t,0,opt='mean')

  if plot:
    import matplotlib.pyplot as plt
    fig = plt.figure()
    fig.set_facecolor('white')
    ax1 = fig.add_subplot(211)
    ax1.plot(t,trace.tr,'k')
    ax1.set_xticklabels('')
    ax2 = fig.add_subplot(212)
    ax2.plot(t,a,'k')
    ax2.plot(tval,val,'r')
    ax2.set_xlabel('Time (s)')
    plt.show()

  return val

# ***********************************************************


def filt_ratio(trace,plot=False):
  """
  Compute the amplitude ratio between filtered and unfiltered traces
  from Langer et al. 2006
  """
  from obspy.signal import filter
  x = trace.tr
  x_filt = filter.bandpass(trace.tr,0.7,1.5,1./trace.dt)
  x_filt = x_filt
  r = x_filt/x
  r = r/np.max(r)

  t = np.arange(len(trace.tr))*trace.dt
  val, tval = window_p(r,t,0,opt='max')

  if plot:
    import matplotlib.pyplot as plt
    fig = plt.figure()
    fig.set_facecolor('white')
    ax1 = fig.add_subplot(311)
    ax1.plot(t,x,'k')
    ax1.set_xticklabels('')
    ax2 = fig.add_subplot(312)
    ax2.plot(t,x_filt,'k')
    ax2.set_xticklabels('')
    ax3 = fig.add_subplot(313)
    ax3.plot(t,r,'k')
    ax3.plot(tval,val,'r')
    ax3.set_xlabel('Time (s)')
    plt.show()

  return val

# ***********************************************************


def signal_duration(trace,it0=None,plot=False):
  """
  Returns the signal duration of a trace.
  The P-onset is taken as the maximum of the kurtosis gradient.
  from Hibert 2012
  """

  n_env = len(trace.tr_env)
  n_grad = len(trace.tr_grad)
  if it0:
    # Search for the maximum of kurtosis gradient in a small window around the origin time
    lim1, lim2 = it0 - 1*1./trace.dt - (n_env-n_grad), it0 + 1.5*1./trace.dt-(n_env-n_grad)
    if lim1 < 0:
      lim1,lim2 = 0, (1+1.5) * 1./trace.dt
  else:
    lim1, lim2 = 50, len(trace.tr)/2-1
  lim3 = lim1 + (n_env-n_grad)
  tstart = np.argmax(trace.tr_grad[lim1:lim2]) + (n_env - n_grad) + lim1
  mean_level = np.mean(trace.tr_env[lim3:lim3+50])

  stop,maxa = trigger(trace.tr_env,tstart,plot=False)

#  max_env=np.argmax(trace.tr_env)
#  popt,vec = exp_fit(trace.tr_env,max_env,stop,mean_level)
#  if popt != None:
#    fit_coda = func(vec,popt[0]+popt[2],popt[1],0)
#    b_coda = np.where(fit_coda < mean_level)[0]
#  else:
#    b_coda = np.array([])
#
#  b_env = np.where(trace.tr_env[tstart:] < mean_level)[0]
#
#  if b_coda.all() and b_coda.any():
#    tend_coda = b_coda[0]+ max_env
#    dur_coda = (tend_coda - tstart) * trace.dt
#  else:
#    tend_coda = 0
#    dur_coda = 0
#
#  if b_env.all() and b_env.any():
#    tend_env = b_env[0] + tstart
#    dur_env = (tend_env - tstart) * trace.dt
#  else:
#    tend_env = 0
#    dur_env = 0
#
#  # Choice of the best end
#  if tend_env and tend_coda:
#    tend = np.min([tend_env,tend_coda])
#    dur = np.min([dur_env,dur_coda])
#  elif tend_env == 0:
#    tend = tend_coda
#    dur = dur_coda
#  elif tend_coda == 0:
#    tend = tend_env
#    dur = dur_env

  # Choice of the best P-onset
  ponset = tstart # default
  if np.abs(tstart-maxa) < 200:
    aa,bb = np.mean(np.abs(trace.tr[tstart-10:tstart])), np.mean(np.abs(trace.tr[tstart:tstart+10]))
    cc,dd = np.mean(np.abs(trace.tr[maxa-10:maxa])), np.mean(np.abs(trace.tr[maxa:maxa+10]))
    if np.abs(aa/bb) > np.abs(cc/dd):
      ponset = maxa
    #print aa, bb, np.abs(aa/bb)
    #print cc, dd, np.abs(cc/dd)


  if plot:

    import matplotlib.pyplot as plt

    if 'station' in trace.__dict__.keys():
      print trace.station, ponset, tend, dur
    else:
      print ponset, tend, dur

    fig = plt.figure(figsize=(9,5))
    fig.set_facecolor('white')
    plt.plot(trace.tr,'k')
    plt.plot(trace.tr_env)
    plt.plot([tstart,tstart],[np.min(trace.tr),np.max(trace.tr)],'g',lw=2)
    plt.plot([maxa,maxa],[np.min(trace.tr),np.max(trace.tr)],color=(0.5,1,0.5),ls=':',lw=2)
    plt.plot(max_env+np.arange(len(fit_coda)),fit_coda,'c')
    plt.plot([ponset,ponset],[np.min(trace.tr),np.max(trace.tr)],'g',lw=4)
    #plt.plot(max_env+np.arange(len(fit_coda)),func(vec,*popt),'m')
    plt.plot([tend_env,tend_env],[np.min(trace.tr),np.max(trace.tr)],'r',label='env')
    plt.plot([tend_coda,tend_coda],[np.min(trace.tr),np.max(trace.tr)],'r:',label='coda')
    plt.plot([stop,stop],[np.min(trace.tr),np.max(trace.tr)],'b--',label='stop')
    plt.plot(mean_level*np.ones(len(trace.tr_env)),'y--')
    #plt.plot(1/np.exp(1)*np.ones(len(env.data)),c=(1,0.5,0),ls='--')
    plt.legend()
    plt.figtext(.70,.20,"tend coda = %d"%tend_coda)
    plt.figtext(.70,.15,"tend env = %d"%tend_env)
    if 'station' in trace.__dict__.keys():
      plt.title(trace.station)
    plt.show()

  return ponset#,tend,dur



def exp_fit(tr_env, tmax, stop, noise):
  """
  Exponential fit of the coda envelope
  """
  coda = tr_env[tmax:]
  N = len(coda)
  if stop !=0 and stop-tmax > 200:
    coda = tr_env[tmax:stop]
  from scipy.optimize import curve_fit
  vec = np.linspace(0,1,(len(coda)))
  try:
    popt, pcov = curve_fit(func,vec,coda,maxfev=5000)
    return popt, vec 

  except RuntimeError:
    print "Number of calls exceeded (station %s). Continue anyway..."%tr_env.stats.station
    return None, None



def func(x, a, b, c):
  """
  Exponential function
  """
  return a * np.exp(-b * x) + c



def trigger(env,ponset,plot=False):
  """
  Determines the end of the window for coda fitting
  """
  import obspy.signal.trigger
  a = obspy.signal.trigger.classicSTALTA(env,10,100)
  sup = np.where(a>=np.mean(a)*3)[0]
  list_i=[]
  for i in range(len(sup)):
    if sup[i] >= 1400 and sup[i] <= 2000:
      list_i.append(i)
  sup = np.delete(sup,list_i)
  if sup.all() and sup.any():
    stop = sup[0]
    if stop < ponset:
      c = np.where(sup > ponset)[0]
      if c.any() and c.all():
        stop = sup[c[0]]
      else:
        stop = 0
  else:
    stop = 0

  if plot:
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(9,5))
    fig.set_facecolor('white')
    plt.plot(a,'r')
    plt.plot(int(np.max(a))*2/5*np.ones(len(a)),'y--')
    plt.plot(np.mean(a)*3*np.ones(len(a)),color=(1,.5,0),ls='--')
    plt.plot([ponset,ponset],[np.min(a),np.max(a)],'b')

  maxa = np.argmax(a)
  if maxa < 1400 or maxa > 2000:
    maxa = np.argmax(a[1400:2000])+1400

  return stop, maxa


# ***********************************************************


def growth_over_decay(trace):
  """
  Returns the duration of growth over the duration of decay of the envelope trace
  from Hibert 2012
  """

  time = np.arange(len(trace.tr))*trace.dt 
  ti = time[trace.ponset]
  tf = time[trace.tend]
  emax = np.argmax(trace.tr_env[trace.ponset:trace.tend])+trace.ponset
  tmax = time[emax]
  p = (tmax - ti)/(tf - tmax)
  print ti, tmax, tf, p
  return p


# ***********************************************************


def desc_and_asc(trace):
  """
  Returns the duration of decay of the envelope trace over the total duration
  and the duration of growth of the envelope trace over the total duration
  """
 
  time = np.arange(len(trace.tr))*trace.dt 
  ti = time[trace.ponset]
  tf = time[trace.tend]
  emax = np.argmax(trace.tr_env[trace.ponset:trace.tend])+trace.ponset
  tmax = time[emax]
  #p1 = (tf - tmax)/trace.dur  # decay
  p2 = (tmax - ti)/trace.dur  # growth
  return p2


# ***********************************************************


def energy_between_10Hz_and_30Hz(trace,dt,wd=10,wf=30,ponset=0,tend=0):
  """
  Returns the energy between 10 and 30 Hz of a trace
  from Hibert 2012
  wd = lowest frequency
  wf = highest frequency 
  """

  befp = 10
  if ponset-befp < 0:
    p1 = 0
  else:
    p1 = ponset-befp
  if tend+befp > len(trace) or tend == 0:
    trace = trace[p1:]
  else:
    trace = trace[p1:tend+befp]

  TF = np.fft.fft(trace)
  TF = np.absolute(TF)

  freqs = np.fft.fftfreq(len(TF),d=dt)

  pas_en_freq = 1.0/dt
  n = len(TF)
  df = pas_en_freq/n
  wi = 0

  xi = (wd - wi)/df
  xf = (wf - wi)/df

  energy = aire(TF[xi:xf+1])*df

  return energy


# ***********************************************************


def kurto_bandpass(trace,plot=False):
  """
  Returns the frequency bandpass which maximizes the kurtosis
  from Antoni 2007
  """

  from waveloc.kurtogram import Fast_Kurtogram
  import matplotlib.pyplot as plt

  data = trace.tr
  N = len(data)
  N2 = np.log2(N)-7
  nlevel = int(np.fix(N2))

  dt = trace.dt

  fig = plt.figure()
  fig.set_facecolor('white')
  Kwav, Level_w, freq_w, c, f_lower, f_upper = Fast_Kurtogram(np.array(data,dtype=float), nlevel,verbose=plot,Fs=1./dt,opt2=1)
  plt.savefig('../results/Piton/figures/Examples/kurto.png')

  return f_lower, f_upper


# ***********************************************************


def spectrogram(trace,plot=False):
  """
  Adapted from obspy.imaging.spectrogram.spectrogram.py
  """
  from matplotlib import mlab
  Fs = 1./trace.dt
  wlen = Fs / 100.
  if ('i1' or 'i2') in trace.__dict__.keys():
    data = trace.tr[trace.i1:trace.i2] - np.mean(trace.tr[trace.i1:trace.i2])
    det_ponset = False
  else:
    data = trace.tr - np.mean(trace.tr)
    det_ponset = True
  npts = len(data)
  nfft = int(nearest_pow2(wlen*Fs))
  if nfft > npts:
    nfft = int(nearest_pow2(npts/8.))
  nlap = int(nfft*.9)
  end = npts/Fs

  mult = int(nearest_pow2(8))
  mult = mult*nfft

  # Normalize data
  data = data/np.max(np.abs(data))
  specgram, f, time = mlab.specgram(data,Fs=Fs,NFFT=nfft,pad_to=mult,noverlap=nlap)
  specgram = np.flipud(specgram)

  # Stack in frequency
  b = np.sum(specgram,axis=0)
  l = cum(b,plot=False)
  nimp = np.argmin(l[:len(l)/2])
  mins = find_local_min(l)*npts/len(time)
  tt = np.arange(len(trace.tr))*trace.dt
  if det_ponset:
    l = cum(b,plot=False)
    ponset = np.argmin(l[:len(l)/2])
    trace.ponset = time[ponset]*1./trace.dt
    if trace.ponset-500 > 0:
      trace.i1 = int(trace.ponset-500)
    else:
      trace.i1 = 0
  else:
    ponset = np.argmin(np.abs(time-tt[trace.ponset]))

  level = np.mean(b[0:10]) # level in the very beginning of the signal
  level_befp = np.mean(b[ponset-20:ponset]) # level just before the P-onset
  imax = np.argmax(b[ponset:ponset+500])+ponset
  idurfs = np.where(b[imax:]<level)[0]+imax
  if idurfs.any() and idurfs.all():
    idurf = idurfs[0]
  else:
    idurf = len(b)-1
  durf = time[idurf]
  trace.dur = durf-time[ponset]
  if trace.dur < 0.5:
    idurf = len(b)-1
    durf = time[idurf]
    trace.dur = durf-time[ponset]
  trace.tend = trace.ponset + trace.dur * 1./trace.dt
  if det_ponset:
    if trace.tend+500 < len(trace.tr):
      trace.i2 = int(trace.tend+500)
    else:
      trace.i2 = int(len(trace.tr)-1)

  # Keeps the predominant frequency at each time sample
  evolf = f[::-1][np.argmax(specgram,axis=0)]
  tend = np.argmin(np.abs(time-tt[trace.tend]))
  val, tval = window_p(evolf,time,ponset,tend,befp=10,opt='mean')

  # Stack in time
  a = np.sum(specgram,axis=1)
  a = a[::-1]
  maxa = find_local_max(a)
  maxb = find_local_max(b)
  # Maximum frequency of the smoothed spectrum
  abs_max = f[np.argmax(a)]
  #print "Dominant frequency = %.1f Hz"%abs_max

  # Half-octave bands
  klow = [0.47,0.70,1.05,1.58,2.37,3.56,5.33,8.00,12.00,18.00]
  khigh = [0.78,1.17,1.76,2.63,3.95,5.93,8.89,13.33,20,30]
  N = len(klow)
  denom = np.sum(specgram[:,ponset-10:],axis=0)
  hob = np.empty(shape=(N,denom.shape[0]))
  for i in range(N):
    i1 = np.where(f >= klow[i])[0][0]
    i2 = np.where(f <= khigh[i])[0][-1]
    num = np.sum(specgram[i1:i2,ponset-10:],axis=0)
    hob[i,:] = np.log(num/denom)
  # Fit
  absc = np.linspace(0,1,hob.shape[1])
  p = np.polyfit(absc,hob.T,5)
  ts = np.linspace(0,1,150)
  s_hob = np.empty(shape=(hob.shape[0],len(ts)))
  for i in range(p.shape[1]):
    s_hob[i] = np.polyval(p.T[i],ts)

  # Looks for the maximum of the spectrogram
  (lmax,cmax) = np.unravel_index(np.argmax(specgram[:,ponset:tend]),specgram[:,ponset:tend].shape)
  cmax = cmax+ponset

  # Looks for the maximum of the spectrogram (relative time)
  tmaxi = (time[np.unravel_index(np.argmax(specgram[:,ponset:tend]),specgram[:,ponset:tend].shape)[1]+ponset]-time[ponset])*1./trace.dur
  if tmaxi < 0:
    tmaxi = 0
  elif tmaxi > 1:
    tmaxi = 1

  # Looks for pure harmonics
  #print f[maxa]
  #for i in range(len(maxa)):
    #for j in range(i+1,len(maxa)):
      #print i, j, f[maxa][j]/f[maxa][i]
  # Keep the interesting part of the spectrogram only
  lim = 150
  f_complete = f
  a_complete = a
  f = f[:lim]
  a = a[:lim]
  specgram = specgram[specgram.shape[0]-lim:,:]
  maxa = maxa[maxa<lim]

  # Width of the spectrum
  w = len(np.where(a > 0.1*np.max(a))[0])*(f[1]-f[0])

  # Peak factor
  #rms = np.sqrt(1./len(b)*aire(b**2))
  #peak_factor = np.abs(b)/rms

  # Plot
  if plot:

    halfbin_time = (time[1] - time[0]) / 2.0
    halfbin_freq = (f[1] - f[0]) / 2.0
    extent = (time[0] - halfbin_time, time[-1] + halfbin_time, f[0] - halfbin_freq, f[-1] + halfbin_freq)

    print "Duration :",trace.dur,'s'
    print "Ponset :", trace.ponset 

    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from mpl_toolkits.mplot3d import Axes3D
    G = gridspec.GridSpec(3,2)

    fig = plt.figure(figsize=(12,7))
    fig.set_facecolor('white')

    gs1 = gridspec.GridSpec(3,1)
    ax1 = fig.add_subplot(gs1[0])
    ax1.plot(data,'k')
    ax1.plot([trace.ponset,trace.ponset],[np.min(data),np.max(data)],'r',lw=2.)
    ax1.plot([cmax*npts/len(time),cmax*npts/len(time)],[np.min(data),np.max(data)],'y',lw=2.)
    ax1.plot([durf*1./trace.dt,durf*1./trace.dt],[np.min(data),np.max(data)],'b',lw=2.)
    ax1.set_xlim([0,len(data)])
    ax1.set_title('Signal')
    ax1.set_xticklabels('')

    ax2 = fig.add_subplot(gs1[1])
    ax2.specgram(data,Fs=Fs,NFFT=nfft,pad_to=mult,noverlap=nlap)
    ax2.set_xlim([0,end])
    ax2.axis('tight')
    ax2.set_xticklabels('')
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_title('Spectrogram')

    ax3 = fig.add_subplot(gs1[2])
    ax3.plot(time,evolf,'k')
    if val:
      ax3.plot(tval,val,'r')
    ax3.set_xlim([0,time[-1]])
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Frequency (Hz)')
    ax3.set_title('Predominant frequency')

    gs1.tight_layout(fig, rect=[None, None, 0.45, None])

    gs2 = gridspec.GridSpec(1,2)
    ax4 = fig.add_subplot(gs2[0,0])
    ax4.specgram(data,Fs=Fs,NFFT=nfft,pad_to=mult,noverlap=nlap)
    ax4.set_xlim([0,end])
    ax4.axis('tight')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Frequency (Hz)')
    ax4.set_title('Spectrogram')
    
    ax5 = fig.add_subplot(gs2[0,1])
    ax5.plot(a_complete,f_complete,'k')
    ax5.set_yticklabels('')
    ax5.set_title('Complete time stack')
    gs2.tight_layout(fig, rect=[0.45, 0.50, None, None])

    gs3 = gridspec.GridSpec(1,2)
    ax6 = fig.add_subplot(gs3[0,0])
    ax6.plot(time,b,'k')
    ax6.plot([time[0],time[-1]],[level,level],'r')
    ax6.plot([time[idurf],time[idurf]],[np.min(b),np.max(b)],'b')
    ax6.plot([time[ponset],time[ponset]],[np.min(b),np.max(b)],'b')
    ax6.set_xlim([0,time[len(b)-1]])
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel(' ')
    ax6.set_title('Frequency stack')

    ax7 = fig.add_subplot(gs3[0,1])
    ax7.plot(f,a,'k')
    if f[np.argmax(a)]-w/2 > 0:
      ax7.plot([f[np.argmax(a)]-w/2.,f[np.argmax(a)]+w/2.],[0.1*np.max(a),0.1*np.max(a)],'y')
    else:
      ax7.plot([0,w],[0.1*np.max(a),0.1*np.max(a)],'y')
    ax7.plot(f[maxa],a[maxa],'ro')
    ax7.set_title('Zoomed-in time stack')
    ax7.set_xlabel('Frequency (Hz)')

    gs3.tight_layout(fig, rect=[0.465, None, None, 0.50])

    top = min(gs1.top, gs2.top)
    bottom = max(gs1.bottom, gs3.bottom)

    gs1.update(top=top, bottom=bottom)
    gs2.update(top=top)
    gs3.update(bottom=bottom)

    plt.figtext(.06,.96,'(a)',fontsize=16)
    plt.figtext(.06,.65,'(b)',fontsize=16)
    plt.figtext(.06,.35,'(c)',fontsize=16)
    plt.figtext(.49,.96,'(d)',fontsize=16)
    plt.figtext(.75,.96,'(e)',fontsize=16)
    plt.figtext(.49,.46,'(f)',fontsize=16)
    plt.figtext(.75,.46,'(g)',fontsize=16)

    plt.show()


  return trace, np.mean(evolf[ponset:]), tmaxi, len(maxa), w, s_hob, val, abs_max



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



def find_local_max(a):

  gradients=np.diff(a)

  maxs=np.array([],dtype=int)
  count=0
  for i in gradients[:-1]:
    count+=1
    if ((cmp(i,0)>0) & (cmp(gradients[count],0)<0) & (i != gradients[count])):
      maxs=np.append(maxs,count)

  return maxs


def find_local_min(a):

  gradients=np.diff(a)

  mins=np.array([],dtype=int)
  count=0
  for i in gradients[:-1]:
    count+=1
    if ((cmp(i,0)<0) & (cmp(gradients[count],0)>0) & (i != gradients[count])):
      mins=np.append(mins,count)

  return mins


# ***********************************************************


def around_freq(TF,freqs,plot=False):
  """
  Returns:
  - the predominant frequency, i.e the frequency of maximum value of Fourier amplitude spectra;
  - the bandwidth bwid, i.e the width of range of frequencies;
  - the central frequency fmean, i.e. the frequency of power concentration
  from Hammer et al. 2012
  """
  TF = TF[:len(TF)/2]
  freqs = freqs[:len(freqs)/2]

  predf = freqs[np.argmax(np.abs(TF))]

  df = freqs[1]-freqs[0]
  #int = np.sum(np.abs(TF))
  int = aire(np.abs(TF))
  fmean = aire(freqs*np.abs(TF))/int
  fc2 = aire(freqs**2*np.abs(TF))/int

  bwid = np.sqrt(aire((freqs-fmean)**2*np.abs(TF))/int)

  if plot:
    import matplotlib.pyplot as plt 
    fig = plt.figure()
    fig.set_facecolor('white')
    plt.plot(freqs,np.abs(TF),'k')
    plt.plot([fmean,fmean],[0,np.max(np.abs(TF))],'r',lw=2,label='central freq')
    plt.plot([predf,predf],[0,np.max(np.abs(TF))],'g',lw=2,label='predominant freq')
    plt.plot([fmean-bwid,fmean+bwid],[0.1*np.max(np.abs(TF)),0.1*np.max(np.abs(TF))],'y',lw=2,label='bandwidth')
    plt.figtext(.6,.65,"Central f = %.2f Hz"%fmean)
    plt.figtext(.6,.6,"Predominant f = %.2f Hz"%predf)
    plt.figtext(.6,.55,"Bandwidth = %.2f Hz"%bwid)
    plt.xlim([0,16])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("abs(TF)")
    plt.title("Amplitude spectrum")
    plt.legend()
    plt.show()
  return predf,bwid,fmean


# ***********************************************************


def cepstrum(TF,freq,plot=False):

  """
  Returns the cepstral coefficients of the signal; gives the spectral energy distribution
  From Hammer et al. 2012
  """

  lnTF = np.log(TF)
  cepstr = np.fft.ifft(lnTF)


  t = np.fft.fftshift(np.fft.fftfreq(len(cepstr),d=freq[1]-freq[0]))
  t = t + np.abs(np.min(t))

  if plot:
    import matplotlib.pyplot as plt
    fig = plt.figure()
    fig.set_facecolor('white')
    ax1 = fig.add_subplot(211)
    ax1.plot(freq[:len(freq)/2],np.abs(TF[:len(TF)/2]),'k')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Amplitude spectrum')
    ax2 = fig.add_subplot(212)
    ax2.plot(t,cepstr,'k')
    ax2.set_xlabel('Quefrency (s)')
    ax2.set_ylabel('Cepstrum')
    ax2.axis('tight')
    plt.show()

  return cepstr


# ***********************************************************


def instant_freq(trace, dt, TF, ponset, tend, plot=False):

  """
  Returns the instantaneous frequency of a signal
  from Hammer et al. 2012
  """

  from math import pi

  freqs = np.fft.fftfreq(len(TF),d=dt)
  hilb = np.fft.ifft(-1j * np.sign(freqs) * TF)
  
  analytic = trace + 1j* hilb

  time = np.arange(len(trace))*dt

  phase = np.angle(analytic,deg=False)
  ifreq = 1./(2*pi) * np.gradient(np.unwrap(phase)) * 1./dt

  p = np.polyfit(time,np.unwrap(phase),1)

  val, tval = window_p(ifreq,time,ponset,tend,befp=50,opt='mean')

  if plot:
    import matplotlib.pyplot as plt
    fig = plt.figure()
    fig.set_facecolor('white')
    ax1 = fig.add_subplot(221,title='Signal')
    ax1.plot(time,trace,'k')
    ax2 = fig.add_subplot(222,title='Analytic signal')
    ax2.plot(time,np.real(analytic),'k',label='real part')
    ax2.plot(time,np.imag(analytic),'r',label='imag part')
    #ax2.legend()
    ax3 = fig.add_subplot(223,title='Instantaneous phase (unwrapped)')
    ax3.plot(time,np.unwrap(phase),'k')
    ax3.plot(time,np.polyval(p,time),'r')
    ax3.set_xlabel('Time (s)')
    ax4 = fig.add_subplot(224,title='Instantaneous frequency')
    ax4.plot(time[10:],ifreq[10:],'k')
    ax4.plot(tval,val,'r')
    ax4.set_xlabel('Time (s)')
    plt.show()

  return val, p[0]


# ***********************************************************


def instant_bw(trace, env, dt, TF, ponset=0, tend=0, plot=False):
  """
  Returns:
  - the instantaneous bandwidth which gives relative change of envelope.
  - the normalized envelope, i.e. the smoothed instantaneous bandwidth normalized by Nyquist frequency (1/(2T)). To enlarge variance, constant is multiplied and exponential is taken.
  from Hammer et al. 2012
  """

  from math import pi

  freqs = np.fft.fftfreq(len(TF),d=dt)
  df = freqs[1]-freqs[0]

  hilb = np.fft.ifft(-1j * np.sign(freqs) * TF)
  analytic = trace + 1j * hilb

  if list(env):
    A = env
  else: 
    A = np.abs(analytic)

  phase = np.angle(analytic,deg=False)
  ifreq = 1./(2*pi) * np.gradient(np.unwrap(phase)) * 1./dt

  gradA = np.gradient(A) * 1./dt
  ibw = np.sqrt((1./(2*pi*A) * gradA)**2)

  fny = 1./(2*dt)
  k = 50
  normEnv = np.exp(k*aire(gradA[ponset:tend]*1./(fny*A[ponset:tend]))*dt)

  time = np.arange(len(A))*dt

  if ponset != 0 or tend != 0:
    val, tval = window_p(ibw,time,ponset,tend,befp=10,opt='max')
  else:
    val = []

  if plot:
    import matplotlib.pyplot as plt
    fig = plt.figure()
    fig.set_facecolor('white')
    ax1 = fig.add_subplot(211,title='Analytic signal')
    ax1.plot(time,trace,'k',lw=2.,label='real')
    ax1.plot(time,hilb,'r',label='imag')
    ax1.plot(time,A,'g',label='envelope')
    ax1.legend()
    ax1.set_xticklabels('')
    ax2 = fig.add_subplot(212,title='Instantaneous bandwidth and frequency')
    ax2.plot(time[10:-10],ibw[10:-10],'k')
    ax2.plot(time[10:],ifreq[10:],'g')
    #ax2.plot(tval,val,'r')
    ax2.set_xlabel('Time (s)')
    plt.show()

  return val, normEnv


# ***********************************************************


def centroid_time(trace,dt,TF,ponset,tend=0,plot=False):
  """
  Returns the centroid time, i.e. the relative fraction of window length T at which half of area 
  below envelope is reached.
  from Hammer et al. 2012
  """
  
  from math import pi
  # Compute Hilbert transform of the signal
  freqs = np.fft.fftfreq(len(TF),d=dt)
  hilb = np.fft.ifft(-1j * np.sign(freqs) * TF)
  analytic = trace + 1j * hilb
  # Compute the amplitude envelope of the signal
  A = np.abs(analytic)

  befp = 50
  if ponset-befp < 0:
    p1 = 0
  else:
    p1 = ponset-befp
  if tend + befp > len(trace) or tend == 0:
    B = A[p1:]
  else:
    B = A[p1:tend]
  integral = aire(B)
  T = len(B)

  vals = np.array([])
  a=0
  for i in range(T):
    a = a + B[i]
    vals = np.append(vals,np.abs(a-0.5*integral))

  C = float(np.argmin(vals))/T

  if plot:
    time = np.arange(len(trace)) * dt
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(9,5))
    fig.set_facecolor('white')
    plt.plot(time,trace,'k')
    #plt.plot(time[ponset-50:ponset+dur*1./dt+50],trace[ponset-50:ponset+dur*1./dt+50],'b')
    plt.plot(time,A,'r')
    plt.plot([(ponset+C*T)*dt,(ponset+C*T)*dt],[np.min(trace),np.max(trace)],'g',lw=3,label='Centroid')
    plt.plot([ponset*dt,ponset*dt],[np.min(trace),np.max(trace)],'c',lw=3,label='Ponset')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.show()

  return C
   
 
# ***********************************************************


def polarization_analysis(filenames,dur,ponset,plot=False):
  """
  Returns:
  - the planarity
  - the rectilinearity
  - the largest eigenvalue
  from Hammer et al. 2012
  """
  from obspy.core import read

  st_n = read(filenames[0])
  tr_n = st_n[0].data
  st_e = read(filenames[1])
  tr_e = st_e[0].data
  st_z = read(filenames[2])
  tr_z = st_z[0].data
  delta = st_z[0].stats.delta

  if not tr_n.any() and not tr_n.all():
    return np.nan, np.nan, np.nan
  if not tr_e.any() and not tr_e.all():
    return np.nan, np.nan, np.nan
  if not tr_z.any() and not tr_z.all():
    return np.nan, np.nan, np.nan

  while len(tr_e) != len(tr_z) or len(tr_n) != len(tr_z) or len(tr_e) != len(tr_n):
    print len(tr_z),len(tr_e),len(tr_n)
    stt_z = st_z[0].stats.starttime
    stt_e = st_e[0].stats.starttime
    stt_n = st_n[0].stats.starttime
    if stt_z == stt_e and stt_z == stt_n:
      lens = np.array([len(tr_z),len(tr_e),len(tr_n)])
      l = np.min(lens)
      tr_z = tr_z[:l]
      tr_e = tr_e[:l]
      tr_n = tr_n[:l]
    else:
      stts = np.array([stt_z,stt_e,stt_n])
      stt_max = np.max(stts)
      tr_z = tr_z[(stt_max-stt_z)*1./delta:]
      tr_e = tr_e[(stt_max-stt_e)*1./delta:]
      tr_n = tr_n[(stt_max-stt_n)*1./delta:]

  cov_mat = np.cov((tr_z,tr_n,tr_e))
  vals, vecs = np.linalg.eig(cov_mat)

  vals_sort,vecs_sort = np.sort(vals)[::-1], np.array([])
  for i in range(len(vals)):
    ind = np.where(vals == vals_sort[i])[0]
    vecs_sort = np.append(vecs_sort,vecs[ind])
  vecs_sort = np.reshape(vecs_sort,vecs.shape)

  rect = 1 - (vals_sort[1]+vals_sort[2])/(2*vals_sort[0])
  plan = 1 - 2*vals_sort[2]/(vals_sort[1]+vals_sort[0])

  az = np.arctan(vecs_sort[1][0]*np.sign(vecs_sort[0][0])/(vecs_sort[2][0]*np.sign(vecs_sort[0][0])))
  iang = np.arccos(vecs_sort[0][0])

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

    # plot
    fig = plt.figure()
    fig.set_facecolor('white')

    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(x, y, z, rstride=4, cstride=4, color='k', alpha=0.2)
    ax.set_xlabel('Z')
    ax.set_ylabel('N')
    ax.set_zlabel('E')
    ax.set_title('Polarisation ellipsoid')

    i1, i2 = ponset-100, ponset+int(dur*1./st_z[0].stats.delta)
    plot_particle_motion(tr_z,tr_e,tr_n,filenames[0].split('/')[4],i1,i2)
    plt.show()

  return rect, plan, az, iang


def plot_particle_motion(tr_z,tr_e,tr_n,station,i1,i2):
  """
  Plots particle motion
  """

  import matplotlib.pyplot as plt
  import matplotlib.gridspec as gridspec
  G = gridspec.GridSpec(6,4)

  fig = plt.figure(figsize=(12,5))
  fig.set_facecolor('white')

  ax1 = fig.add_subplot(G[:2,:2],title='Comp Z')
  ax1.plot(tr_z,'k')
  ax1.plot(range(i1,i2),tr_z[i1:i2],'r')
  ax1.set_xticklabels('')

  ax2 = fig.add_subplot(G[2:4,:2],title='Comp N')
  ax2.plot(tr_n,'k')
  ax2.plot(range(i1,i2),tr_n[i1:i2],'r')
  ax2.set_xticklabels('')

  ax3 = fig.add_subplot(G[4:,:2],title='Comp E')
  ax3.plot(tr_e,'k')
  ax3.plot(range(i1,i2),tr_e[i1:i2],'r')

  ax1 = fig.add_subplot(G[:3,2])
  ax1.plot(tr_n[i1:i2],tr_z[i1:i2],'k')
  ax1.set_xlabel('N')
  ax1.set_ylabel('Z')

  ax2 = fig.add_subplot(G[3:,2])
  ax2.plot(tr_n[i1:i2],tr_e[i1:i2],'k')
  ax2.set_xlabel('N')
  ax2.set_ylabel('E')

  ax3 = fig.add_subplot(G[:3,3])
  ax3.plot(tr_z[i1:i2],tr_e[i1:i2],'k')
  ax3.set_xlabel('Z')
  ax3.set_ylabel('E')
 
  fig.suptitle(station)


# ***********************************************************


def window_p(feat,time,ponset,tend=0,befp=50,opt='mean'):
  if ponset-befp < 0:
    p1 = 0
  else:
    p1 = ponset-befp
  if tend == 0 or tend+befp >= len(time):
    time_p = time[p1:]-time[p1]
  else:
    time_p = time[p1:tend+befp]-time[p1]
  rel_time = time_p*1./time_p[-1]
  n = 1
  b1 = p1
  val,tval = [],[]
  while n <= 10:
    b2 = np.where(rel_time<n*.1)[0][-1] + p1
    if opt == 'mean':
      val.append(np.mean(feat[b1:b2]))
    elif opt == 'max':
      val.append(np.max(feat[b1:b2]))
    tval.append(np.mean(time[b1:b2]))
    n+=1
    b1 = b2
  return val, tval


# ***********************************************************


def aire(x):
  """
  Computes an integral using the trapezoidal method
  """
  a = 0
  i = 0
  dx = 1
  while i < len(x)-1:
    a = a + dx*(x[i]+x[i+1])/2.
    i = i + dx
  return a


def cum(x,plot=False):
  s=np.cumsum(x)
  p=np.polyfit(range(len(x)), s, deg=1)
  line=p[0]*np.arange(len(x))+p[1]
  l=s-line

  mins = find_local_min(l)

  if plot:
    import matplotlib.pyplot as plt
    fig = plt.figure()
    fig.set_facecolor('white')
    plt.plot(x/np.max(x),'k')
    plt.plot(s/np.max(s),'y')
    plt.plot(line/np.max(line),'b')
    plt.plot(l/np.max(l),'r')
    plt.plot(mins,l[mins]/np.max(l),'ro')
    plt.show()

  return l
