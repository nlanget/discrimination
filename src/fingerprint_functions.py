import numpy as np
import glob, sys, os
import matplotlib.pyplot as plt
import pandas as pd
import cPickle

def spectrogram(trace,param=[200.,.6,2]):
  """
  Adapted from obspy.imaging.spectrogram.spectrogram.py
  """
  from matplotlib import mlab
  from waveform_features import nearest_pow2
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
  print specgram.shape
  return (specgram,f,time,end)


def smallest_pow2(x):
  from math import pow,ceil,floor
  return pow(2,floor(np.log2(x)))

# ***********************************************************

def ponset_stack(x,grad,time,plot=False):
  pas = .01
  t = np.arange(0,len(x)*pas,pas)
  ponset = np.argmax(grad)
  if plot:
    fig = plt.figure(figsize=(9,4))
    fig.set_facecolor('white')
    plt.plot(t,x/np.max(np.abs(x)),'k')
    plt.plot(t,np.abs(grad/np.max(np.abs(grad))),'r')
    plt.plot([t[ponset],t[ponset]],[np.min(x/np.max(np.abs(x))),np.max(x/np.max(np.abs(x)))],'g-',lw=2.)
    plt.xlabel('Time (s)')
    ponset = np.argmin(np.abs(time-t[ponset]))
  return ponset

# ***********************************************************
def DecompositionStep(c):
  """
  Computation of the normalized coefficients in the Haar basis.
  From E.J. Stollnitz, T.D. DeRose and D.H. Salesin 
  "Wavelets for Computer Graphics: a primer" (1995, IEEE)

  :type c: numpy array
  """
  scale = np.sqrt(2)
  new_c = c.copy()
  for i in range(len(c)/2):
    new_c[i] = (c[2*i]+c[2*i+1])*1./scale
    new_c[len(c)/2+i] = (c[2*i]-c[2*i+1])*1./scale
  return new_c


def Decomposition(d):
  c = d.copy()
  h = len(c)
  #c = c*1./np.sqrt(h) # Normalize input coefficients
  while h > 1:
    c[:h] = DecompositionStep(c[:h])
    h = h/2
  return c


def StandardDecomposition(D,plot=False):
  C = D.copy()
  # Transform on the rows
  for row in range(np.size(C,0)):
    C[row,:] = Decomposition(C[row,:])
  if plot:
    fig = plt.figure()
    fig.set_facecolor('white')
    plt.imshow(np.log10(np.abs(C)),cmap=plt.cm.jet)
    #print C
  # Transform on the columns
  for col in range(np.size(C,1)):
    C[:,col] = Decomposition(C[:,col])
  if plot:
    #print C
    fig = plt.figure()
    fig.set_facecolor('white')
    plt.imshow(np.log10(np.abs(C)),cmap=plt.cm.jet)
  return C


def NonStandardDecomposition(D):
  C = D.copy()
  h = np.size(C,0)
  w = np.size(C,1)
  C = C/h # normalize input coefficients
  while h > 1:
    for row in range(np.size(C,0)):
      C[row,:w] = DecompositionStep(C[row,:w])
    for col in range(np.size(C,1)):
      C[:h,col] = DecompositionStep(C[:h,col])
    h = h/2
    w = w/2
    #fig = plt.figure(figsize=(9,4))
    #fig.set_facecolor('white')
    #plt.imshow(C,cmap=plt.cm.jet)
    #plt.show()
  return C


def ihaar(a):
  """
  Inverse Haar transform
  from E. Mikulic
  """
  scale = np.sqrt(2)
  if len(a) == 1:
    return a.copy()
  assert len(a) % 2 == 0, "length needs to be even"
  mid = ihaar(a[0:len(a)/2]) * scale
  side = a[len(a)/2:] * scale
  out = np.zeros(len(a), dtype=float)
  out[0::2] = (mid + side) / 2.
  out[1::2] = (mid - side) / 2.
  return out

def ihaar_2d(coeffs):
  """
  Inverse Haar transform
  from E. Mikulic
  """
  h,w = coeffs.shape
  cols = np.zeros(coeffs.shape, dtype=float)
  for x in range(w):
    cols[:,x] = ihaar(coeffs[:,x])
  rows = np.zeros(coeffs.shape, dtype=float)
  for y in range(h):
    rows[y] = ihaar(cols[y])
  return rows


def TopWavelets(mat,plot=False):
  # Selection of the top wavelet coefficients (wrt their magnitude)
  e = [0]
  mat_abs = np.abs(mat).ravel()
  e_all = np.cumsum(mat_abs)[-1]
  vec = np.arange(0,1.01,.01)
  for i in vec[1:]:
    t = np.floor(i*mat.shape[0]*mat.shape[1])
    a = np.array([])
    a = np.cumsum(mat_abs[np.argsort(mat_abs)[-t:]])
    if a.all() and a.any():
      e.append(a[-1]/e_all)
  e = np.array(e)

  if plot:
    fig = plt.figure()
    fig.set_facecolor('white')
    plt.plot(vec,e,'k')
    plt.xlabel('Fraction of wavelet coefficients retained')
    plt.ylabel('Fraction of energy retained')

  # Determination of the number of top wavelet coefficients retained
  b = np.where(e>0.95)[0][0]
  print "Proportion of maximum magnitude wavelet coefficients : ",vec[b]
  t = np.floor(vec[b]*mat.shape[0]*mat.shape[1])
  return t


def FuncFingerprint(mat,time,tr,f,end,plot=False,error=False):

  haar = StandardDecomposition(mat)

  if error:
    MSE,CORR,ENE = [],[],[]
    x = np.arange(0.001,0.101,0.001)
    #x = np.arange(0.01,1.01,0.01)
    ene_tot = np.cumsum(np.abs(haar))[-1]
    for a in x:
      t = np.floor(a*haar.shape[0]*haar.shape[1])
      haar_top = haar.copy()
      haar_top.ravel()[np.argsort(np.abs(haar_top.ravel()))[:-t]] = 0
      lossy = ihaar_2d(haar_top)

      # Compute the mean squared error
      MSE.append(np.mean((lossy-mat)**2))
      # Compute the correlation coefficient
      ml = np.mean(lossy)
      mm = np.mean(mat)
      cov = np.mean((lossy-ml)*(mat-mm))
      stdl = np.sqrt(np.mean((lossy-ml)**2))
      stdm = np.sqrt(np.mean((mat-mm)**2))
      CORR.append(cov*1./(stdl*stdm))
      # Compute the fraction of the total energy
      ENE.append(np.cumsum(np.abs(haar_top))[-1]*1./ene_tot)


  #t = TopWavelets(haar,plot=True)
  # Fix t
  frac = .01
  t = np.floor(frac*haar.shape[0]*haar.shape[1])
  print "Number of maximum magnitude wavelet coefficients : ", int(t),haar.shape[0]*haar.shape[1]
  haar_top = haar.copy()
  haar_top.ravel()[np.argsort(np.abs(haar_top.ravel()))[:-t]] = 0
  # Inverse Haar transform
  lossy = ihaar_2d(haar_top) 
  # Keep the sign of top wavelet coefficients only
  haar_sign = np.sign(haar_top)
  haar_sign = np.array(map(int,haar_sign.ravel())).reshape(haar_top.shape[0],haar_top.shape[1])
  # Binarization
  #from sklearn.preprocessing import Binarizer
  #haar_bin = Binarizer(threshold=0).fit(haar_sign).transform(haar_sign)
  haar_bin = np.abs(haar_sign)

  if plot:
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(12,7))
    fig.set_facecolor('white')

    halfbin_time = (time[1] - time[0]) / 2.0
    halfbin_freq = (f[1] - f[0]) / 2.0
    extent = (time[0] - halfbin_time, time[-1] + halfbin_time, f[0] - halfbin_freq, f[-1] + halfbin_freq)

    gs1 = gridspec.GridSpec(3,1)
    ax1 = fig.add_subplot(gs1[0])
    ax1.plot(tr,'k')
    ax1.set_xlim([time[0]*100,time[-1]*100])
    ax1.set_xticklabels('')
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    axins = inset_axes(ax1,width="30%",height="30%",loc=4)
    xx = range(len(tr))
    axins.plot(xx,tr,'k')
    axins.plot(xx[int(time[0]*100):int(time[-1]*100)],tr[time[0]*100:time[-1]*100],'r')
    axins.xaxis.set_ticklabels('')
    axins.yaxis.set_ticklabels('')
    #axins.set_frame_on(False)
    axins.xaxis.set_visible(False)
    axins.yaxis.set_visible(False)
    ax1.set_title('Raw signal')

    ax2 = fig.add_subplot(gs1[1])
    ax2.imshow(mat,cmap=plt.cm.jet,extent=extent)#,interpolation="nearest")
    ax2.set_xlim([0,end])
    ax2.axis('tight')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_title('Spectrogram')
    print "duration : %f s"%(time[-1]-time[0])

    ax3 = fig.add_subplot(gs1[2])
    extent = (0,len(time),0,len(f))
    #ax3.imshow(np.log10(np.abs(haar)),cmap=plt.cm.jet,extent=extent)
    ax3.imshow(lossy,cmap=plt.cm.jet,extent=extent)
    #ax3.imshow(np.flipud(haar_sign),cmap=plt.cm.gray,extent=extent)
    ax3.set_xlim([0,len(time)])
    ax3.axis('tight')
    ax3.set_title('Lossy spectrogram')
    ax3.set_xlabel('Wavelet coefficient index')
    ax3.set_ylabel('Wavelet coefficient index')

    gs1.tight_layout(fig, rect=[None, None, 0.45, None])

    gs2 = gridspec.GridSpec(1,2)
    ax4 = fig.add_subplot(gs2[0,0])
    ax4.imshow(mat,cmap=plt.cm.jet)#,interpolation="nearest")
    ax4.set_title('Spectrogram')
    
    ax5 = fig.add_subplot(gs2[0,1])
    ax5.imshow(np.flipud(haar_sign),cmap=plt.cm.gray)
    ax5.set_yticklabels('')
    ax5.set_title('Sign of top wavelet coefficients')
    gs2.tight_layout(fig, rect=[0.45, 0.50, None, None])

    gs3 = gridspec.GridSpec(1,2)
    ax9 = fig.add_subplot(gs3[0,0])
    ax9.imshow(np.flipud(haar_bin),cmap=plt.cm.gray)
    ax9.set_title('Fingerprint')

    if error:
      ax6 = fig.add_subplot(gs3[0,1])
      ax6.plot(x*100,MSE,'b')
      ax6.set_xlabel('% of wavelet coefficients retained')
      ax6.set_ylabel('Mean squared error',color='b')
      for tl in ax6.get_yticklabels():
        tl.set_color('b')
      ax7 = ax6.twinx()
      ax7.plot(x*100,CORR,'r')
      ax7.set_ylabel('Correlation coefficient',color='r')
      for tl in ax7.get_yticklabels():
        tl.set_color('r')
      ax7.plot([frac*100,frac*100],[np.min(CORR),np.max(CORR)],'g',lw=2.)
      #ax8 = ax6.twinx()
      #ax8.plot(x*100,ENE,'y--')
      #ax8.set_ylabel('Fraction of energy retained',color='y')
      #for tl in ax8.get_yticklabels():
      #  tl.set_color('y')
    gs3.tight_layout(fig, rect=[0.45, None, 0.96, 0.48])

    top = min(gs1.top, gs2.top)
    bottom = max(gs1.bottom, gs3.bottom)

    gs1.update(top=top, bottom=bottom)
    gs2.update(top=top)
    gs3.update(bottom=bottom)

    #plt.savefig('Pics/event_8.png')

  return haar_bin

# ***********************************************************

def define_permutation(size,Nperm):
  permut = []
  for i in range(Nperm):
    permut.append(np.random.permutation(size))
  return permut


def vec_compute_signature(vec,permut):
  permut = np.array(permut)
  nb_perm = permut.shape[0]  # number of permutations
  l = len(vec)  # length of the vector
  
  if permut.shape[1] != l:
    print "Warning - check lengths"
    sys.exit()

  MH_sign = np.zeros(nb_perm,dtype=int)
  for i in range(nb_perm):
    permut_vec = vec[permut[i]]
    MH_sign[i] = np.where(permut_vec==1)[0][0]

  return MH_sign


def hash_function(liste):
  h = 0
  for i in range(len(liste)):
    h = h + (i+1)*liste[i]
  return h


def LSH(vec,l=50):
  """
  l =  number of hash tables
  """
  k = len(vec)  # length of the vector

  if k%l != 0:
    print "Choose a number of hash tables l which is a multiple of the number of rows k !"
    l = int(raw_input("k = %d ; l = ?\n"%k))
  r = k/l

  Hash_tables = np.zeros(l,dtype=int)
  for i in range(l):
    Hash_tables[i] = hash_function(vec[i*r:(i+1)*r])
    #Hash_tables[i] = np.sum(vec[i*r:(i+1)*r])

  return Hash_tables


def histograms(mat):
  m = mat.shape[1]
  l = mat.shape[0]
  for i in range(l):
    uni = len(np.unique(mat[i,:]))
    fig = plt.figure()
    fig.set_facecolor('white')
    plt.hist(mat[i,:],uni)
    plt.xlabel('Hash value')
    plt.ylabel('Number of items')
    plt.title('Table %d'%i)
    plt.show()


def read_MH(filename,test=False):
  if not test:
    filename = 'train_%s'%filename
  with open(filename,'rb') as file:
    my_depickler = cPickle.Unpickler(file)
    dic_mh = my_depickler.load()
    file.close()

  dic_ht = {}
  for key in sorted(dic_mh):
    MH_sign = dic_mh[key]
    HashTab = LSH(MH_sign,l=50)
    dic_ht[key] = HashTab

  savefile = 'hash_tables_8'
  if not test: 
    savefile = 'train_%s'%savefile
  with open(savefile,'wb') as file:
    my_pickler = cPickle.Pickler(file)
    my_pickler.dump(dic_ht)
    file.close()

