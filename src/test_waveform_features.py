from waveform_features import *
import numpy as np
import os, unittest,sys
from obspy.core import read, utcdatetime
from math import exp, sqrt, pi
from random import gauss

def suite():
  suite = unittest.TestSuite()
  #suite.addTest(FeatureTests('test_energy_between_10Hz_and_30Hz'))
  #suite.addTest(FeatureTests('test_gaussian_kurtosis'))
  #suite.addTest(FeatureTests('test_max_over_mean'))
  suite.addTest(FeatureTests('test_signal_duration_and_growth_over_decay'))
  #suite.addTest(FeatureTests('test_spectral_features'))
  #suite.addTest(FeatureTests('test_cepstrum'))
  #suite.addTest(FeatureTests('test_instantaneous_bandwidth'))
  #suite.addTest(FeatureTests('test_centroid_time'))
  #suite.addTest(FeatureTests('test_polarization'))
  #suite.addTest(TestAnalyticFeatures('test_sin'))
  #suite.addTest(TestAnalyticFeatures('test_linear_mod'))
  #suite.addTest(TestAnalyticFeatures('test_sum_sin'))
  #suite.addTest(TestRicker('test_ricker'))
  #suite.addTest(TestRicker('test_trace'))
  return suite



class FeatureTests(unittest.TestCase):

  def setUp(self):
    self.filename = os.path.join('../data/Test','YA.UV15.00.HHZ.MSEED' )
    self.st = read(self.filename)
    self.tr = self.st[0]

    #import matplotlib.pyplot as plt
    #fig = plt.figure()
    #fig.set_facecolor('white')
    #plt.plot(self.tr.data)
    #plt.show()

    # Compute a ricker
    sig = .1
    self.dt = .01
    t = np.arange(-1,1,self.dt)
    self.ricker = 2/(pi**(1/4)*np.sqrt(3*sig))*(1-t**2/sig**2)*np.exp(-t**2/(2*sig**2))

    # Sinusoidal function
    self.F = 1 
    t = np.arange(0,10,self.dt)
    self.sin_func = np.sin(2*pi*self.F*t)



  def test_energy_between_10Hz_and_30Hz(self):
    dt = self.tr.stats.delta
    data = np.array([40*dt*np.sinc(40*i*dt) for i in range(-10000,10000)])
    Energy = energy_between_10Hz_and_30Hz(data,dt)
    EnergyTest = 10
    self.assertAlmostEqual(Energy, EnergyTest, 2)


  def test_signal_duration_and_growth_over_decay(self):
    st_filt = read('../data/Test/synth.sac')
    self.tr_filt = st_filt[0]

    st_env = read('../data/Test/synth_env.sac')
    self.tr_env = st_env[0]

    st_grad = read('../data/Test/synth_grad.sac')
    self.tr_grad = st_grad[0]

    #self.ponset,self.tend,dur = signal_duration(self,it0=1500,plot=False)
    #self.assertAlmostEquals(self.ponset, 1600, delta=5)
    #self.assertAlmostEquals(dur, 6, delta=1)

    self.ponset = 1600
    self.tend = len(self.tr_filt)-1
    GrowthOverDecay = growth_over_decay(self)
    GrowthOverDecayTest = 2   
    #self.assertEqual(GrowthOverDecay, GrowthOverDecayTest)


  def test_gaussian_kurtosis(self):
    data = np.array([gauss(5,1) for i in range(100000)])
    from obspy.signal import filter
    env = filter.envelope(data)
    KurtosisEnvelope = kurtosis_envelope(env)
    KurtosisEnvelopeTest = 3
    self.assertAlmostEquals(KurtosisEnvelope, KurtosisEnvelopeTest, places = 1)


  def test_max_over_mean(self):
    data = np.array([1,0,1,0,1,0])
    MaxOverMean = max_over_mean(data)
    MaxOverMeanTest = 2
    self.assertEqual(MaxOverMean, MaxOverMeanTest)


  def test_spectral_features(self):
    from math import pi
    F = 1
    dt = .01
    t = np.arange(0,10,dt)
    tr_filt = np.cos(2*pi*F*t)

    TF = np.fft.fft(tr_filt)
    f = np.fft.fftfreq(len(TF),d=dt)
    TF = TF[:len(f)/2]
    f = f[:len(f)/2]

    predf, bandw, centralf = around_freq(TF,f,plot=False)

    self.assertEqual(predf,F)
    self.assertAlmostEquals(centralf,F,places=2)
    self.assertAlmostEquals(bandw,0,places=2)



  def test_cepstrum(self):
   # Create the seismic trace with an echo
    trace = np.ones(52*1./self.dt)
    trace[5*1./self.dt] = 10
    trace[25*1./self.dt] = 5
    trace = np.convolve(trace,self.ricker,mode='same')
    trace = trace[100:-100]
    trace = trace / np.max(trace)
    t = np.arange(0,len(trace)*self.dt,self.dt)

    import matplotlib.pyplot as plt
    fig = plt.figure()
    fig.set_facecolor('white')
    plt.plot(t,trace,'k')
    #plt.close()

    TF = np.fft.fft(trace)
    f = np.fft.fftfreq(len(TF),d=t[1]-t[0])
    cepstr = cepstrum(TF,f,plot=True)



  def test_instantaneous_bandwidth(self):
    data = self.sin_func

    #st_filt = read('../data/Test/synth.sac')
    #tr = st_filt[0]
    #dt = tr.stats.delta
    #data = tr.data[1500:2500]

    TF = np.fft.fft(data)
    ibw,normEnv = instant_bw(data,self.dt,TF,plot=False)
    self.assertAlmostEquals(np.mean(ibw),0,places=1)


  def test_centroid_time(self):
    #data = self.ricker
    data = self.sin_func
    TF = np.fft.fft(data)
    C = centroid_time(data,self.dt,TF,0,plot=True)
    self.assertAlmostEquals(C,(len(data)/2-1)*1./len(data),places=1)


  def test_polarization(self):
    self.mat = np.array([[1,0,0],[0,-1,1],[1,1,1]])
    l1, l2, l3 = np.sqrt(2), 1, -np.sqrt(2)
    expRect = 1 - (l2+l3)/(2*l1)
    expPlan = 1 - (2*l3)/(l1+l2)

    rect, plan, lambda_max = polarization_analysis(self.mat,plot=False)
    self.assertAlmostEquals(lambda_max,l1,places=6)
    self.assertAlmostEquals(rect,expRect,places=6)
    self.assertAlmostEquals(plan,expPlan,places=6)

    file = "/home/nadege/waveloc/data/Piton/2011-02-02/2011-02-02T00:00:00.YA.UV15.HHZ.filt.mseed"
    cmin = utcdatetime.UTCDateTime("2011-02-02T00:58:47.720000Z")-15
    cmax = utcdatetime.UTCDateTime("2011-02-02T00:58:47.720000Z")+135

    ponset = 1400 
    st_z = read(file,starttime=cmin,endtime=cmax)
    tr_z = st_z[0].data[ponset-10:ponset+30]
 
    file_n = "%s/HHN/*%s*HHN*.filt.*"%(os.path.dirname(file),st_z[0].stats.station)
    file_e = "%s/HHE/*%s*HHE*.filt.*"%(os.path.dirname(file),st_z[0].stats.station)
    st_n = read(file_n,starttime=cmin,endtime=cmax)
    st_e = read(file_e,starttime=cmin,endtime=cmax)
    tr_n = st_n[0].data[ponset-10:ponset+30]
    tr_e = st_e[0].data[ponset-10:ponset+30]

    x=np.array([tr_z,tr_n,tr_e])
    print tr_e.shape
    mat = np.cov(x)
    rect, plan, lambda_max = polarization_analysis(mat,plot=True)
    from obspy.signal.polarization import eigval
    leigenv1, leigenv2, leigenv3, rect, plan, dleigenv, drect, dplan = eigval(tr_e,tr_n,tr_z,[1,1,1,1,1])
    print lambda_max
    print leigenv1, leigenv2, leigenv3



class TestAnalyticFeatures(unittest.TestCase):

  def test_sin(self):
    from math import pi
    # Sinusoidal function
    F = 1
    dt = .01
    t = np.arange(0,10,dt)
    sin_func = np.sin(2*pi*F*t)

    TF = np.fft.fft(sin_func)
    vals = instant_freq(sin_func,dt,TF,plot=False)

    for i in range(len(vals)):
      self.assertAlmostEquals(vals[i],F,places=1)


  def test_linear_mod(self):
    from math import pi
    A = .1
    B = .0025
    dt = .5
    t = np.arange(0,60,dt)
    data = np.sin(2*pi*(A+B/2*t)*t)

    TF = np.fft.fft(data)
    vals = instant_freq(data,dt,TF,plot=True)

    print vals

    #iphase, pf = instant_freq(data,dt,TF,plot=False)
    #self.assertAlmostEquals(A,pf[1],places=1)
    #self.assertAlmostEquals(B,pf[0],places=1)


  def test_sum_sin(self):
    from math import pi
    f1 = .5
    f2 = 1.25
    dt = .01
    t = np.arange(0,10,dt)
    data = np.sin(2*pi*f1*t) + np.sin(2*pi*f2*t)

    TF = np.fft.fft(data)
    iphase, pf = instant_freq(data,dt,TF,plot=False)
    #self.assertAlmostEquals(iphase,F,places=1)


class TestRicker(unittest.TestCase):

  """
  For visual comparison, refer to Barnes 1993
  """

  def setUp(self):

    self.dt = .001
    tr = np.arange(-0.1,0.1,self.dt)
    sig = 0.0074
    ricker = 2/(pi**(1/4)*np.sqrt(3*sig))*(1-tr**2/sig**2)*np.exp(-tr**2/(2*sig**2))
    self.ricker = ricker/np.max(ricker)

    # Plot the spectrum
    import matplotlib.pyplot as plt
    tf = np.fft.fft(self.ricker)
    f = np.fft.fftfreq(len(tf),d=self.dt)
    tf = tf[:len(tf)/2]
    f = f[:len(f)/2]
    
    fig = plt.figure()
    fig.set_facecolor("white")
    ax1 = fig.add_subplot(211)
    ax1.plot(tr,self.ricker,'k')
    ax2 = fig.add_subplot(212)
    ax2.plot(f,np.abs(tf),'k')
    ax2.set_xlabel('Frequency')
    ax2.set_ylabel('abs(TF)')


  def test_ricker(self):

    from obspy.signal import filter
    env = filter.envelope(self.ricker)

    tf = np.fft.fft(self.ricker)
    ifreq = instant_freq(self.ricker, self.dt, tf, 50, 150, plot=True)
    ibw = instant_bw(self.ricker, env, self.dt, tf, plot=True)


  def test_trace(self):

    t = np.arange(0,0.5,self.dt)
    impuls = np.zeros(len(t))
    impuls[0.08/self.dt] = 1
    impuls[0.12/self.dt] = -1
    impuls[0.235/self.dt] = 1
    impuls[0.265/self.dt] = -1
    impuls[0.39/self.dt] = 1
    impuls[0.41/self.dt] = -1
    trace = np.convolve(impuls,self.ricker,mode='same')
    

    import matplotlib.pyplot as plt
    fig = plt.figure()
    fig.set_facecolor('white')
    ax1 = fig.add_subplot(211)
    ax1.plot(t,impuls,'k')
    ax2 = fig.add_subplot(212)
    ax2.plot(trace,'k')


    from obspy.signal import filter
    env = filter.envelope(trace)
    ifreq = instant_freq(trace, self.dt, np.fft.fft(trace), 0, len(trace)-1, plot=True)
    ibw = instant_bw(trace, env, self.dt, np.fft.fft(trace), ponset = 0, tend=len(trace), plot=True)


if __name__ == '__main__':

  import logging
  logging.basicConfig(level=logging.INFO, format='%(levelname)s : %(asctime)s : %(message)s')

  unittest.TextTestRunner(verbosity=2).run(suite())
