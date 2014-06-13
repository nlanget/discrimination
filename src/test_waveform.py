import os, unittest
import numpy as np
from waveform_io import *

def suite():
    suite = unittest.TestSuite()
    suite.addTest(waveformTests('test_waveform_read'))
    suite.addTest(waveformTests('test_kurtogram'))
    suite.addTest(waveformTests('test_filter'))
    suite.addTest(waveformTests('test_positive_gradient'))
    suite.addTest(waveformTests('test_trigger'))
    suite.addTest(waveformTests('test_trigger_data'))
    suite.addTest(waveformTests('test_extract_data'))
    return suite

#@unittest.skip('Not bothering with this test')
class waveformTests(unittest.TestCase):

    def setUp(self): 
        from obspy.core import read 

        self.filename = os.path.join('test_data','YA.UV15.00.HHZ.MSEED' ) 
        self.st = read(self.filename)
        self.st.detrend(type='linear')
        self.tr=self.st[0]

    def test_waveform_read(self):
        tr=self.st[0]
        starttime = tr.stats.starttime
        npts = len(tr.data)
        delta = tr.stats.delta
        sampling_rate = tr.stats.sampling_rate

        self.assertEquals(starttime.isoformat(), '2010-10-14T00:14:00')
        self.assertEquals(npts, 24001)
        self.assertEquals(delta, 0.01)
        self.assertEquals(sampling_rate, 100.)

    def test_kurtogram(self):
        x=np.empty(len(self.tr.data))
        x[:]  = self.tr.data[:]
        dt = self.tr.stats.delta

        fc, bw = get_kurtogram_filter_parameters(x,dt)
        self.assertAlmostEquals(fc, 30.20833, 4)
        self.assertAlmostEquals(bw,  2.08333, 4)
        
    def test_filter(self):
        from scipy.signal import chirp
        tr=self.tr

        # generate test signal
        npts= len(self.tr.data)
        dt  = tr.stats.delta
        t   = np.arange(npts)*dt
        f0 = 0.5
        f1 = 40.
        t1 = max(t)
        # set up a linear chirp dataset
        tr.data = chirp(t,f0, t1, f1) * 100.0
        f   = f0 + (f1 - f0) * t / t1

        # do filter
        fc=30.208
        bw=2.0833
        gaussian_filter_trace(tr,fc,bw)

        # check frequency of maximum is within bounds
        itmax=np.argmax(tr.data)
        fmax=f[itmax]
        self.assertGreater(fmax,fc-bw/2.)
        self.assertLess(fmax,fc+bw/2.)

    def test_positive_gradient(self):
        npts= len(self.tr.data)
        dt  = self.tr.stats.delta
        x = np.arange(npts)*dt

        # set up a polynomial function
        y = (3 + 2*x +4*x*x +5*x*x*x)
        dy_exp = (2 + 8*x +15*x*x)

        self.tr.data=y
        positive_gradient_trace(self.tr)

        np.testing.assert_almost_equal(self.tr.data[20:100], dy_exp[20:100],2)

    def test_trigger(self):
        tr=self.tr
        npts= len(tr.data)
        dt  = tr.stats.delta
        x = np.ones(npts)
        starttime = tr.stats.starttime

        # test spikes at start and end of signal (difficult)
        x[0]=10
        x[-1]=10
        snr=5
        tr.data=x
        times=get_max_trigger_times(tr,snr)

        self.assertEquals(len(times),2)
        self.assertAlmostEquals(times[0]-starttime, 0.0)
        self.assertAlmostEquals(times[1]-starttime, (npts-1)*dt)


    @unittest.expectedFailure
    def test_trigger_data(self):
        tr = self.tr
        dt = tr.stats.delta
        x  = tr.data
        kwin = 4.0
        snr = 3.0

        fc, bw = get_kurtogram_filter_parameters(x, dt)
        gaussian_filter_trace(tr, fc, bw)
        kurtosis_trace(tr, kwin, rec=True)
        positive_gradient_trace(tr)
        times = get_max_trigger_times(tr,snr)
        self.assertEquals(len(times),3)

    @unittest.skip('Skipping data extraction test')
    def test_extract_data(self):
        from copy import deepcopy

        tr = self.tr
        tr_data=tr.copy()
        dt = tr.stats.delta
        x  = tr.data
        kwin = 4.0
        snr = 3.0

        fc, bw = get_kurtogram_filter_parameters(x, dt)
        gaussian_filter_trace(tr, fc, bw)
        kurtosis_trace(tr, kwin, rec=True)
        positive_gradient_trace(tr)
        times = get_max_trigger_times(tr,snr)

        tr_data.filter('bandpass',freqmin=4.0, freqmax=10.0)
        #gaussian_filter_trace(tr_data, fc, bw)
        events = extract_events_by_envelope(tr_data,times,lag_time=300.0)
        self.assertEquals(len(events),2)
        for val in events.values():
            val.plot()



if __name__ == '__main__':

  import logging
  logging.basicConfig(level=logging.INFO, format='%(levelname)s : %(asctime)s : %(message)s')
 
  unittest.TextTestRunner(verbosity=2).run(suite())
 
