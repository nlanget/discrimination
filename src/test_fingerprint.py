from fingerprint_functions import *
import numpy as np
from math import pi
import matplotlib.pyplot as plt
import unittest


def suite():
  suite = unittest.TestSuite()
  suite.addTest(HaarTransformTest('test_1d'))
  suite.addTest(HaarTransformTest('test_2d'))
  return suite


class HaarTransformTest(unittest.TestCase):


  def test_1d_norm(self):
    a = np.array([9,7,3,5])
    btrue = np.array([6,2,1./np.sqrt(2),-1./np.sqrt(2)])

    b = Decomposition(a)
    for i in range(len(a)):
      self.assertAlmostEqual(b[i],btrue[i],3)


  def test_1d(self):
    output = example_2()
    with open('../lib/Test/1d_output.txt','r') as fichier:
      lines = fichier.readlines()
      fichier.close()
    for i,line in enumerate(lines):
      self.assertAlmostEqual(float(line),output[i],3)


  def test_2d(self):
    output = example_3()
    with open('../lib/Test/2d_output.txt','r') as fichier:
      lines = fichier.readlines()
      fichier.close()
    for i,line in enumerate(lines):
      l = line.split(',')
      for j in range(len(l)):
        self.assertAlmostEqual(float(l[j]),output[i][j],3)

# ----------------------------------------------------------------------------

def example_1():
  """
  Example 1 (2d)
  Diagonal 32x32 matrix with ones on the diagonal
  """
  mat = np.diag(np.ones(32))
  trans = StandardDecomposition(mat)


def example_2(save=False,plot=False):
  """
  Example 2 (1d)
  Let x in  [0,1] be discretized by 32 grid points uniformly (this should give you the grid spacing h = 0.032258064516129). Set u = sin(2*pi*x) as the input data.
  """
  x = np.arange(0,1,1./31)
  u = np.sin(2*pi*x)
  u = np.append(u,0)
  trans = Decomposition(u)

  if save:
    with open('../results/Test/my_1d_output.txt','w') as file:
      for i in range(len(trans)):
        file.write("%f\n"%trans[i])
    file.close()

  if plot:
    fig = plt.figure(figsize=(9,3))
    fig.set_facecolor('white')
    ax1 = fig.add_subplot(121)
    ax1.plot(u,'k')

    ax2 = fig.add_subplot(122)
    ax2.plot(trans,'k')
    plt.show()

  return trans


def example_3(save=False,plot=False):
  """
  Example 3 (2d)
  Let (x,y) in [0,1] x [0,1] be discretized by 32 x 32 grid points uniformly (this should give you the grid spacings hx = hx = h). Set u = sin(2*pi*x)*sin(2*pi*y) as the input data.
  """
  x = np.arange(0,1,1./31)
  x = np.append(x,0)
  x = np.array([x])

  y = np.arange(0,1,1./31)
  y = np.append(y,0)
  y = np.array([y])

  u = np.dot(np.sin(2*pi*x).T,np.sin(2*pi*y))
  trans = StandardDecomposition(u)


  if save:
    with open('../results/Test/my_2d_output.txt','w') as file:
      for i in range(trans.shape[0]):
        for j in range(trans.shape[1]):
          file.write("%f "%trans[i][j])
        file.write("\n")
    file.close()


  if plot:
    fig = plt.figure(figsize=(12,4))
    fig.set_facecolor('white')
    ax1 = fig.add_subplot(141)
    ax1.imshow(np.flipud(u),cmap=plt.cm.jet)
    ax1.set_title('Initial image')
    ax1.invert_yaxis()

    ax2 = fig.add_subplot(142)
    ax2.imshow(np.flipud(trans),cmap=plt.cm.jet)
    ax2.set_title('Haar transform')
    ax2.set_yticklabels([])

    from haar_lossy import ihaar_2d
    t = np.floor(.05*trans.shape[0]*trans.shape[1])
    compress = trans.copy()
    compress.ravel()[np.argsort(np.abs(trans.ravel()))[:-t]] = 0
    lossy = ihaar_2d(compress)

    ax3 = fig.add_subplot(143)
    compress[compress!=0] = 1
    ax3.imshow(np.flipud(compress),cmap=plt.cm.gray)
    ax3.set_title('Fingerprint')
    ax3.set_yticklabels([])

    ax4 = fig.add_subplot(144)
    ax4.imshow(lossy,cmap=plt.cm.jet)
    ax4.set_title('Compressed image')
    ax4.set_yticklabels([])

    plt.figtext(.11,.77,'(a)')
    plt.figtext(.31,.77,'(b)')
    plt.figtext(.51,.77,'(c)')
    plt.figtext(.71,.77,'(d)')
    plt.savefig('../results/Test/figures/auto/haar_trans.png')
    plt.show()

  return trans

# ----------------------------------------------------------------------------

if __name__ == '__main__':
  unittest.TextTestRunner(verbosity=2).run(suite())
