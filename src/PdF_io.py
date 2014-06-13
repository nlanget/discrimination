import os
import numpy as np
from numpy import log
from scipy.io.matlab import loadmat
from bayes_gaussian import data_to_eigen_projection_dD, \
        data_to_gaussian_dD, sigma_to_corr


def read_training_data():
    """
    Returns a dictionary of features for the training data
    """
    filename=os.path.join('..','data/Piton','TrainingSet_2.mat')
    data_orig=loadmat(filename)
    
    # create a clean dictionnary of data
    # taking logarithms of the features for which
    # the test set also has logarithms (thanks Clement!)
    
    # for now only deal with the two features that are ok in the two datasets
    data={}
    data['KurtoEB']=log(np.array(data_orig['KurtoEB'].flat))
    data['KurtoVT']=log(np.array(data_orig['KurtoVT'].flat))
    data['AsDecVT']=log(np.array(data_orig['AsDecVT'].flat))
    data['AsDecEB']=log(np.array(data_orig['AsDecEB'].flat))
    data['RappMaxMeanEB']=log(np.array(data_orig['RappMaxMeanEB'].flat))
    data['RappMaxMeanVT']=log(np.array(data_orig['RappMaxMeanVT'].flat))
    data['DurVT']=np.abs(np.array(data_orig['DurVT'].flat))
    data['DurEB']=np.abs(np.array(data_orig['DurEB'].flat))
    data['EneEB']=log(np.array(data_orig['EneFFTeB'].flat))
    data['EneVT']=log(np.array(data_orig['EneFFTvT'].flat))

    return data

def read_test_data():
    """
    Returns a dictionnary of features for test data, including OVPFID and
    Clement's probabilities.

    OVPFID : 1=EB, 2 = VT, 3 = deep VT
    ValMetaHypo : VT<0.5 / EB>0.5
    """

    filename=os.path.join('..','data/Piton','TestDataSet_2.mat')
    data_orig=loadmat(filename)

    data={}
    data['KurtoVal']=np.array(data_orig['KurtoVal'].flat)
    data['AsDecVal']=np.array(data_orig['AsDecVal'].flat)
    data['AsDecVal'][np.isinf(data['AsDecVal'])] = 0.0
    data['RappMaxMeanVal']=np.array(data_orig['RappMaxMeanVal'].flat)
    data['DurVal']=np.array(data_orig['DurationVal'].flat)
    data['EneVal']=np.array(data_orig['EneHighFreqVal'].flat)
    data['OVPFID']=np.array(data_orig['OVPFID'].flat)
    data['ValMetaHypo']=np.array(data_orig['ValMetaHYpo'].flat)

    return data

def make_uncorrelated_data(data,testdata):
    """
    Takes the raw data and test data and does required PCA
    """

    # Combine Kurto and RappMaxMean
    f_eb = ['KurtoEB', 'RappMaxMeanEB', 'AsDecEB', 'DurEB', 'EneEB'] ; 
    f_vt = ['KurtoVT', 'RappMaxMeanVT', 'AsDecVT', 'DurVT', 'EneVT']; 
    f_test = ['KurtoVal', 'RappMaxMeanVal', 'AsDecVal', 'DurVal', 'EneVal']

    # set up EB and VT matrices (nsamples x nfeatures)
    v_eb = [data[name] for name in f_eb]
    v_vt = [data[name] for name in f_vt]
    v_test = [testdata[name] for name in f_test]

    x_eb=np.vstack(tuple(v_eb)).T
    x_vt=np.vstack(tuple(v_vt)).T
    x_test=np.vstack(tuple(v_test)).T
    (n_eb,d)=x_eb.shape
    (n_vt,d)=x_vt.shape
    (n_test,d)=x_test.shape

    # stack them together
    x_all=np.vstack((x_eb,x_vt,x_test))
    mu,sigma=data_to_gaussian_dD(x_all)
    corr=sigma_to_corr(sigma)
    print "Correlation matrix for (Kurto, Rapp, Asdec, Dur, Ene):"
    print corr

    # only the first two are correlated
    # combine only Kurto and RappMaxMean
    ###############
    e_values, e_vectors, x_new = data_to_eigen_projection_dD(x_all[:,0:2])
    print "Eigenvalues for Kurto and RappMaxMean: "
    print e_values
    # put the principle component in a new observable
    data['KRappEB']=x_new[0:n_eb,0]
    data['KRappVT']=x_new[n_eb:n_eb+n_vt,0]
    testdata['KRappVal']=x_new[n_eb+n_vt:,0]

    # no need to combine all features as only the first two are correlated
    ## combine all features
    #e_values, e_vectors, x_new = data_to_eigen_projection_dD(x_all)
    #print "Eigenvalues for all data: "
    #print e_values
    # put the principle component in a new observable
    #for i in xrange(d):
    #    eb_name='u%dEB'%i
    #    vt_name='u%dVT'%i
    #    test_name='u%dVal'%i
    #    data[eb_name]=x_new[0:n_eb,i]
    #    data[vt_name]=x_new[n_eb:n_eb+n_vt,i]
    #    testdata[test_name]=x_new[n_eb+n_vt:,i]
    


def get_EB_VT_priors():
    """
    Returns fixed priors for EB and VT events
    as dictionnary, whose keys are 'EB' and 'VT'
    """

    prior = {}
    prior['EB'] = 0.51
    prior['VT'] = 0.49

    return prior

