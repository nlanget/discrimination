#!/usr/bin/env python
# encoding: utf-8

import os,sys

def run_test():

  from options import TestOptions
  opt = TestOptions()

  from do_classification import classifier
  classifier(opt)


def run_all():

  from options import MultiOptions
  opt = MultiOptions()
  #opt.count_number_of_events()

  ### UNSUPERVISED METHOD ### 
  if opt.opdict['method'] == 'kmeans':
    from unsupervised import classifier
    classifier(opt)

  ### SUPERVISED METHODS ###
  elif opt.opdict['method'] in ['lr','svm','svm_nl','lrsk']:
    from do_classification import classifier
    classifier(opt)

    from results import AnalyseResults
    res = AnalyseResults()
    if res.opdict['plot_confusion']:
      res.plot_confusion()

  elif opt.opdict['method'] in ['ova','1b1']:
    from do_classification import classifier
    classifier(opt)

    from results import AnalyseResultsExtraction
    res = AnalyseResultsExtraction()


if __name__ == '__main__':
  #run_test()
  run_all()

