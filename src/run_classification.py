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
  opt = MultiOptions(opt='norm')
 
  from do_classification import classifier
  #classifier(opt)

  if opt.opdict['method'] == 'lr' or opt.opdict['method'] == 'svm' or opt.opdict['method'] == 'lrsk':
    from results import AnalyseResults
    res = AnalyseResults(opt='norm')
    if res.opdict['plot_confusion']:
      res.plot_confusion()

  else:
    from results import AnalyseResultsExtraction
    res = AnalyseResultsExtraction(opt='norm')


def run_unsupervised():

  from options import MultiOptions
  opt = MultiOptions(opt='norm')
  opt.opdict['method'] = 'kmean'

  from unsupervised import classifier
  classifier(opt)


if __name__ == '__main__':
  #run_test()
  #run_all()
  run_unsupervised()

