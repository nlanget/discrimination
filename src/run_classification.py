import os,sys
from do_classification import classifier


def run_test():

  from options import TestOptions
  opt = TestOptions()

  classifier(opt)
  class_final(opt)
  final_result(opt)


def run_all():

  from options import MultiOptions
  opt = MultiOptions(opt='norm')
  
  #classifier(opt)

  if opt.opdict['method'] == 'lr' or opt.opdict['method'] == 'svm':
    from results import AnalyseResults
    res = AnalyseResults(opt='norm')
    #stats(opt)
    if res.opdict['plot_confusion']:
      res.plot_confusion()

  else:
    from extraction import read_extraction_results
    if opt.opdict['method'] == 'ova':
      filename = '%s/OVA/OVA_%s_%s_svm'%(opt.opdict['outdir'],opt.opdict['feat_filename'].split('.')[0],opt.opdict['stations'][0])
    elif opt.opdict['method'] == '1b1':
      filename = '%s/1B1/1B1_%s_%s_svm-red'%(opt.opdict['outdir'],opt.opdict['feat_filename'].split('.')[0],opt.opdict['stations'][0])
    read_extraction_results(filename)


if __name__ == '__main__':
  #run_test()
  run_all()
