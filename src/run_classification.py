import os,sys
from do_classification import classifier,class_final,final_result


def run_test():

  from options import TestOptions
  opt = TestOptions()

  classifier(opt)


def run_all():

  from options import MultiOptions
  opt = MultiOptions(opt='norm')
  
  classifier(opt)

  if opt.opdict['method'] == 'lr' or opt.opdict['method'] == 'svm':
    class_final(opt)
    final_result(opt)

  else:
    from extraction import read_extraction_results
    if opt.opdict['method'] == 'ova':
      filename = '%s/OVA_%s'%(opt.opdict['outdir'],opt.opdict['feat_filename'].split('.')[0])
    elif opt.opdict['method'] == '1b1':
      filename = '%s/1B1_%s'%(opt.opdict['outdir'],opt.opdict['feat_filename'].split('.')[0])
    read_extraction_results(filename)


if __name__ == '__main__':
  #run_test()
  run_all()
