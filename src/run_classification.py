import os,sys
from do_classification import classifier,class_final,final_result


def run_test():

  from options import TestOptions
  opt = TestOptions()

  classifier(opt)
  class_final(opt)
  final_result(opt)


def run_all():

  from options import MultiOptions
  opt = MultiOptions(opt='norm')
  
  classifier(opt)

  if opt.opdict['method'] == 'lr' or opt.opdict['method'] == 'svm':
    from do_classification import class_final,final_result,stats
    #stats(opt)
    class_final(opt)
    final_result(opt)
    if opt.opdict['boot'] > 1:
      from do_classification import plot_test_vs_train
      plot_test_vs_train(opt)

  else:
    from extraction import read_extraction_results
    if opt.opdict['method'] == 'ova':
      filename = '%s/OVA/OVA_%s_%s_svm'%(opt.opdict['outdir'],opt.opdict['feat_filename'].split('.')[0],opt.opdict['stations'][0])
    elif opt.opdict['method'] == '1b1':
      filename = '%s/1B1/1B1_%s_%s_lr'%(opt.opdict['outdir'],opt.opdict['feat_filename'].split('.')[0],opt.opdict['stations'][0])
    read_extraction_results(filename)


if __name__ == '__main__':
  #run_test()
  run_all()
