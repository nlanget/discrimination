#!/usr/bin/env python
# encoding: utf-8

import os, sys, optparse, glob
import numpy as np
import logging
from obspy.core import read

def make_SDS_data_links(datadir,outdir):

  data_dir=os.path.abspath(datadir)
  out_dir=os.path.abspath(outdir)
  logging.debug('Data from directory %s'%data_dir)
  logging.debug('SDS data to be put in directory %s'%out_dir)

  all_sta = glob.glob(os.path.join(data_dir,'*'))
  all_sta.sort()

  filedict = {}
  for sta in all_sta:
    all_files = glob.glob(os.path.join(sta,'grad','*.*'))
    all_files.sort()
    for filename in all_files:
      st=read(filename)
      net=st.traces[0].stats.network
      sta=st.traces[0].stats.station
      cha=st.traces[0].stats.channel
      dirid="%s.%s.%s"%(net,sta,cha)
      if filedict.has_key(dirid):
        filedict[dirid].append(filename)
      else:
        filedict[dirid]=[filename]

  for dirid,filelist in filedict.iteritems():
    net=dirid.split('.')[0]
    sta=dirid.split('.')[1]
    cha=dirid.split('.')[2]
    dirname=os.path.join(out_dir,net,sta,"%s.D"%cha)
    try:
      os.makedirs(dirname)
      logging.info("Made directories : %s"%dirname)
    except OSError:
      logging.debug("Directories already exist : %s"%dirname)
      pass

    for my_file in filelist:
      dest_file=os.path.join(dirname,os.path.basename(my_file))
      try:
        os.symlink(my_file,dest_file)
        logging.info("Linked %s"%dest_file)
      except OSError:
        logging.debug("Removing old %s"%dest_file)
        os.remove(dest_file)
        os.symlink(my_file,dest_file)
        logging.info("Linked %s"%dest_file)
    


if __name__ == '__main__':

  make_SDS_data_links('/media/disk/CLASSIFICATION','../data/Ijen/GRAD')

