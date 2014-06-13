import numpy as np
import sys
from math import factorial as fact
# --------------------------------------------------------------
def counter(n,deg):
  level={}
  level[0]=np.ones(deg+n-1,dtype=int)
  for j in range(1,n):
    level[j]=[]
    for k in range(len(level[j-1])):
      level[j].append(np.sum(level[j-1][:k],dtype=int))
    level[j]=np.array(level[j])
    level[j]=np.delete(level[j],np.where(level[j]==0))

  if n != 1:
    nb_terms=0
    for j in range(len(level[n-2])):
      nb_terms=nb_terms+np.sum(level[n-2][:len(level[n-2])-j])
    nb_terms=nb_terms-1
  else:
    nb_terms=deg

  for key in sorted(level):
    level[key]=level[key][:deg]

  return level,nb_terms
# --------------------------------------------------------------
def do_list_add_ind(level,n,deg):
  list=[]
  iter,j,ii,l=0,0,0,0
  k=level[n-1][2]
  nb_x=level[n-1][-3]
  iter_max=level[n-1][deg-2]
  while iter < iter_max:
    for i in np.arange(n-1-j,-1,-1):
      list.append(level[i][2])
      iter+=1
    if iter == iter_max:
      break
    if iter == k and iter != nb_x:
      ii+=1
      if ii > n-1:
        ii=1
      j=ii
      k=k+level[n-1-j][2]
    elif iter == nb_x:
      l+=1
      nb_x=nb_x+level[n-1-l][-3]
      k=k+level[n-1-l][2]
      j=l
      ii=1
    elif j > n-1:
      j=n-1
    else:
      j+=1
  return list
# --------------------------------------------------------------
def do_list_iter(list_ind,n):
  list=np.array(list_ind)
  list_iter=np.empty(len(list_ind),dtype=int)
  for i in range(n):
    list_iter[np.where(list_ind==list_ind[i])]=i
  return list_iter
# --------------------------------------------------------------
def poly(deg,x):
  """
  Computes the polynome of degree deg
  """
  # Initialisation to degree = 1
  x_deg = x.copy()
  x_deg.columns = range(1,x.shape[1]+1)
  ind=x.shape[1]

  list_feat = x.columns
  degrees=range(2,deg+1)
  n = x.shape[1] # number of features

  k1=1
  k2=ind
  k=0
  for ideg,deg in enumerate(degrees):
    ii=0
    k3=k2
    list_keys = x_deg.columns
    level,nb_terms=counter(n,deg)
    if deg == 2:
      list_index=[nb_terms-n]
      list_iter=[0]
    elif deg == 3 or deg == 4:
      list_index=do_list_add_ind(level,n,deg)
      list_iter=do_list_iter(list_index,n)
      if deg == 3:
        basis=list_index
        pattern=[]
      else:
        pattern=list_index[len(basis):]
    else:
      basis=list_index
      if n == 3:
        p=pattern[n:-1]
      else:
        p=[]
        for i in range(n-3):
          p=np.concatenate([p,pattern[-(n+i+k):]])
        k+=1
      list_index=np.concatenate([basis,pattern,p,[1]])
      list_iter=do_list_iter(list_index,n)
      pattern=list_index[len(basis):]
    iter=list_iter[0]
    for nfeat,feat in enumerate(list_keys):
      if nfeat+1 >= k1:
        if ind == k3+list_index[ii]:
          if ind != nb_terms-1:
            iter=list_iter[ii+1]
            ii+=1
            k3=ind
          else:
            iter=len(list_feat)-1
        for f in list_feat[iter:]:
          ind=ind+1
          x_deg[ind] = x_deg[feat]*x[f]
        iter=iter+1
    k1=k2+1
    k2=ind
    if x_deg.shape[1] != fact(deg+n)/(fact(n)*fact(deg))-1:
      print "Warning !! Wrong number of features"
      print "degree: %d, # features: %d, expected # features: %d"%(deg, x_deg.shape[1], fact(deg+n)/(fact(n)*fact(deg))-1)
  return x_deg
