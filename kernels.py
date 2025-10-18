import numpy as np

def cosKern(x,y,param):
  theta = param[0]
  sigma = param[1]
  x = np.atleast_2d(x).T
  y = np.atleast_2d(y)
  dist = (x-y)/theta
  return sigma**2*np.cos(dist)

def expKern(x,y,param):
  theta = param[0]
  sigma = param[1]
  x = np.atleast_2d(x).T
  y = np.atleast_2d(y)
  dist = (x-y)/theta
  return sigma**2*np.exp(-abs(dist)**2)

def mat5_2Kern(x,y,param):
  theta = param[0]
  sigma = param[1]
  x = np.atleast_2d(x).T
  y = np.atleast_2d(y)
  dist = abs(x-y) * np.sqrt(5) / theta
  return sigma**2*(1+dist+dist**2/3)*np.exp(-dist)

def mat3_2Kern(x,y,param):
  theta = param[0]
  sigma = param[1]
  x = np.atleast_2d(x).T
  y = np.atleast_2d(y)
  dist = abs(x-y) * np.sqrt(3) / theta
  return sigma**2*(1+dist)*np.exp(-dist)

def sqr_expKern(x,y,param):
  theta = param[0]
  sigma = param[1]
  x = np.atleast_2d(x).T
  y = np.atleast_2d(y)
  dist = abs(x-y) / theta
  return sigma**2*np.exp(-dist**2/2)

def RdKernel(x,y,param,kernel,type):
  #input:
  # x,y: arrays of position, dimension =d
  # param: parameters of the 1D kernel
  # kernel: type of the 1D kernel
  # type: either "sum" or "product"

  d = x.shape[0]
  n = x.shape[1]
  kern = np.zeros((n,n))
  for i in range(d):
    if type =="sum":
        kern += kernel(x[i,:],y[i,:],param)
    elif type == "product":
        kern *= kernel(x[i,:],y[i,:],param)
    else:
        print("Type must be either sum or product\n")
        break
  return kern

def condMean(x,X,Y,kern,param):
  k_xX = kern(x, X,param)
  k_XX = kern(X, X,param)
  return k_xX @ np.linalg.inv(k_XX) @ Y

def condVar(x,X,Y,kern,param):
  k_xx = kern(x, x, param)
  k_xX = kern(x, X,param)
  k_Xx = np.transpose(k_xX)
  k_XX = kern(X, X,param)
  return k_xx - k_xX @ np.linalg.inv(k_XX) @ k_Xx