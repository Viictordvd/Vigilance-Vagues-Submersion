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
  # x,y: arrays of position, shape = (n_x,d) and (n_y,d)
  # param: parameters of the 1D kernel
  # kernel: type of the 1D kernel
  # type: either "sum" or "product"

  n_x, d = x.shape
  n_y = y.shape[0]

  #Initialisation
  if type == "sum":
        K = np.zeros((n_x, n_y))
  elif type == "product":
        K = np.ones((n_x, n_y))
  else:
        raise ValueError("type must be either 'sum' or 'product'")

  for i in range(d):
    Ki = kernel(x[:, i], y[:, i], param)
    if type =="sum":
        K += Ki
    else:
        K *= Ki
  return K

def condMean(x,X,Y,kern,param,multikern=None,type=None):
  #multikern : precise kernel type if kern is a multidimensional kernel
  if multikern == None :
    k_xX = kern(x, X,param)
    k_XX = kern(X, X,param)
  else :
    k_xX = kern(x, X,param,multikern,type)
    k_XX = kern(X, X,param,multikern,type)
  return k_xX @ np.linalg.inv(k_XX) @ Y

def condVar(x,X,Y,kern,param,multikern=None,type=None):
  if multikern == None :
    k_xx = kern(x, x, param)
    k_xX = kern(x, X,param)
    k_Xx = np.transpose(k_xX)
    k_XX = kern(X, X,param)
  else :
    k_xx = kern(x, x, param,multikern,type)
    k_xX = kern(x, X,param,multikern,type)
    k_Xx = np.transpose(k_xX)
    k_XX = kern(X, X,param,multikern,type)
  return k_xx - k_xX @ np.linalg.inv(k_XX) @ k_Xx