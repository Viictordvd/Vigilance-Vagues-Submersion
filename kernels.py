import numpy as np

def cosKern(x,y,theta,sigma):
  x = np.atleast_2d(x).T
  y = np.atleast_2d(y)
  dist = (x-y)/theta
  return sigma**2*np.cos(dist)

def expKern(x,y,theta,sigma):
  x = np.atleast_2d(x).T
  y = np.atleast_2d(y)
  dist = (x-y)/theta
  return sigma**2*np.exp(-abs(dist)**2)

def mat5_2Kern(x,y,theta,sigma):
  x = np.atleast_2d(x).T
  y = np.atleast_2d(y)
  dist = abs(x-y) * np.sqrt(5) / theta
  return sigma**2*(1+dist+dist**2/3)*np.exp(-dist)

def mat3_2Kern(x,y,theta,sigma):
  x = np.atleast_2d(x).T
  y = np.atleast_2d(y)
  dist = abs(x-y) * np.sqrt(3) / theta
  return sigma**2*(1+dist)*np.exp(-dist)

def sqr_expKern(x,y,theta,sigma):
  x = np.atleast_2d(x).T
  y = np.atleast_2d(y)
  dist = abs(x-y) / theta
  return sigma**2*np.exp(-dist**2/2)

def condMean(x,X,Y,kern,sigma,theta):
  k_xX=kern(x, X, sigma,theta)
  k_XX=kern(X, X, sigma,theta)
  return k_xX @ np.linalg.inv(k_XX) @ Y

def condVar(x,X,Y,kern,sigma,theta):
  k_xx=kern(x, x, sigma,theta)
  k_xX=kern(x, X, sigma,theta)
  k_Xx=np.transpose(k_xX)
  k_XX=kern(X, X, sigma,theta)
  return k_xx - k_xX @ np.linalg.inv(k_XX) @ k_Xx