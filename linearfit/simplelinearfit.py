#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 11:57:13 2017

@author: sunshower
"""

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


""" 
Generates the linear fit matrix M defined as M[i,(j,l)] = g_j(x_i,l),
i = 1,...n is the number of data points
(j,l), j=0,...,dim is the subspace dimensionality and
l=1,...,k is the data point dimensionality
where {g_j}_j=0...,dim span the function subspace for the fit and
the k-dimensional data points {x_i}_i=1,...,n have the one-dimensional
label values {y_i}. The default subspace is defined as the span of the
functions g_0 :=1, g_j(x) := exp(j*x), j=1,...,dim

input:
    data: np.array of shape = (n,k) where (x_i,y_i) are the input data to be
    fitted to the target 
    y_i = a_0 + ... + a_(k-1)                                                ^
    + a_k*g_1(x_i,0) + ... + a_(2k-1)*g_1(x_i,(k-1))                         |
    + ...                                                                    | (k-1)*(dim+1) elements in a = number of columns in M
    + a_((k-1)*dim) *g_dim(x_i,0) + ... + a_((k-1)*(dim+1))*g_dim(x_i,(k-1)) v
    where x_i,l is the l-th coordinate of the i-th input data point
    dim: number of basis functions spanning the function subspace

output:
    np.array of shape = (n,(k-1)*(dim+1)) defined by M[i,(j,l)] = g_j(x_i,l)
"""
def generate_Fit_Matrix(data, dim=1, func=(lambda c,x: np.exp(c*x))):
    (n,k) = data.shape
    indices = np.arange(0,dim+1)
    return (np.array([[func(c,x) for c in indices] for x in data[:,:-1]]).reshape(n,(k-1)*(dim+1)))

"""
Solves the linear fit problem described in generate_Fit_Matrix using the
SVD of the matrix generated therein.

input:
    data: np.array of shape = (n,k) where (x_i,y_i) are the input data to be
    fitted to the target 
    y_i = a_0 + ... + a_(k-1)                                                ^
    + a_k*g_1(x_i,0) + ... + a_(2k-1)*g_1(x_i,(k-1))                         |
    + ...                                                                    | (k-1)*(dim+1) elements in a = number of columns in M
    + a_((k-1)*dim) *g_dim(x_i,0) + ... + a_((k-1)*(dim+1))*g_dim(x_i,(k-1)) v
    where x_i,l is the l-th coordinate of the i-th input data point

output:
    np.array of shape = (dim+1,1) being the solution of M a = data[:,1]
"""
def solve_for_coeff(data,dim=1,func=(lambda c,x: np.exp(c*x))):
    M = generate_Fit_Matrix(data,dim,func)
    (n,m) = M.shape
    U,s,V = np.linalg.svd(M)
    S_inv = np.zeros((n,m))
    S_inv[:m,:m] = np.diag(1./s)
    S_inv = S_inv.transpose()
    return (V.transpose().dot(S_inv.dot(U.transpose().dot(data[:,1]))))
    
"""
Test function for a polynomial fit
"""
def test_func(c,x):
    return (x**c)


#################################

# some 1D examples

dim1 = 1
# dim defines how many exponential functions to use to fit your data
# I was lazy here, keep dim smaller than the number of data points you have!
# That makes sense anyway since large dims won't really do a better job
# (quite the contrary actually)

# we start by taking some points that actually follow an exponential function,
# but we add some Gaussian noise onto it
X = np.linspace(0,1,100)
Y = np.exp(X) + np.random.normal(0.1,0.1,100)

# we fill our data vector with those pairs (x_i,y_i)
data = np.vstack((X,Y)).transpose()

# now we basically assume (by default) that our data could be represented
# by y_i = a_0 + a_1*g_1(x_i) + ... + a_dim*g_dim(x_i) and solve_for_coeff
# computes the coefficients {a_j}_j=0,...,dim such that the differences
# |y_i - (a_0 + a_1*g_1(x_i) + ... + a_dim*g_dim(x_i))|² are minimized
a = solve_for_coeff(data,dim1)
M = generate_Fit_Matrix(data,dim1)
b = M.dot(a)

plt.figure(1)

plt.scatter(X,Y)
plt.plot(X,b,'red')

dim1 = 2
a = solve_for_coeff(data,dim1,test_func)
M = generate_Fit_Matrix(data,dim1,test_func)
b = M.dot(a)

plt.plot(X,b,'green')
plt.grid()
plt.show()

#################################

# 1D example loading a .txt file with data
fname = './data.txt'
data = np.loadtxt(fname)

dim1 = 4

a = solve_for_coeff(data,dim1)
np.savetxt('./result.txt',a)
M = generate_Fit_Matrix(data,dim1)
b = M.dot(a)

plt.figure(2)

plt.scatter(data[:,0], data[:,1])
plt.plot(data[:,0],b,'red')
plt.grid()
plt.show()

#################################
# 2D example loading a .txt file with data
fname = './data3D.txt'
# this was created by just generating normal-distributed vectors in 2D space
data = np.loadtxt(fname)
(n,k) = data.shape

a_exp = solve_for_coeff(data,1)
# solve using the exponential kind of fit (higher dimensions than 1 don't
# make that much sense because the values become far too large!)
a_lin = solve_for_coeff(data,1,(lambda c,x: np.ones(x.size) if c==0 else x))
# solve using a linear fit defining a fitting hyperplane; the function passed
# is a row of ones for the first pair of columns and the identity for the next
# which will act on each ROW of the data individually
# --> we solve with respect to the target
# described in the generate_Fit_Matrix docstring
a_poly = solve_for_coeff(data,3,(lambda c,x: x**c))
# use a polynomial function base 

np.savetxt('./result3D.txt',a_exp)

M = generate_Fit_Matrix(data,1,(lambda c,x: np.ones(x.size) if c==0 else x))
N = generate_Fit_Matrix(data,3,(lambda c,x: x**c))
print("\n \n Linear Fit Matrix M = \n "+str(M))
print("\n \n Polynomial Fit Matrix N = \n "+str(N))

fig = plt.figure(3)
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(data[:,0],data[:,1],data[:,2], c='red')

X = np.linspace(-1,1,50)
Y = X
X,Y = np.meshgrid(X,Y)

Z_exp = (a_exp[0] + a_exp[1] + a_exp[2] * np.exp(X) + a_exp[3] * np.exp(Y))
# the example with an exponential kind of fit
Z_lin = (a_lin[0] + a_lin[1] + a_lin[2] * X + a_lin[3] * Y)
# the example with a purely linear fit (hyperplane)
Z_poly = (a_poly[0] + a_poly[1] + a_poly[2] * X + a_poly[3] * Y + a_poly[4] * (X**2) + a_poly[5] * (Y**2) + a_poly[6]*(X**3) + a_poly[7] *(Y**3))
# the example with a polynomial fit - for some reason the results of Z_poly
# and Z_lin are completely different, but the plot does not really
# resolve them as different hyperplanes...hence, we just don't plot it for now

ax.plot_surface(X,Y,Z_exp)
ax.plot_surface(X,Y,Z_lin)
#ax.plot_surface(X,Y,Z_poly)
plt.show()
