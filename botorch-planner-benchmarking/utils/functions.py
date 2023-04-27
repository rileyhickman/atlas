# -*- coding: utf-8 -*-
"""
@author: huongha
@description: Some common synthetic functions for testing Bayesian Optimization
"""


import numpy as np
from collections import OrderedDict


def reshape(x, input_dim):
    '''
    Reshapes x into a matrix with input_dim columns

    '''
    x = np.array(x)
    if x.size == input_dim:
        x = x.reshape((1, input_dim))
    return x


class functions:
    def plot(self):
        print("not implemented")


class hartman_6d:
    '''
    Ackley function
    param sd: standard deviation, to generate noisy evaluations of the function
    '''
    def __init__(self, bounds=None, sd=None):
        self.input_dim = 6

        if bounds is None:
            self.bounds = [(0, 1)]*self.input_dim
        else:
            self.bounds = bounds

        self.min = [(0.)*self.input_dim]
        self.fmin = -3.32237
        self.ismax = -1
        self.name = 'hartman_6d'

    def func(self, X):
        X = reshape(X, self.input_dim)
        n = X.shape[0]

        alpha = [1.0, 1.2, 3.0, 3.2]

        A = [[10, 3, 17, 3.5, 1.7, 8],
             [0.05, 10, 17, 0.1, 8, 14],
             [3, 3.5, 1.7, 10, 17, 8],
             [17, 8, 0.05, 10, 0.1, 14]]
        A = np.asarray(A)
        P = [[1312, 1696, 5569, 124, 8283, 5886],
             [2329, 4135, 8307, 3736, 1004, 9991],
             [2348, 1451, 3522, 2883, 3047, 6650],
             [4047, 8828, 8732, 5743, 1091, 381]]

        P = np.asarray(P)
        c = 10**(-4)
        P = np.multiply(P, c)
        outer = 0

        fval = np.zeros((n, 1))
        for idx in range(n):
            outer = 0
            for ii in range(4):
                inner = 0
                for jj in range(6):
                    xj = X[idx, jj]
                    Aij = A[ii, jj]
                    Pij = P[ii, jj]
                    inner = inner + Aij*(xj-Pij)**2
                new = alpha[ii] * np.exp(-inner)
                outer = outer + new
            fval[idx] = -(2.58 + outer) / 1.94

        if (n == 1):
            return self.ismax*(fval[0][0])
        else:
            return self.ismax*(fval)


# class hartman_6d_repeat:
#     '''
#     Ackley function
#     param sd: standard deviation, to generate noisy evaluations of the function
#     '''
#     def __init__(self, bounds=None, sd=None):
#         self.input_dim = 10

#         if bounds is None:
#             self.bounds = [(-1, 1)]*self.input_dim
#         else:
#             self.bounds = bounds

#         self.min = [(0.)*self.input_dim]
#         self.fmin = -3.32237
#         self.ismax = -1
#         self.name = 'hartman_6d_repeat'

#     def func(self, X):
#         fx = hartmann6_np(X)
        
#         return fx*self.ismax


# class branin_repeat(functions):
#     def __init__(self):
#         self.input_dim = 10
#         self.bounds = [(-1, 1)]*self.input_dim
#         self.fmin = 0.397887
#         self.ismax = -1
#         self.name = 'branin_repeat'

#     def func(self, X):
#         fx = branin_np(X)

#         return fx*self.ismax


class branin(functions):
    def __init__(self):
        self.input_dim = 2
        self.bounds = [(-5, 10), (0, 15)]
        self.fmin = 0.397887
        self.min = [9.424, 2.475]
        self.ismax = -1
        self.name = 'branin'

    def func(self, X):
        X = np.asarray(X)
        if len(X.shape) == 1:
            x1 = X[0]
            x2 = X[1]
        else:
            x1 = X[:, 0]
            x2 = X[:, 1]
        a = 1
        b = 5.1/(4*np.pi*np.pi)
        c = 5/np.pi
        r = 6
        s = 10
        t = 1/(8*np.pi)
        fx = a*(x2-b*x1*x1+c*x1-r)**2+s*(1-t)*np.cos(x1)+s
        return fx*self.ismax


class gSobol:
    '''
    gSolbol function

    param a: one-dimensional array containing the coefficients of the function
    param sd: standard deviation, to generate noisy evaluations of the function
    '''
    def __init__(self, input_dim, bounds=None, sd=None):
        self.input_dim = input_dim

        a = np.zeros((1, self.input_dim))
        for i in range(1, self.input_dim+1):
            a[0, i-1] = (i-2)/2
        self.a = a

        if bounds is None:
            self.bounds = [(0, 1)]*self.input_dim
        else:
            self.bounds = bounds

        self.S_coef = (1/(3*((1+self.a)**2))) / (np.prod(1+1/(3*((1+self.a)**2)))-1)
        if sd is None:
            self.sd = 0
        else:
            self.sd = sd

        self.ismax = -1
        self.fmin = 0
        self.name = 'gSobol'

    def func(self, X):
        X = reshape(X, self.input_dim)
        n = X.shape[0]
        aux = (abs(4*X-2)+np.ones(n).reshape(n, 1)*self.a) / (1+np.ones(n).reshape(n, 1)*self.a)
        fval = np.cumprod(aux, axis=1)[:, self.input_dim-1]

        return self.ismax*fval


class gSobol_new:
    '''
    gSolbol function

    param a: one-dimensional array containing the coefficients of the function
    param sd: standard deviation, to generate noisy evaluations of the function
    '''
    def __init__(self, input_dim, bounds=None, sd=None):
        self.input_dim = input_dim

        a = np.zeros((1, self.input_dim))
        for i in range(1, self.input_dim+1):
            if (i == 1) | (i == 2):
                a[0, i-1] = 0
            else:
                a[0, i-1] = 6.52
        self.a = a

        if bounds is None:
            self.bounds = [(0, 1)]*self.input_dim
        else:
            self.bounds = bounds

        self.S_coef = (1/(3*((1+self.a)**2))) / (np.prod(1+1/(3*((1+self.a)**2)))-1)
        if sd is None:
            self.sd = 0
        else:
            self.sd = sd

        self.ismax = -1
        self.fmin = 0
        self.name = 'gSobol'

    def func(self, X):
        X = reshape(X, self.input_dim)
        n = X.shape[0]
        aux = (abs(4*X-2)+np.ones(n).reshape(n, 1)*self.a) / (1+np.ones(n).reshape(n, 1)*self.a)
        fval = np.cumprod(aux, axis=1)[:, self.input_dim-1]

        return self.ismax*fval


class hartman_4d:
    '''
    Ackley function
    param sd: standard deviation, to generate noisy evaluations of the function
    '''
    def __init__(self, bounds=None, sd=None):
        self.input_dim = 4

        if bounds is None:
            self.bounds = [(0, 1)]*self.input_dim
        else:
            self.bounds = bounds

        self.min = [(0.)*self.input_dim]
        self.fmin = -3.32237
        self.ismax = -1
        self.name = 'hartman_4d'

    def func(self, X):
        X = reshape(X, self.input_dim)
        n = X.shape[0]

        alpha = [1.0, 1.2, 3.0, 3.2]

        A = [[10, 3, 17, 3.5, 1.7, 8],
             [0.05, 10, 17, 0.1, 8, 14],
             [3, 3.5, 1.7, 10, 17, 8],
             [17, 8, 0.05, 10, 0.1, 14]]
        A = np.asarray(A)
        P = [[1312, 1696, 5569, 124, 8283, 5886],
             [2329, 4135, 8307, 3736, 1004, 9991],
             [2348, 1451, 3522, 2883, 3047, 6650],
             [4047, 8828, 8732, 5743, 1091, 381]]

        P = np.asarray(P)
        c = 10**(-4)
        P = np.multiply(P, c)
        outer = 0

        fval = np.zeros((n, 1))
        for idx in range(n):
            X_idx = X[idx, :]
            outer = 0
            for ii in range(4):
                inner = 0
                for jj in range(4):
                    xj = X_idx[jj]
                    Aij = A[ii, jj]
                    Pij = P[ii, jj]
                inner = inner + Aij*(xj-Pij)**2
                new = alpha[ii] * np.exp(-inner)
                outer = outer + new
            fval[idx] = (1.1 - outer) / 0.839
        if n == 1:
            return self.ismax*(fval[0][0])
        else:
            return self.ismax*(fval)


class hartman_3d:
    '''
    hartman_3d: function
    param sd: standard deviation, to generate noisy evaluations of the function
    '''
    def __init__(self, bounds=None, sd=None):
        self.input_dim = 3

        if bounds is None:
            self.bounds = [(0, 1)]*self.input_dim
        else:
            self.bounds = bounds

        self.min = [(0.)*self.input_dim]
        self.fmin = -3.86278
        self.ismax = -1
        self.name = 'hartman_3d'

    def func(self, X):
        X = reshape(X, self.input_dim)
        n = X.shape[0]

        alpha = [1.0, 1.2, 3.0, 3.2]
        A = [[3.0, 10, 30],
             [0.1, 10, 35],
             [3.0, 10, 30],
             [0.1, 10, 35]]
        A = np.asarray(A)
        P = [[3689, 1170, 2673],
             [4699, 4387, 7470],
             [1091, 8732, 5547],
             [381, 5743, 8828]]

        P = np.asarray(P)
        c = 10**(-4)
        P = np.multiply(P, c)
        outer = 0

        fval = np.zeros((n, 1))
        for idx in range(n):
            outer = 0
            for ii in range(4):
                inner = 0
                for jj in range(3):
                    xj = X[idx, jj]
                    Aij = A[ii, jj]
                    Pij = P[ii, jj]
                    inner = inner + Aij*(xj-Pij)**2
                new = alpha[ii] * np.exp(-inner)
                outer = outer + new
            fval[idx] = -outer

        if n == 1:
            return self.ismax*(fval[0][0])
        else:
            return self.ismax*(fval)
    

class ackley:
    '''
    Ackley function 

    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self, input_dim, bounds=None,sd=None):
        self.input_dim = input_dim

        if bounds == None: 
            self.bounds =[(-32.768,32.768)]*self.input_dim
        else: 
            self.bounds = bounds

        self.min = [(0.)*self.input_dim]
        self.fmin = 0
        self.ismax=-1
        self.name='ackley'
        
    def func(self,X):
        X = reshape(X,self.input_dim)
        fval = (20+np.exp(1)-20*np.exp(-0.2*np.sqrt((X**2).sum(1)/self.input_dim))-np.exp(np.cos(2*np.pi*X).sum(1)/self.input_dim))
        
      
        return self.ismax*fval


class beale(functions):
    '''
    beale function

    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,bounds=None, sd=None):
        self.input_dim = 2
        if bounds is None:
            self.bounds = [(-4.5, 4.5), (-4.5, 4.5)]
        else:
            self.bounds = bounds
        self.min = [(3, 0.5)]
        self.fmin = 0
        self.ismax = -1
        self.name = 'Beale'

    def func(self, X):
        X = np.asarray(X)
        if len(X.shape) == 1:
            x1 = X[0]
            x2 = X[1]
        else:
            x1 = X[:, 0]
            x2 = X[:, 1]
        fval = (1.5-x1+x1*x2)**2+(2.25-x1+x1*x2**2)**2+(2.625-x1+x1*x2**3)**2
        return self.ismax*fval


class egg_holder(functions):
    '''
    Egg holder function
    '''
    def __init__(self, bounds=None, sd=None):
        self.input_dim = 2
        if bounds is None:
            self.bounds = [(-512, 512), (-512, 512)]
        else:
            self.bounds = bounds
        self.min = [(512, 404.2319)]
        self.fmin = 0
        self.ismax = -1
        self.name = 'Eggholder'

    def func(self, X):
        X = np.asarray(X)
        if len(X.shape) == 1:
            x1 = X[0]
            x2 = X[1]
        else:
            x1 = X[:, 0]
            x2 = X[:, 1]
        fval = -(x2+47)*np.sin(np.sqrt(np.abs(x2+x1/2+47))) - x1*np.sin(np.sqrt(np.abs(x1-x2-47)))
        return self.ismax*fval


class Shekel(functions):
    '''
    Egg holder function
    '''
    def __init__(self, bounds=None, sd=None):
        self.input_dim = 4
        if bounds is None:
            self.bounds = [(0, 10)]*self.input_dim
        else:
            self.bounds = bounds
        self.min = -10.5364
        self.fmin = 0
        self.ismax = -1
        self.name = 'Shekel'

    def func(self, X):
        m = 10
        beta = 1/m*np.array([1, 2, 2, 4, 4, 6, 3, 7, 5, 5])
        C = np.array([[4, 1, 8, 6, 3, 2, 5, 8, 6, 7],
                      [4, 1, 8, 6, 7, 9, 3, 1, 2, 3.6],
                      [4, 1, 8, 6, 3, 2, 5, 8, 6, 7],
                      [4, 1, 8, 6, 7, 9, 3, 1, 2, 3.6]])

        X = np.asarray(X)
        if len(X.shape) == 1:
            x1 = X[0]
            x2 = X[1]
            x3 = X[2]
            x4 = X[3]
        else:
            x1 = X[:, 0]
            x2 = X[:, 1]
            x3 = X[:, 2]
            x4 = X[:, 3]

        fval = 0
        for i in range(m):
            fval += -np.divide(1, beta[i] + (x1-C[0, i])**2 + (x2-C[1, i])**2 + (x3-C[2, i])**2 + (x4-C[3, i])**2)

        return self.ismax*fval


class Levy(functions):
    '''
    Egg holder function
    '''
    def __init__(self, input_dim, bounds=None, sd=None):
        self.input_dim = input_dim
        if bounds is None:
            self.bounds = [(-10, 10)]*self.input_dim
        else:
            self.bounds = bounds
        self.min = [(1.)*self.input_dim]
        self.fmin = 0
        self.ismax = -1
        self.name = 'Levy'

    def func(self, X):
        X = reshape(X, self.input_dim)

        w = np.zeros((X.shape[0], self.input_dim))
        for i in range(1, self.input_dim+1):
            w[:, i-1] = 1 + 1/4*(X[:, i-1]-1)

        fval = (np.sin(np.pi*w[:, 0]))**2 + ((w[:, self.input_dim-1]-1)**2)*(1+(np.sin(2*np.pi*w[:, self.input_dim-1]))**2)
        for i in range(1, self.input_dim):
            fval += ((w[:, i]-1)**2)*(1+10*(np.sin(np.pi*w[:, i]))**2) 

        return self.ismax*fval


class rosenbrock(functions):
    '''
    rosenbrock function
    param sd: standard deviation, to generate noisy evaluations of the function
    '''
    def __init__(self, input_dim, bounds=None):
        self.input_dim = input_dim
        if bounds is None:
            self.bounds = [(-2.048, 2.048)]*self.input_dim
        else:
            self.bounds = bounds

        self.min = [(0.)*self.input_dim]
        self.fmin = 0
        self.ismax = -1
        self.name = 'Rosenbrock'
    
    def func(self, X):
        X = reshape(X, self.input_dim)
        fval = 0
        for i in range(self.input_dim-1):
            fval += (100*(X[:, i+1]-X[:, i]**2)**2 + (X[:, i]-1)**2)
        
        return self.ismax*fval


class alpine1:
    '''
    Alpine1 function

    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''

    def __init__(self, input_dim, bounds=None, sd=None):
        if bounds is None:
            self.bounds = [(0, 10)]*input_dim
        else:
            self.bounds = bounds
        self.min = [(0)]*input_dim
        self.fmin = 0
        self.input_dim = input_dim
        if sd is None:
            self.sd = 0
        else:
            self.sd = sd

        self.ismax = -1
        self.name = 'alpine1'

    def func(self, X):
        X = reshape(X, self.input_dim)
        temp = abs(X*np.sin(X) + 0.1*X)
        if len(temp.shape) <= 1:
            fval = np.sum(temp)
        else:
            fval = np.sum(temp, axis=1)

        return self.ismax*fval


class sixhumpcamel(functions):
    '''
    Six hump camel function

    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self, bounds=None, sd=None):
        self.input_dim = 2
        if bounds is None:
            self.bounds = [(-2, 2)]*self.input_dim
        else:
            self.bounds = bounds
        self.min = [(0.0898, -0.7126), (-0.0898, 0.7126)]
        self.fmin = -1.0316
        self.ismax = -1

        self.name = 'Six-hump camel'

    def func(self, X):
        X = np.asarray(X)
        if len(X.shape) == 1:
            x1 = X[0]
            x2 = X[1]
        else:
            x1 = X[:, 0]
            x2 = X[:, 1]
        term1 = (4-2.1*x1**2+(x1**4)/3) * x1**2
        term2 = x1*x2
        term3 = (-4+4*x2**2) * x2**2
        fval = term1 + term2 + term3
        return self.ismax*fval
