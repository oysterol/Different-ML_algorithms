import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np


def OLS(x,y):
    """Takes the ordinary least squares of the two input matrixes,
    essentialy this is the regression line """
 
    w=np.linalg.pinv(x.T.dot(x)).dot(x.T.dot(y))
    return w

def Error_mse(x,y,w):
    """Returns mean squaree error of the Ordinary least squares function """
    e=w.T.dot(x.dot(x.T)).dot(w)-2*(x.T.dot(w)).T.dot(y.T)+y.dot(y.T)
    return e[0][0]


def zigma(z):
    return 1/(1+np.exp(-z))


def update_rule(w0,l,it,x,y):
    """Training module maximum logarithmic likelihood, l is the learning rate, it is nr of iterations """
    w=w0
    for i in range(it):
        Xt=0
        w0=w
        for j in range(0,len(x[0])):
            Xt+=(zigma(w0.T.dot(x[:,j]))-y[j])*x[:,j]
            for i in range(len(Xt)):
                w[i]=w0[i]-l*Xt[i]
    return w

def Entropy_error(z,y):
    """Entropy error is the error in correlation with the update rule """
    res=0
    for i in range(len(y)):
        res+=y[i]*np.log(zigma(z))+(1-y[i])*np.log(1-zigma(z))
        return (-res/len(x[0]))[0]

def print_Entropy_error(w0,l,it,x,y):
    w=w0
    for i in range(it):
        Xt=0
        w0=w
        for j in range(0,len(x[0])):
            Xt+=(zigma(w0.T.dot(x[:,j]))-y[j])*x[:,j]
        for i in range(len(Xt)):
            w[i]=w0[i]-l*Xt[i]
        z=w.T.dot(x[:,i])
        plt.plot(z,Entropy_error(z,y),'r.')

