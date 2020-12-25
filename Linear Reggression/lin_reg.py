import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

def Get_data_reg(data):
    """This function is directed towards two specific datasets,
    returns variables matrix X and target matrix Y """
    datas=pd.read_csv(data, sep=',')
    df=pd.DataFrame(datas)
    I=[]
    for i in range(len(df.x1)):
        I.append(1)
    if 'x2' in df:
        X=np.array([I,df.x1,df.x2])
    else:
        X=np.array([I,df.x1])
    Y=np.array([df.y])
    return X,Y

def OLS(x,y):
    """Takes the ordinary least squares of the two input matrixes,
    essentialy this is the regression line """
 
    w=np.linalg.pinv(x.T.dot(x)).dot(x.T.dot(y))
    return w

def Error_mse(x,y,w):
    """Returns mean squaree error of the Ordinary least squares function """
    e=w.T.dot(x.dot(x.T)).dot(w)-2*(x.T.dot(w)).T.dot(y.T)+y.dot(y.T)
    return e[0][0]

def polt_1D(X,Y,w):
    plt.plot(X[1],w[0]+w[1]*X[1],'-r')
    plt.plot(X[1],Y[0],'.g')


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


def plot_cl(x,y):
    for i in range(len(y)):
        if y[i]==1:
            plt.plot(x[1,i],x[2,i],'ro')
        else:
            plt.plot(x[1,i],x[2,i],'bo')


def plot_DB_1D(w):
    x1=np.array([0,1/w[2]])
    x2=np.array([0,1/w[2]])
    plt.plot(-x1*w[1]+w[0],x2*w[2])

def plot_DB_2D(w):
    x1=np.linspace(0,1,100)
    x2=np.linspace(0,1,100)
    x1,x2=np.meshgrid(x1,x2)
    Y=w[0]
    X1=w[3]*x1**2+w[1]*x1
    X2=w[4]*x2**2+w[2]*x2
    plt.contour(x1,x2,(X1+X2-Y),1)

def Get_data_cl(data):

    datas=pd.read_csv(data, sep=',')
    df=pd.DataFrame(datas)
    I=[]
    for i in range(len(df)):
        I.append(1)
    df.columns=['x1','x2','y']
    X=np.array([I,df.x1,df.x2])
    y=np.array(df.y)
    return X,y

def Get_data_cl_2D(data):

    datas=pd.read_csv(data, sep=',')
    df=pd.DataFrame(datas)
    I=[]
    for i in range(len(df)):
        I.append(1)
    df.columns=['x1','x2','y']
    X=np.array([I,df.x1,df.x2,df.x1*df.x1,df.x2*df.x2])
    y=np.array(df.y)
    return X,y

def plot_1D(X,Y,w):
    plt.plot(X[1],w[0]+w[1]*X[1],'-r')
    plt.plot(X[1],Y[0],'.g')


if "__name__"=="__main__":
    ##Get training data
    data=Get_data_reg('train_2d_reg_data.csv')
    ##train model
    print(data[0].shape,data[1].shape)
    w=OLS(data[0].T,data[1].T)
    print('Weights : ',w)
    ##Mean square error training data
    print('Error Training: ',Error_mse(data[0],data[1],w))
    ###Get_test data
    data=Get_data_reg('test_2d_reg_data.csv')
    ###Mean Square error test data
    print('Error Test: ',Error_mse(data[0],data[1],w))
    ##Get training data
    data=Get_data_reg('train_1d_reg_data.csv')
    ##train model
    w=OLS(data[0].T,data[1].T)
    ###Get_test data
    data=Get_data_reg('test_1d_reg_data.csv')
    ##plot test data against weights
    plot_1D(data[0],data[1],w)
    data=Get_data_cl('cl_train_1.csv')
    w0=np.array([[0.7],[0.65], [0.2]])
    l=0.0001
    it=1000
    X1=data[0]
    y1=data[1]
    w=update_rule(w0,l,it,X1,y1)
    print(w)
    plt.figure(0)
    plot_cl(X1,y1)
    plt.savefig('sepfig')
    plt.figure(1)
    plot_DB_1D(w)
    data=Get_data_cl('cl_test_1.csv')
    X2=data[0]
    y2=data[1]
    plot_cl(X2,y2)
    plt.savefig('DB_1D')
    plt.figure(2)
    print_Entropy_error(w0,l,it,X1,y1)
    plt.savefig('Entropyerror_train')
    plt.figure(3)
    print_Entropy_error(w0,l,it,X2,y2)
    plt.savefig('Entropyerror_test')


    data=Get_data_cl_2D('cl_train_2.csv')
    w0=np.array([[0.5],[-1], [-1],[1.4], [1.4]])
    l=0.0001
    it=1000
    X1=data[0]
    y1=data[1]
    #print(len(y))
    plt.figure(0)
    plot_cl(X1,y1)
    w=update_rule(w0,l,it,X1,y1)
    plt.savefig('non_sep')
    #Entropy_error(X,y,w)
    print(w)
    data=Get_data_cl('cl_test_2.csv')
    X2=data[0]
    y2=data[1]
    plt.figure(1)
    plot_cl(X2,y2)
    plot_DB_2D(w)
    #plt.savefig('DB_2D')
    plt.show()