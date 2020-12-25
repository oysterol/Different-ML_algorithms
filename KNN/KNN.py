import pandas as pd
import numpy as np

def KNN(K,K_inn,df, w):
    v=len(df.iloc[0])-1
    N=[[0]*2]*K
    for i in range(len(df)):
        dist=0
        for j in range(v):
            dist+=abs(K_inn[j]-df.iloc[i][j]*w[j])
        if i>=K:
                temp=max(N,key=lambda x:x[1])
                pos=N.index(temp)
        else:
            pos=i
            N[pos]=[i,dist]
            continue
        if dist<temp[i]:
            N[pos]=[i,dist]
    return N

def class_determiner(df, N):
    unique = []
    y_values=[]
    out=0
    for i in range(len(N)):
        y=df.iloc[N[i][0]][-1]
        y_values.append(y)
        if y not in unique:
            unique.append(y)
            
    for i in range(len(unique)):
        temp_out=y_values.count(unique[i])
        if temp_out>out:
            out=temp_out
            res=unique[i]
    return res

def determiner_avg(df,N):
    y=0
    for i in range(len(N)):
        y+=df.iloc[N[i][0]][-1]
    return y/len(N)

def show_NN_details(df, N):
    for i in range(len(N)):
        print('Sample number: ',N[i][0],' Value: ',df.iloc[N[i][0]][-1])
        