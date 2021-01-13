import pandas as pd
import numpy as np

class KNN (object):
    def __init__(self,X,y,K=10):
        self.K=K
        self.X=X
        self.y=y
        self.neighbours=[[0]*2]*K
        #self.max_dist=np.inf()


    def find_NN(self,Target):
        
        for case in range(len(self.X[0])):
            dist=0
            for feature in range(len(self.X)-1):
                dist+=self.X[case][feature]-Target[feature]
            if case>self.K:
                temp=max(neighbours,key=lambda x:x[1])
                pos=N.index(temp)
            else:
                pos=i
                neighbours[pos]=[i,dist]
            if dist<temp[1]:
                neighbours[pos]=[i,dist]
    
        

    def determine_average(self):
        val=0
        for i in range(len(neighbours[0])):
            val+=y[neighbours[i][0]]
        return val/K

    def class_determiner(self):
        N_classes=np.arrary(self.K*[0])
        for i in range(len(neighbours[0])):
            N_classes[i]=y[neighbours[i][0]]

        unique=np.unique(N_classes)
        nr_uniques=unique.values()
        pred=nr_uniques.index(max(nr_uniques))
        return unique[pred]

        