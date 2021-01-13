import numpy as np

class Adaline(object):
    def __init__(self, eta=0.01, n_iter=50, rand_state=1):
        self.eta=eta
        self.n_iter=n_iter
        self.rand_state=rand_state

    def fit(self, X,y):
        rgen=np.random.RandomState(self.rand_state)
        self.w_=rgen.normal(loc=0.0,scale=0.01, size=1+X.shape[1])
        self.cost=[]

        for _ in range(self.n_iter):
            output=self.net_input(X)
            errors=(y-output)
            self.w_[1:]+=self.eta*X.T.dot(errors)
            self.w_[0]+=self.eta*errors.sum()
            cost=(errors**2).sum()/2.0

        return self
    def net_input(self,X):
        return np.dot(X,self.w_[1:]+self.w_[0])


    def predict(self, X):
        return np.where(self.net_input(X)>=0.0,1,-1)



        