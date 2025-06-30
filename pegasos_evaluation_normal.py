import numpy as np
import GraphKernelFunc as gkf
from sklearn.model_selection import train_test_split

class Pegasos():
    def __init__(self, graphs, classes, iter, lamda):
        # self.graphs = graphs
        # self.classes = classes
        self.iter = iter
        self.lamda = lamda
        self.G_train, self.G_test, self.y_train, self.y_test = train_test_split(graphs, classes, test_size=0.1)

    def train(self):
        # learning

        # initialize
        alpha = np.zeros(len(self.G_train))

        # iterate
        for t in range(1,self.iter+1):
            i_t = np.random.randint(len(self.G_train))
            sigma_loss = 0

            for j in range(len(self.G_train)):
                sigma_loss += alpha[j] * self.y_train[j] * gkf.GraphkernelFunc.k_func_wl(self.G_train[i_t], self.G_train[j], 2)
            
            if(self.y_train[i_t] / (self.lamda * t) * sigma_loss < 1):
                alpha[i_t] += 1
        print("iteration finished")        

        return alpha
    
    def predict(self, alpha):
        # predict
        predict = []
        truevalue = []
        for i in range(len(self.G_test)):
            y = 0
            for j in range(len(self.G_train)):
                y +=  alpha[j] * self.y_train[j] * gkf.GraphkernelFunc.k_func_wl(self.G_test[i], self.G_train[j], 2)
            predict.append(np.sign(y)[0])
            truevalue.append(self.y_test[i])

        return np.array(predict), np.array(truevalue)

    def accuracy(self, alpha):
        # accuracy
        predict, truevalue = self.predict(alpha)
        return (predict == truevalue).mean()
            
   





