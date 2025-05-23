import numpy as np
import GraphKernelFunc as gkf

class Pegasos():
    def __init__(self, graphs, classes, iter, lamda):
        self.graphs = graphs
        self.classes = classes
        self.iter = iter
        self.lamda = lamda

    def train(self):
        # learning

        # initialize
        alpha = np.zeros(len(self.graphs))

        # iterate
        for t in range(1,self.iter+1):
            print("t = ",t)
            i_t = np.random.randint(len(self.graphs))
            sigma_loss = 0

            for j in range(len(self.graphs)):
                sigma_loss += alpha[j] * self.classes[i_t] * gkf.GraghkernelFunc.k_func_wl(self.graphs[i_t], self.graphs[j], 1)

            if(self.classes[i_t] / (self.lamda * t) * sigma_loss < 1):
                alpha[i_t] += 1

        return alpha
    
    def predict(self, index, alpha):
        # predict
        predict = []
        for i in index:
            print("index = ",i)
            y = 0
            for j in range(len(self.graphs)):
                y +=  alpha[j] * self.classes[j] * gkf.GraghkernelFunc.k_func_wl(self.graphs[i], self.graphs[j], 1)
            predict.append(y)

        return predict

    def accuracy(self, alpha):
        # accuracy
        picked = np.random.choice(len(self.classes), size=300, replace=False)
        predict = self.predict(picked, alpha)
        
        return np.sum(predict==self.classes[picked])/len(picked)
            
   





