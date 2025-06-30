import numpy as np
import GraphKernelFunc as gkf
from sklearn.model_selection import train_test_split

class Pegasos():
    def __init__(self, graphs, classes, iter, lamda):
        self.iter = iter
        self.lamda = lamda
        self.delta_c = 0.9
        self.G_train, self.G_test, self.y_train, self.y_test = train_test_split(graphs, classes, test_size=0.1)

    def train(self):
        # learning

        # initialize
        alpha = np.zeros(len(self.G_train))
        kernel_cache = np.zeros((len(self.G_train), 2))
    
        # iterate
        for t in range(1,self.iter+1):
            i_t = np.random.randint(len(self.G_train))
            sigma_loss = 0

            support_indices = np.where(alpha > 0)[0]
            if support_indices.shape[0] > 0:
                kernel_vec = gkf.GraphkernelFunc.k_vec_wl(self.G_train[i_t], np.array(self.G_train)[support_indices], 2)

            for idx, j in enumerate(support_indices):
                kernel_cache[j][0] = kernel_vec[idx]

            alphas = alpha[support_indices]
            ys = self.y_train[support_indices]
            ks = kernel_cache[support_indices, 0]
            sigma_loss = np.dot(alphas * ys, ks)
            
            if(self.y_train[i_t] / (self.lamda * t) * sigma_loss < 1):
                alpha[i_t] += 1
                kernel_cache[i_t] = [1, t]

                #optimize dictionary
                if (t>100):
                    # optimize dictionary
                    coh_max = 0
                    coh_index = 0
                    for j in support_indices:
                        if(j == i_t):
                            continue

                        if(t - kernel_cache[j][1] < 100):
                            continue

                        if(alpha[j] > 1):
                            continue

                        coh = kernel_cache[j][0]
                        if(coh > coh_max):
                            coh_max = coh
                            coh_index = j    

                    if(coh_max > self.delta_c):
                        alpha[coh_index] = 0
    
        return alpha
    
    def predict(self, alpha):
        # predict
        predict = []
        truevalue = []
        for i in range(len(self.G_test)):
            y = 0
            
            support_indices = np.where(alpha > 0)[0]
            y = sum(alpha[j] * self.y_train[j] * gkf.GraphkernelFunc.k_func_wl(self.G_test[i], self.G_train[j], 2) for j in support_indices) #<class 'numpy.ndarray'>

            predict.append(np.sign(y)[0])
            truevalue.append(self.y_test[i])

        return np.array(predict), np.array(truevalue)

    def accuracy(self, alpha):
        # accuracy
        predict, truevalue = self.predict(alpha)
        return (predict == truevalue).mean()
            
   




