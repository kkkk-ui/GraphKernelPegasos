import numpy as np
import GraphKernelFunc as gkf
from sklearn.model_selection import train_test_split

class Pegasos():
    def __init__(self, graphs, classes, iter, lamda):
        self.iter = iter
        self.lamda = lamda
        self.delta_c = 0.8
        self.relax = 0.5
        self.G_train, self.G_test, self.y_train, self.y_test = train_test_split(graphs, classes, test_size=0.1)

    def train(self):
        # learning
        # initialize
        alpha = np.zeros(len(self.G_train))
        kernel_cache = np.zeros((len(self.G_train), 2))
    
        # iterate
        for t in range(1,self.iter+1):
            eta_t = 1.0 / t 

            i_t = np.random.randint(len(self.G_train))
            sigma_loss = 0

            support_indices = np.where(np.abs(alpha) > 0)[0]
            if support_indices.shape[0] > 0:
                kernel_vec = gkf.GraphkernelFunc.k_vec_wl(self.G_train[i_t], np.array(self.G_train)[support_indices], 2)

            for idx, j in enumerate(support_indices):
                kernel_cache[j][0] = kernel_vec[idx]

            alphas = alpha[support_indices]
            ys = self.y_train[support_indices]
            ks = kernel_cache[support_indices, 0]
            sigma_loss = np.dot(alphas * ys, ks)
            
            if(self.y_train[i_t] / (self.lamda * t) * sigma_loss < 1):
                if (t==1):
                    alpha[i_t] += 1
                    kernel_cache[i_t] = [1, t]
                # optimize dictionary(Whether to add or not)
                if (t>=2):
                    coh_max = 0
                    for j in support_indices:
                        if(j == i_t):
                            continue
                    
                        coh = kernel_cache[j][0]
                        if(coh > coh_max):
                            coh_max = coh

                    if(coh_max < self.delta_c):
                        # alpha[i_t] += eta_t / self.lamda
                        alpha[i_t] += 1
                        kernel_cache[i_t] = [1, t]
            # --------------------------------------------------------------------------------- #
            # projection
            support_indices = np.where(np.abs(alpha) > 0)[0]
            myKernel = gkf.GraphkernelFunc(2, kernelParam=1)
            gram_mat = myKernel.createMatrix(np.array(self.G_train)[support_indices], 2)

            ys = self.y_train[support_indices]
            ks = kernel_cache[support_indices, 0]
            phi = np.dot(alpha[support_indices] * ys, ks)
            yt = self.y_train[i_t]
            e = yt - phi 

            # Tikhonov 正則化パラメータ (epsilon) を定義
            # データとカーネルに依存するが、1e-6 や 1e-8 が一般的
            epsilon = 1e-8

            # Gram行列の対角成分に epsilon を加えることで正則化
            # G_reg = G_n + epsilon * I を作成
            regularized_gram_mat = gram_mat + epsilon * np.identity(gram_mat.shape[0])

            # direction vector
            # 正則化された Gram行列を用いて z を計算
            # (G_n + epsilon*I) * z = k を解く
            z = np.linalg.solve(regularized_gram_mat, ks)

            # norm of direction vector (G_nノルムを計算)
            # k^T * z で正規化されたノルムを計算
            denom = ks @ z                            

            step = self.relax * (e / denom)
            alpha[support_indices] += (step * z) / ys 
            # --------------------------------------------------------------------------------- #
    
        return alpha
    
    def predict(self, alpha):
        # predict
        predict = []
        truevalue = []
        for i in range(len(self.G_test)):
            y = 0
            
            support_indices = np.where(np.abs(alpha) > 0)[0]
            if support_indices.shape[0] > 0:
                ks = gkf.GraphkernelFunc.k_vec_wl(self.G_test[i], np.array(self.G_train)[support_indices], 2)
            alphas = alpha[support_indices]
            ys = self.y_train[support_indices]
            y = np.dot(alphas * ys, ks) 

            predict.append(np.sign(y))
            truevalue.append(self.y_test[i])

        return np.array(predict), np.array(truevalue)

    def accuracy(self, alpha):
        # accuracy
        predict, truevalue = self.predict(alpha)
        return (predict == truevalue).mean()
            
   




