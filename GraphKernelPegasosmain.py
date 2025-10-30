# -*- coding: utf-8 -*-
import numpy as np
import pegasos_evaluation as pg
import data
import time

#-------------------
# Make data
myData = data.classification(negLabel=-1.0,posLabel=1.0)
myData.makeData(dataType=int(input("data type =>")))
#-------------------

#-------------------
# Learning and evaluation.
print("lamda,acc")
for lamda in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10, 100]:
    Pegasos = pg.Pegasos(myData.graphs, myData.classes, 500, lamda)

    start = time.time()
    
    # default
    # cProfile.run("alpha = Pegasos.train()")
    alpha = Pegasos.train() 
    

    #cython
    #cProfile.run("train(np.array(Pegasos.G_train, dtype=object), Pegasos.y_train.astype(np.int32), Pegasos.iter,Pegasos.lamda, Pegasos.delta_c)")
    # alpha = train(np.array(Pegasos.G_train, dtype=object),  
    #               Pegasos.y_train.astype(np.int32),         
    #               Pegasos.iter,
    #               Pegasos.lamda,
    #               Pegasos.delta_c)
    
    end = time.time()
    print(f"処理時間: {end - start:.4f} 秒")
    
    acc = Pegasos.accuracy(alpha)
    print(lamda,",",acc)
    print("基底数", len(np.where(alpha > 0)[0]))

# iter = int(input("iter = "))
# lamda = float(input("lamda ="))
# Pegasos = pg.Pegasos(myData.graphs, myData.classes, iter, lamda)
# alpha = Pegasos.train()
# acc = Pegasos.accuracy(alpha)
# print("acc = ", acc)
#-------------------

