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
    Pegasos = pg.Pegasos(myData.graphs, myData.classes, 100, lamda)

    start = time.time()
    alpha = Pegasos.train()
    end = time.time()
    print(f"処理時間: {end - start:.4f} 秒")
    
    acc = Pegasos.accuracy(alpha)
    print(lamda,",",acc)

# iter = int(input("iter = "))
# lamda = float(input("lamda ="))
# Pegasos = pg.Pegasos(myData.graphs, myData.classes, iter, lamda)
# alpha = Pegasos.train()
# acc = Pegasos.accuracy(alpha)
# print("acc = ", acc)
#-------------------

