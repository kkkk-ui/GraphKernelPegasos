# -*- coding: utf-8 -*-
import numpy as np
import pegasos_evaluation as pg
import data

#-------------------
# Make data
myData = data.classification(negLabel=-1.0,posLabel=1.0)
myData.makeData(dataType=int(input("data type =>")))
#-------------------

#-------------------
# Learning and evaluation.
print("lamda,acc")
for lamda in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
    Pegasos = pg.Pegasos(myData.graphs, myData.classes, 100, lamda)
    alpha = Pegasos.train()
    acc = Pegasos.accuracy(alpha)
    print(lamda,",",acc)

# iter = int(input("iter = "))
# lamda = float(input("lamda ="))
# Pegasos = pg.Pegasos(myData.graphs, myData.classes, iter, lamda)
# alpha = Pegasos.train()
# acc = Pegasos.accuracy(alpha)
# print("acc = ", acc)
#-------------------

