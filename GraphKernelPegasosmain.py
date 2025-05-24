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
iter = int(input("iter = "))
lamda = float(input("lamda ="))

Pegasos = pg.Pegasos(myData.graphs, myData.classes, iter, lamda)

alpha = Pegasos.train()

acc = Pegasos.accuracy(alpha)

print("acc = ", acc)

