# -*- coding: utf-8 -*-
import numpy as np
import pegasos_evaluation_proj as pg
import data
import time

#-------------------
# Make data
myData = data.classification(negLabel=-1.0,posLabel=1.0)
myData.makeData(dataType=int(input("data type =>")))
#-------------------

#-------------------
# Learning and evaluation.
print("t,acc")
for t in [100, 300, 500, 1000, 2000, 3000, 4000, 5000]:
    Pegasos = pg.Pegasos(myData.graphs, myData.classes, t, 1e-2)

    start = time.time()
    
    alpha = Pegasos.train() 
    
    end = time.time()
    print(f"処理時間: {end - start:.4f} 秒")
    
    acc = Pegasos.accuracy(alpha)
    print(t,",",acc)
    print("基底数", len(np.where(np.abs(alpha) > 0)[0]))

#-------------------
