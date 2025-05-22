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


