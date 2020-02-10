# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 13:30:37 2019

@author: Mohammadmahdi
"""
import numpy as np


def berovey(B1,B2,B3,PAN):
    
    BB = B1 + B2 + B3
    R = ((B3)/(BB))*PAN
    G = ((B2)/(BB))*PAN
    B = ((B1)/(BB))*PAN
    
    return R,G,B