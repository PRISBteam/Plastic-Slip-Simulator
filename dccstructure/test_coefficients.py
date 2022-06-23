# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 10:58:34 2022

@author: fonso
"""

import numpy as np
from math import factorial

def coef_star(p):
    """
    Parameters
    ----------
    p : int OR list of ints
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.
    """
    
    # Combining Theorem 5.2 from Wilson (2007) and the formula for the star operator in Berbatov (2022), the formula for the star
    # operator in simplicial complexes ends up having this coefficient.
    
    if type(p) == int:
    
        return (factorial(3 - p) * factorial(p)) / factorial(3 + 1)
    
    if type(p) in [list, np.ndarray]:
        
        return [(factorial(3 - i) * factorial(i)) / factorial(3 + 1) for i in p]


def coef_cup(p,q):
    """
    Parameters
    ----------
    p : int OR list of ints
        DESCRIPTION.
    q : int OR list of ints
        DESCRIPTION.

    Returns
    -------
    
    """
    
    # This is the coefficient in the cup product as defined for simplicial complexes by Wilson (2007).
    
    if type(p) == int and type(q) == int:
    
        return (factorial(p) * factorial(q)) / factorial(p + q + 1)
    
    if type(p) in [list, np.ndarray] and type(q) == int:
        
        return [(factorial(p) * factorial(q)) / factorial(p + q + 1) for i in p]
    
    if type(p) == int and type(q) in [list, np.ndarray]:
        
        return [(factorial(p) * factorial(q)) / factorial(p + q + 1) for i in q]

    if type(p) in [list, np.ndarray] and type(q) in [list, np.ndarray]:
        
        a = np.zeros((len(p), len(q)))
        
        for i in p:
            for j in q:
                
                a[i,j] = (factorial(i) * factorial(j)) / factorial(i + j + 1)
                
        return a
