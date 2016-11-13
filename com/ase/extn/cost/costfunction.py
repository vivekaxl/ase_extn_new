'''
Created on 2016-02-01

@author: Atri

'''

from com.ase.extn.constants import configs 
import os
import numpy as np

system = configs.system

def load_output():
    file_to_load = os.path.join(configs.base_dir_out,system+'_out_progressive')
    data = np.loadtxt(file_to_load,  delimiter=',')
    return data

def cost_eqn(n,e,s,r):
    return (configs.th*n + (e*r*s))

'''
This is called for each system.
'''    
def minima_progressive():
    data = load_output()
    cost_curve = np.empty((len(data),1))
    for i in range(len(data)):
        n = data[i][0]
        e = data[i][1]
        R = configs.r
        S = configs.details_map[system][1]//3
        cost_curve[i][0] = cost_eqn(n, e, S, R)
        if i > 0 and cost_curve[i][0] > cost_curve[i-1][0]:
            print(data[i-1][0],data[i-1][1],cost_curve[i-1][0])
            break        


    
minima_progressive()   