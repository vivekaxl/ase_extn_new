'''
Created on 2016-02-01

@author: Atri
'''
import os
import sys

thismodule = sys.modules[__name__]
loc = os.path.dirname(__file__)

base_dir = os.path.join(loc,'data')
base_dir_in = os.path.join(base_dir,'input')
base_dir_tway_in = os.path.join(base_dir_in,'tway')
base_dir_out = os.path.join(base_dir,'output')

details_map = {"apache" : [9,192], "llvm" : [11,1024], "x264" : [16,1152], "bc" : [18,2560], "bj" : [26,180], "sqlite" : [39,4553]}
all_systems = ['apache','bc','bj','llvm','sqlite','x264']

projective_feature_threshold = 5
repeat = 30

tway = 2

sense_curve = True
ignore_initial = 4
curve_selection = 'dynamic'

r_0_to_1 = False

strategy = 'projective'
system = 'x264'

''' r = Cost of prediction error / Cost of measurement'''
r = 1
th = 2