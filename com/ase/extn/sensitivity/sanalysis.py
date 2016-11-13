'''
Created on 2016-02-04

@author: Atri
'''

from SALib.sample import sobol_sequence
from com.ase.extn.cart import base
from com.ase.extn.constants import configs
import numpy as np


param_values = sobol_sequence.sample(10, 1)
param_values = np.append(param_values,1)
base.print_detail = False

sens_data = dict()
print("System-id : "+str(configs.system))
for values in param_values:
    if values <= 1 and values >=.1:
        if configs.r_0_to_1 is not True:
            configs.r = float(1 + (9*values))
        else:
            configs.r = float(values)
        correlation_data = base.main()
        for keys in correlation_data:
            if correlation_data[keys]['selected'] is True:
                data_list = [keys,correlation_data[keys]['correlation'],correlation_data[keys]['optimal sample size'],correlation_data[keys]['accuracy'],
                             correlation_data[keys]['standard deviation'],correlation_data[keys]['total cost']]
                ''' If optimal sample size is found to be more than the maximal score set (no. of un-measured configurations)
                then, it is not a great idea to use the prediction model. maximal score set = total configs - (training + testing)
                or = total configs - th*(training))'''
                if correlation_data[keys]['optimal sample size'] > (configs.details_map[configs.system][1] - (configs.th * correlation_data[keys]['optimal sample size'])) or correlation_data[keys]['optimal sample size'] < 1:
                    data_list.append('UNR')
                print(str(configs.r) + " - " + str(data_list))  
        sens_data[str(configs.r)] = data_list       
                    