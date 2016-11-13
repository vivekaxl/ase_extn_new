'''
Created on 2016-02-11

@author: Atri
'''
import os
import numpy as np
import scipy.stats as sp
from com.ase.extn.constants import configs
from sklearn import tree
from com.ase.extn.cart import base

base_dir = configs.base_dir
base_dir_in = configs.base_dir_tway_in
base_dir_out = configs.base_dir_out
system = configs.system
file_name_train = system+'_'+str(configs.tway)+'_way_perf_train'
file_name_test = system+'_'+str(configs.tway)+'_way_perf_test'
print_detail = True

def load_data(train):
    fname = os.path.join(base_dir_in,file_name_train) if train is True else os.path.join(base_dir_in,file_name_test)
    num_features = range(0,configs.details_map[system][0])
    data = np.loadtxt(fname,  delimiter=',', dtype=bytes,skiprows=1,usecols=num_features).astype(str)
    return data

def load_perf_values(train):
    fname = os.path.join(base_dir_in,file_name_train) if train is True else os.path.join(base_dir_in,file_name_test)
    data = np.loadtxt(fname,  delimiter=',', dtype=float,skiprows=1,usecols=(configs.details_map[system][0],))
    return data

def get_projected_accuracy(optimal_size,data_train,perf_values_train,data_test,perf_values_test):
    results = np.empty((1,configs.repeat))
    for j in range(configs.repeat):
        np.random.seed(j)
        if optimal_size > data_train.shape[0]:
            training_set_indices = np.random.choice(data_test.shape[0],(optimal_size-data_train.shape[0]),replace=False)
            diff_indices = set(range(data_test.shape[0])) - set(training_set_indices)
            temp = data_test[training_set_indices]
            training_set = np.append(temp,data_train,0)
            test_set_indices = np.random.choice(np.array(list(diff_indices)),optimal_size,replace=False)
            test_set = data_test[test_set_indices]
            y = np.append(perf_values_test[training_set_indices],perf_values_train)
            
        else:
            training_set_indices = np.random.choice(data_train.shape[0],optimal_size,replace=False)
            training_set = data_train[training_set_indices]
            test_set_indices = np.random.choice(data_test.shape[0],optimal_size,replace=False)
            test_set = data_test[test_set_indices]
            y = perf_values_train[training_set_indices]
            
        X = training_set
        built_tree = base.cart(X, y)
        out = base.predict(built_tree, test_set, perf_values_test[test_set_indices])
        results[0][j] = 100 - base.calc_accuracy(out,perf_values_test[test_set_indices])
    mean = results.mean()
    sd = np.std(results)
    return (mean,sd)

def sample():
    data_train = load_data(True)
    perf_values_train = load_perf_values(True)
    data_test = load_data(False)
    perf_values_test = load_perf_values(False)
    
    data_train[data_train == 'Y'] = 1
    data_train[data_train == 'N'] = 0
    data_train = data_train.astype(bool)    
    
    data_test[data_test == 'Y'] = 1
    data_test[data_test == 'N'] = 0
    data_test = data_test.astype(bool)
    
    repeat = configs.repeat
    results = dict()
    print('Size of '+str(system)+' '+str(configs.tway)+'-way sample is: '+str(data_train.shape[0]))
    for j in range(configs.repeat):
        i=0
        while True:
            if i==data_train.shape[0]:
                break
            else:
                i=i+1
            curr_size = i
            np.random.seed(j)
            training_set_indices = np.random.choice(data_train.shape[0],curr_size,replace=False)
            training_set = data_train[training_set_indices]
            
            test_set_indices = np.random.choice(data_test.shape[0],curr_size,replace=False)
            test_set = data_test[test_set_indices]
            
            X = training_set
            y = perf_values_train[training_set_indices]
            
            built_tree = base.cart(X, y)
            out = base.predict(built_tree, test_set, perf_values_test[test_set_indices])
            
            if curr_size in results:
                results[curr_size].append(base.calc_accuracy(out,perf_values_test[test_set_indices]))
            else:
                results[curr_size] = [base.calc_accuracy(out,perf_values_test[test_set_indices])]
            
            
    results_hold = dict()
    
    for size in results:
        mean_fault = sum(results[size])/float(len(results[size]))
        if len(results[size]) >= (repeat - 0.1*repeat) and mean_fault>4.9:
            results_hold[size] = sum(results[size])/float(len(results[size]))
    results_orig = results.copy()
    for s in range(repeat):
        if configs.sense_curve is False:
            results=results_hold
        else:
            results = dict()
            for entry in results_orig:
                if s < len(results_orig[entry]):
                    results[entry] = results_orig[entry][s]
            
                
        print('Size of lambda set: '+ str(len(results)))    
        '''
        Transform the axes and calculate pearson correlation with
        each learning curve
        '''
        curve_data = base.transform_axes(base.smooth(base.dict_to_array(results)))
        parameter_dict = dict()
        correlation_data = dict()
        ''' keys here are individual curves for a given system. Values are 2-d array. x: transformed "no. of sample" values
        and y: transformed accuracy at that sample value'''
        for keys in curve_data:
            slope, intercept, rvalue, pvalue, stderr = sp.stats.linregress(curve_data[keys][configs.ignore_initial:,0],curve_data[keys][configs.ignore_initial:,1])
            value_a = base.get_intercept(intercept,keys)
            value_b = base.get_slope(slope,keys)
            parameter_dict[keys] = {'a' : value_a, 'b':value_b}
            value_r = configs.r
            value_s = configs.details_map[system][1]/3
            optimal_size = base.get_optimal(value_a,value_b,value_r,value_s,keys)
            estimated_error = 100
            if optimal_size <= (data_train.shape[0]+data_test.shape[0])//configs.th and optimal_size > 1:
                mean_accu,sd = get_projected_accuracy(optimal_size,data_train,perf_values_train,data_test,perf_values_test)
                r = configs.r
                th = configs.th
                total_cost = base.cost_eqn(th,optimal_size, 100-float(mean_accu), configs.details_map[system][1] // 3, r)
                estimated_error = base.get_error_from_curve(value_a,value_b,optimal_size,keys)
                estimated_cost = base.cost_eqn(th,optimal_size,estimated_error,configs.details_map[system][1] // 3, r)
            else:
                mean_accu,sd,total_cost,estimated_cost = (None,None,None,None)
            
            correlation_data[keys] = {'correlation' : rvalue,
                                      'p-value' : str(pvalue),
                                      'optimal sample size' :int(optimal_size),
                                      'accuracy' :mean_accu,
                                      'estimated accuracy': 100 - estimated_error,
                                      'standard deviation' :sd,
                                      'total cost' :total_cost,
                                      'estimated cost' : estimated_cost}
        selected_curve = base.select_curve(correlation_data)
        if print_detail is True:
            print()
            print('Detailed learning projections:')
            print('<curve-id> : {<details>}')
            print()
            
        for keys in correlation_data:
            if keys == selected_curve:
                correlation_data[keys]['selected'] = True
                if print_detail is True:
                    print(str(keys) +"**:"+str(correlation_data[keys]))
            else:
                correlation_data[keys]['selected'] = False
                if print_detail is True:
                    print(str(keys) +":"+str(correlation_data[keys]))
        if print_detail is True:            
            print("-----------------------------------------------")
            print()
        if configs.sense_curve is False:
            break     
    ''' It would return meaningful data only when sense_curve is False'''        
    
sample()