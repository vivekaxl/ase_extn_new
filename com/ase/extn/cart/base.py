'''
Created on 2016-01-23

@author: Atri
'''
import sys
import numpy as np
import scipy.stats as sp
import os
import collections
import math as math
from sklearn import tree
from numpy import mean, dtype
from com.ase.extn.constants import configs
from pygments.lexers.inferno import LimboLexer

'''
Set 
strategy = projective|progressive
system = all|apache|bc|bj|llvm|sqlite|x264
'''
strategy = configs.strategy
system = configs.system

thismodule = sys.modules[__name__]
loc = configs.loc

base_dir = configs.base_dir
base_dir_in = configs.base_dir_in
base_dir_out = configs.base_dir_out

all_systems = configs.all_systems
print_detail = True

'''
details_map holds the following data-
details_map = {<system-id> :[<no_of_features>,<size_of_sample_space>]}
'''
details_map = configs.details_map


def get_min_params(training_set_size):
    if training_set_size > 100:
        min_split = math.floor((training_set_size/100) + 0.5)
        min_bucket = math.floor(min_split/2)
    else:
        min_bucket = math.floor((training_set_size/10) + 0.5)
        min_split = 2 * min_bucket
    
    min_bucket=2 if min_bucket < 2 else min_bucket
    min_split=4 if min_split < 4 else min_split
    return [min_bucket,min_split]
        
   
def load_data():
    fname = os.path.join(base_dir_in,system)
    num_features = range(0,details_map[system][0])
    data = np.loadtxt(fname,  delimiter=',', dtype=bytes,skiprows=1,usecols=num_features).astype(str)
    return data

def load_perf_values():
    fname = os.path.join(base_dir_in,system)
    data = np.loadtxt(fname,  delimiter=',', dtype=float,skiprows=1,usecols=(details_map[system][0],))
    return data

def load_feature_names():
    fname = os.path.join(base_dir_in,system)
    f = open(fname).readline().rstrip('\n').split(',',details_map[system][0])
    return f[:len(f)-1]
    
def cart(X,y):
    training_set_size = X.shape[0]
    params = get_min_params(training_set_size)
    clf = tree.DecisionTreeRegressor(min_samples_leaf=params[0],min_samples_split=params[1])
    clf = clf.fit(X, y)
    return clf

def predict(clf,test_set,values):
    out = clf.predict(test_set) 
    return out

def calc_accuracy(pred_values,actual_values):
    return mean((abs(pred_values - actual_values)/actual_values)*100)

def all_true(in_list):
    for i in in_list:
        if not i:
            return False
    return True  

def progressive(system_val):
    global system
    system = system_val    
    data = load_data()
    perf_values = load_perf_values()
    data[data == 'Y'] = 1
    data[data == 'N'] = 0
    data = data.astype(bool)
    repeat = 3
    total_range = range((details_map[system][1]//10)//configs.th)
    results = np.empty((len(total_range),repeat))
    for j in range(repeat):
        for i in total_range:
            np.random.seed(j)
            curr_size = 10*(i+1)
            training_set_indices = np.random.choice(data.shape[0],curr_size,replace=False)
            diff_indices = set(range(data.shape[0])) - set(training_set_indices)
            training_set = data[training_set_indices]
            test_set_indices = np.random.choice(np.array(list(diff_indices)),curr_size,replace=False)
            test_set = data[test_set_indices]
            X = training_set
            y = perf_values[training_set_indices]
            built_tree = cart(X, y)
            out = predict(built_tree, test_set, perf_values[test_set_indices])
            results[i][j] = calc_accuracy(out,perf_values[test_set_indices])
        print('['+system+']' + " iteration :"+str(j+1))
    print()
    out_file = open(os.path.join(base_dir_out,system)+"_out_"+strategy,'w')
    out_file.truncate()
    
    for i in range(results.shape[0]):
        out_file.write(str((i+1)*10)+","+ str(mean(results[i])))
        out_file.write('\n')

def transform_axes(results):
    curve_data = dict()
    original = np.copy(results)
    
    results[:,0] = np.log10(original[:,0])
    results[:,1] = original[:,1]
    curve_data['log'] = np.copy(results)
    
    results[:,0] = original[:,0]/(original[:,0]+1)
    results[:,1] = original[:,1]
    curve_data['weiss'] = np.copy(results)
    
    results[:,0] = original[:,0]
    results[:,1] = np.log10(original[:,1])
    curve_data['exp'] = np.copy(results)
    
    results[:,0] = np.log10(original[:,0])
    results[:,1] = np.log10(original[:,1])
    curve_data['power'] = np.copy(results)
    return curve_data

def dict_to_array(dict_struct):
    dictlist = []
    for key, value in dict_struct.items():
        if isinstance(value, collections.Iterable):
            value = value[0]
        dictlist.append([key,value])
    return np.array(dictlist)

def smooth(result_array):
    fault_rates = result_array[:,1]
    for i in range(1, len(fault_rates)-1):
        fault_rates[i] = (fault_rates[i-1] + fault_rates[i] + fault_rates[i+1])/3    
    result_array[:,1] = fault_rates
    return result_array

def get_projected_accuracy(size,repeat,data,perf_values):
    results = np.empty((1,repeat))
    for j in range(repeat):
        np.random.seed(j)
        training_set_indices = np.random.choice(data.shape[0],size,replace=False)
        diff_indices = set(range(data.shape[0])) - set(training_set_indices)
        training_set = data[training_set_indices]
        test_set_indices = np.random.choice(np.array(list(diff_indices)),size,replace=False)
        test_set = data[test_set_indices]
        
        X = training_set
        y = perf_values[training_set_indices]
        built_tree = cart(X, y)
        out = predict(built_tree, test_set, perf_values[test_set_indices])
        results[0][j] = 100 - calc_accuracy(out,perf_values[test_set_indices])
    mean = results.mean()
    sd = np.std(results)
    return (mean,sd)
        
def get_optimal(a,b,r,s,curve):
    if curve=='log':
        n = -(r*s*b)/configs.th
    elif curve=='weiss':
        if b > 0:
            n = -1
        else:
            n = np.power(((-r*s*b)/configs.th),0.5)
    elif curve=='power':
        if b < 0:
            n = np.power((-configs.th/(r*s*a*b)),(1/(b-1)))
        else:
            n = -1
    elif curve=='exp':
        if b > 1:
            n = -1
        else :
            n = math.log((-configs.th/(r*s*(a*(np.log(b))))),b)    
    return n

def get_error_from_curve(a,b,n,curve):
    if curve == 'log':
        return a + (b*np.log10(n))
    elif curve == 'weiss':
        return a + (b*n /(n+1))
    elif curve == 'power':
        return a*(np.power(n,b))
    elif curve == 'exp':
        return a*(np.power(b,n))

def cost_eqn(th,n,e,s,r):
    return (th*n + (e*r*s))

def get_intercept(intercept,curve):
    if curve=='power' or curve=='exp':
        return np.exp(intercept)
    else:
        return intercept

def get_slope(slope,curve):
    if curve=='exp':
        return np.exp(slope)
    else:
        return slope

def get_next_size(curve,array,curve_array,index):
    i = curve_array.index(curve)
    s = array[i]
    temp_list = []
    for v in array:
        if v<s:
            temp_list.append(v)
    if len(temp_list) == 0:
        return 0
    else:
        return max(temp_list)
    
        

def select_curve_dynamic(correlation_data,data,perf_values,parameter_dict,results):
    ''' First, we transform the correlation_data structure from key: {a: a1, b:b1...} format to an array where
    keys(curves) are individual rows and columns are detailed data of that curve'''
    trans_array = np.empty([len(correlation_data),len(next(iter(correlation_data.values())))])
    curve_array = []
    index = dict()
    i=0
    for values in next(iter(correlation_data.values())):
        index[values] = i
        i = i+1
    i=0
    for keys in correlation_data:
        value_dict = correlation_data[keys]
        curve_array.append(keys)
        for values in value_dict:
            trans_array[i][index[values]] = value_dict[values]
        i=i+1
    
    lambda_size = len(results)
    curve = select_curve(correlation_data)
    if curve is not None:
        size_to_sample = get_next_size(curve,trans_array[:,index['optimal sample size']],curve_array,index)
        print(size_to_sample)
    else:
        size_to_sample = len(results)
    if size_to_sample > lambda_size:
        lims = [len(results),size_to_sample]
        results = build_data_points(results,1, data, perf_values, False,lims)
        curve_data = transform_axes(smooth(dict_to_array(results)))
        selection_data = dict()
        ''' keys here are individual curves for a given system. Values are 2-d array. x: transformed "no. of sample" values
        and y: transformed accuracy at that sample value'''
        for keys in curve_data:
            slope, intercept, rvalue, pvalue, stderr = sp.stats.linregress(curve_data[keys][configs.ignore_initial:,0],curve_data[keys][configs.ignore_initial:,1])
            value_a = get_intercept(intercept,keys)
            value_b = get_slope(slope,keys)
            value_r = configs.r
            value_s = details_map[system][1]/3
            optimal_size = get_optimal(value_a,value_b,value_r,value_s,keys)
            if optimal_size <= data.shape[0]//configs.th and optimal_size > 1:
                mean_accu,sd = get_projected_accuracy(optimal_size,configs.repeat,data,perf_values)
                proj_err = get_error_from_curve(parameter_dict[keys]['a'], parameter_dict[keys]['b'], optimal_size, keys)
                diff = abs((100-mean_accu) - proj_err)
                selection_data[keys] = {'correlation' : rvalue,
                                      'accuracy' :mean_accu,
                                      'diff' :diff
                                    }
            else:
                mean_accu,sd,total_cost = (None,None,None)
            
            
        selected_curve = select_curve(selection_data)
        selected_curve_2 = select_curve_diff_error(selection_data,data,perf_values,parameter_dict,results)
        print('diff: '+ str(selected_curve_2))
        print('corr:' + str(selected_curve))
        print('original' +str(curve))
        

def select_curve(correlation_data):
    curve = None
    min_corr = -0.7
    for keys in correlation_data:
        if float(correlation_data[keys]['correlation']) < min_corr and correlation_data[keys]['accuracy'] is not None:
            min_corr = float(correlation_data[keys]['correlation'])
    for keys in correlation_data:
        if float(correlation_data[keys]['correlation']) == min_corr:
            curve = keys
    return curve    

def select_curve_diff_error(correlation_data,data,perf_values,parameter_dict,results):
    curve = ''
    min_diff = 100
    for keys in correlation_data:
        if float(correlation_data[keys]['diff']) < min_diff and correlation_data[keys]['diff'] is not None:
            min_diff = float(correlation_data[keys]['diff'])
    for keys in correlation_data:
        if float(correlation_data[keys]['diff']) == min_diff:
            curve = keys
    return curve    

def build_data_points(results,repeat,data,perf_values,stop_by_freq,lims):
    '''
    Initialise frequency table values to a 'ridiculous' number for all mandatory features
    if data is all true for a feature <=> mandatory <=> frequency table value = sys.maxsize 
    (These kind of hacks causes funny bugs!!)
    '''
    freq_table = np.empty([2,details_map[system][0]])
    for k in range(details_map[system][0]):
        if all_true(data[:,k]==1):
            freq_table[:,k]=sys.maxsize
    
    
    for j in range(repeat):
        if lims is not None:
            i = lims[0]
        else:
            i = 0
        while True:
            '''print('['+system+']'+' running size :'+ str(i+1))'''
            curr_size = (i+1)
            np.random.seed(j)
            training_set_indices = np.random.choice(data.shape[0],curr_size,replace=False)
            diff_indices = set(range(data.shape[0])) - set(training_set_indices)
            training_set = data[training_set_indices]
            test_set_indices = np.random.choice(np.array(list(diff_indices)),curr_size,replace=False)
            test_set = data[test_set_indices]
            X = training_set
            y = perf_values[training_set_indices]
            built_tree = cart(X, y)
            out = predict(built_tree, test_set, perf_values[test_set_indices])
            if curr_size in results:
                results[curr_size].append(calc_accuracy(out,perf_values[test_set_indices]))
            else:
                results[curr_size] = [calc_accuracy(out,perf_values[test_set_indices])]
            
            if stop_by_freq is True:
                '''
                Update frequency table based on training set feature activation/de-activation
                We are refreshing the values in each iteration instead of making it incremental.
                This is in-efficient but keeps thing simple.
                '''
                for k in range(details_map[system][0]):
                    if not freq_table[0][k]==sys.maxsize:
                        active_count = np.count_nonzero(training_set[:,k])
                        deactive_count = training_set.shape[0] - active_count
                        freq_table[0][k] = active_count
                        freq_table[1][k] = deactive_count
                    else:
                        continue
                        
                '''
                We are done if the frequency table values hits the threshold
                '''
                if np.all(freq_table>=configs.projective_feature_threshold):
                    break
                i=i+1
            else:
                i=i+1
                if i > lims[1]:
                    break
    return results
                
def projective(system_val):
    if print_detail is True:
        print('System-id : '+system_val)
        print('R value : '+str(configs.r))
        print('th value : '+str(configs.th))
    global system
    system = system_val
    data = load_data()
    perf_values = load_perf_values()
    data[data == 'Y'] = 1
    data[data == 'N'] = 0
    data = data.astype(bool)    
    repeat = configs.repeat
    results = dict()
    results = build_data_points(results,repeat, data, perf_values, True,None)
    
    '''
    We account for variation in the size of the lambda set due to random sampling. We consider sizes which were present in
    at least 90% of the runs. 
    '''
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
            
                
        if print_detail is True:
            print('Size of lambda set: '+ str(len(results)))    
        '''
        Transform the axes and calculate pearson correlation with
        each learning curve
        '''
        curve_data = transform_axes(smooth(dict_to_array(results)))
        parameter_dict = dict()
        correlation_data = dict()
        ''' keys here are individual curves for a given system. Values are 2-d array. x: transformed "no. of sample" values
        and y: transformed accuracy at that sample value'''
        for keys in curve_data:
            slope, intercept, rvalue, pvalue, stderr = sp.stats.linregress(curve_data[keys][configs.ignore_initial:,0],curve_data[keys][configs.ignore_initial:,1])
            value_a = get_intercept(intercept,keys)
            value_b = get_slope(slope,keys)
            parameter_dict[keys] = {'a' : value_a, 'b':value_b}
            value_r = configs.r
            value_s = details_map[system][1]/3
            optimal_size = get_optimal(value_a,value_b,value_r,value_s,keys)
            estimated_error = 100
            if optimal_size <= data.shape[0]//configs.th and optimal_size > 1:
                mean_accu,sd = get_projected_accuracy(optimal_size,repeat,data,perf_values)
                r = configs.r
                th = configs.th
                total_cost = cost_eqn(th,optimal_size, 100-float(mean_accu), details_map[system][1] // 3, r)
                estimated_error = get_error_from_curve(value_a,value_b,optimal_size,keys)
                estimated_cost = cost_eqn(th,optimal_size,estimated_error,details_map[system][1] // 3, r)
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
        if configs.curve_selection == 'dynamic':
            select_curve_dynamic(correlation_data,data,perf_values,parameter_dict,results)
        selected_curve = select_curve(correlation_data)
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
    return correlation_data
 
def main():           
    if system=='all':
        for i in all_systems:
            func = getattr(thismodule, strategy)
            func(i)
    else:
        func = getattr(thismodule, strategy)
        return func(system)     
main()