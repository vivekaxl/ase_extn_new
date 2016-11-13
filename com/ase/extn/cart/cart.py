'''
Created on 2016-01-23

@author: Atri
'''
import sys
import numpy as np
import scipy.stats as sp
import os
import math as math
from sklearn import tree
from numpy import mean

'''
Set 
strategy = projective|progressive
system = all|apache|bc|bj|llvm|sqlite|x264
'''
strategy = "projective"
system = 'all'

thismodule = sys.modules[__name__]
loc = os.path.dirname(__file__)

base_dir = os.path.join(loc,'data/')
base_dir_in = base_dir+'input/'
base_dir_out = base_dir+'output/'

all_systems = ['apache','bc','bj','llvm','sqlite','x264']
'''
details_map holds the following data-
details_map = {<system-id> :[<no_of_features>,<size_of_sample_space>]}
'''
details_map = {"apache" : [9,192], "llvm" : [11,1024], "x264" : [16,1152], "bc" : [18,2560], "bj" : [26,180], "sqlite" : [39,4553]}


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
    fname = base_dir_in+system
    num_features = range(0,details_map[system][0])
    data = np.loadtxt(fname,  delimiter=',', dtype=bytes,skiprows=1,usecols=num_features).astype(str)
    return data

def load_perf_values():
    fname = base_dir_in+system
    data = np.loadtxt(fname,  delimiter=',', dtype=float,skiprows=1,usecols=(details_map[system][0],))
    return data

def load_feature_names():
    fname = base_dir_in+system
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
    repeat = 30
    total_range = range((details_map[system][1]//10)//2)
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
    out_file = open(base_dir_out+system+"_out_"+strategy,'w')
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
        n = -(r*s*b)/2
    elif curve=='weiss':
        n = np.power(((-r*s*b)/2),0.5)
    elif curve=='power':
        n = np.power((-2/(r*s*a*b)),(1/(b-1)))
    elif curve=='exp':
        n = math.log((-2/(r*s*(a*(np.log(b))))),b)    
    return n

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
                
def projective(system_val):
    print('System-id : '+system_val)
    global system
    system = system_val
    data = load_data()
    perf_values = load_perf_values()
    data[data == 'Y'] = 1
    data[data == 'N'] = 0
    data = data.astype(bool)    
    repeat = 30
    threshold = 5
    '''
    Initialise frequency table values to a 'ridiculous' number for all mandatory features
    if data is all true for a feature <=> mandatory <=> frequency table value = sys.maxsize 
    (These kind of hacks causes funny bugs!!)
    '''
    freq_table = np.empty([2,details_map[system][0]])
    for k in range(details_map[system][0]):
        if all_true(data[:,k]==1):
            freq_table[:,k]=sys.maxsize
    
    results = dict()
    for j in range(repeat):
        i=0
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

            # print list(freq_table[0])
            # print list(freq_table[1])
            # print results

            '''
            We are done if the frequency table values hits the threshold
            '''
            if np.all(freq_table>=threshold):
                break
            i=i+1
    
    '''
    We account for variation in the size of the lambda set due to random sampling. We consider sizes which were present in
    at least 90% of the runs. 
    '''
    results_hold = dict()
    for size in results:
        mean_fault = sum(results[size])/float(len(results[size]))
        if len(results[size]) >= (repeat - 0.1*repeat) and mean_fault>4.9:
            results_hold[size] = sum(results[size])/float(len(results[size]))
    results=results_hold
    # print('Size of lambda set: '+ str(len(results)))
    '''
    Transform the axes and calculate pearson correlation with
    each learning curve
    '''
    curve_data = transform_axes(smooth(dict_to_array(results)))
    correlation_data = dict()
    for keys in curve_data:
        slope, intercept, rvalue, pvalue, stderr = sp.stats.linregress(curve_data[keys][1:,0],curve_data[keys][1:,1])
        value_a = get_intercept(intercept,keys)
        value_b = get_slope(slope,keys)
        value_r = 1
        value_s = details_map[system][1]/3
        optimal_size = get_optimal(value_a,value_b,value_r,value_s,keys)
        if optimal_size <= data.shape[0]//2:
            mean_accu,sd = get_projected_accuracy(optimal_size,repeat,data,perf_values)
        else:
            mean_accu,sd = (None,None)
        correlation_data[keys] = {'correlation' : rvalue,
                                  'optimal sample size' :int(optimal_size),
                                  'accuracy' :mean_accu,
                                  'standard deviation' :sd}
    print()
    print('Detailed learning projections:')
    print('<curve-id> : {<details>}')
    print()
    for keys in correlation_data:
        print(str(keys) +":"+str(correlation_data[keys]))
    print("-----------------------------------------------")
    print()
 
def main():           
    if system=='all':
        for i in all_systems:
            func = getattr(thismodule, strategy)
            func(i)
    else:
        func = getattr(thismodule, strategy)
        func(system)

main()        