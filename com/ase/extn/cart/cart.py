'''
Created on 2016-01-23

@author: Atri
'''
from __future__ import division
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

all_systems = ["Apache.csv", "BDBC.csv", "Dune.csv", "HSMGP.csv", "rs-6d-c3_obj1.csv",
               "rs-6d-c3_obj2.csv", "sol-6d-c2-obj1.csv", "sol-6d-c2-obj2.csv", "sort_256_obj2.csv",
               "SQL.csv", "wc+rs-3d-c4-obj1.csv", "wc+rs-3d-c4-obj2.csv", "wc+sol-3d-c4-obj1.csv",
               "wc+sol-3d-c4-obj2.csv", "wc+wc-3d-c4-obj1.csv", "wc+wc-3d-c4-obj2.csv", "wc-3d-c4_obj2.csv",
               "wc-6d-c1-obj1.csv", "wc-6d-c1-obj2.csv", "WGet.csv", "X264_AllMeasurements.csv"]

# all_systems = ['sort_256_obj2.csv']
'''
details_map holds the following data-
details_map = {<system-id> :[<no_of_features>,<size_of_sample_space>]}
'''
details_map = {
    "Apache.csv" : [9, 192],
    "BDBC.csv" : [18, 2560],
    "Dune.csv" : [11, 2304],
    "HSMGP.csv" : [14, 3456],
    "rs-6d-c3_obj1.csv" : [6, 3840],
    "rs-6d-c3_obj2.csv" : [6, 3840],
    "sol-6d-c2-obj1.csv" : [6, 2866],
    "sol-6d-c2-obj2.csv" : [6, 2862],
    "sort_256_obj2.csv" : [3, 206],
    "SQL.csv" : [39, 4653],
    "wc+rs-3d-c4-obj1.csv" : [3, 196],
    "wc+rs-3d-c4-obj2.csv" : [3, 196],
    "wc+sol-3d-c4-obj1.csv" : [3, 196],
    "wc+sol-3d-c4-obj2.csv" : [3, 196],
    "wc+wc-3d-c4-obj1.csv" : [3, 196],
    "wc+wc-3d-c4-obj2.csv" : [3, 196],
    "wc-3d-c4_obj2.csv" : [3, 756],
    "wc-6d-c1-obj1.csv" : [6, 2880],
    "wc-6d-c1-obj2.csv" : [6, 2880],
    "WGet.csv" : [16, 188],
    "X264_AllMeasurements.csv" : [16, 1152],

}


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
    data = np.loadtxt(fname,  delimiter=',', dtype=bytes,skiprows=1,usecols=num_features).astype(float)
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
    accs = []
    for pred, act in zip(pred_values, actual_values):
        if act != 0:
            accs.append((abs(pred-act)/abs(act))*100)
    return mean(accs)

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
    import pdb
    pdb.set_trace()
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
    repeat = 30
    threshold = 5
    '''
    Initialise frequency table values to a 'ridiculous' number for all mandatory features
    if data is all true for a feature <=> mandatory <=> frequency table value = sys.maxsize
    (These kind of hacks causes funny bugs!!)
    '''
    # find number of unique values in each column
    freq_table = {}
    number_of_columns = data.shape[1]
    for column_no in xrange(number_of_columns):
        unique_values = set(data[:, column_no])
        freq_table[column_no] = {}
        for unique_value in unique_values:
            if len(unique_values) != 1:
                freq_table[column_no][unique_value] = 0
            else:
                freq_table[column_no][unique_value] = sys.maxsize  # this feature always takes only single value.


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
                if not len(freq_table[k].keys()) == 1:
                    values = training_set[:,k]
                    # possible values for this column
                    possible_values = freq_table[k].keys()
                    for possible_value in possible_values:
                        freq_table[k][possible_value] = values.tolist().count(possible_value)
                else:
                    continue
            # check condition
            for k in range(details_map[system][0]):
                if not len(freq_table[k].keys()) == 1:
                    possible_values = freq_table[k].keys()
                    sum_value = sum([freq_table[k][possible_value] for possible_value in possible_values])
                    assert(sum_value == training_set.shape[0]), "Something is wrong"


            def stopping_condition(passed_freq_table):
                for k in range(details_map[system][0]):
                    if not len(passed_freq_table[k].keys()) == 1:
                        possible_values = freq_table[k].keys()
                        for possible_value in possible_values:
                            if freq_table[k][possible_value] < threshold:
                                return False
                return True

            '''
            We are done if the frequency table values hits the threshold
            '''
            if stopping_condition(freq_table) is True or i >= (details_map[system][1]/2)-1:
                break
            i += 1

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
    print
    print('Detailed learning projections:')
    print('<curve-id> : {<details>}')
    print()
    for keys in correlation_data:
        print(str(keys) +":"+str(correlation_data[keys]))
    print("-----------------------------------------------")
 
def main():           
    if system=='all':
        for i in all_systems:
            func = getattr(thismodule, strategy)
            func(i)
    else:
        func = getattr(thismodule, strategy)
        func(system)

main()        