import os
import pandas as pd

files = [f for f in os.listdir(".") if ".csv" in f]
for file in files:
    # content = pd.read_csv(file)
    # print "\"" + file + "\" : [" + str(len(content.columns)-1) + ", " + str(len(content)) + "],"
    print "\""+file+"\",",