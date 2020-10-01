import sys
import numpy as np
import csv
import math

if __name__ == '__main__':
    infile=sys.argv[-2]
    outfile=sys.argv[-1]
    datain=open(infile,'r')
    data_csv=csv.reader(datain,delimiter=',')
    dataout=open(outfile,'w')

    data = list()
    for i in data_csv:
        data.append(i)

    data = np.asarray(data)

# data = np.genfromtxt( datain, dtype='str', delimiter=',')

data = data[:,-1]
# print(data)
labels = dict()
total = len(data)

for i in range(1,total):
    label = data[i]
    labels[label] = labels.get(label, 0) + 1

x = labels.keys()

entropy=0
error=0

for i in x:
    # print(i,"=",labels[i])
    Total=total-1
    # print('Total',"=",Total)
    probability=(labels[i]/Total)
    # print('probability',"=",probability)
    logvalue=math.log2(probability)
    # print('logvalue',"=",logvalue)
    entropy+=probability*logvalue*-1
    # Minority=min(labels.keys(), key=(lambda k: labels[k]))
    minority=labels[min(labels, key=labels.get)]
    # print('minority=',minority)
    error=minority/Total

# print('entropy:',entropy)
# print('error:',error)

ans1='entropy: '+str(entropy)
ans2='error: '+str(error)

dataout.write(ans1+'\n')
dataout.write(ans2+'\n')
