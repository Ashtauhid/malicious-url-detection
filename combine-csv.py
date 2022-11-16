import pandas as pd

fout=open("data/mergedData.csv","a")
# first file:
for line in open("data/data1.csv"):
    fout.write(line)
# now the rest:
for num in range(2,3):
    f = open("data/data"+str(num)+".csv")
    f.__next__() # skip the header
    for line in f:
         fout.write(line)
    f.close() # not really needed
fout.close()