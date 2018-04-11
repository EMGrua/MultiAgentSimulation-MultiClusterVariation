import pandas as pd
from scipy.stats import wilcoxon as w


dfp2 = pd.read_csv('reward_p2.csv')
#saved_column = df.column_name #you can also use df['column_name']
#names = df.Names

#print(df.iloc[:,0]) #to get the first column

#kdtwn i = 2
#heuad i = 5
#separate i = 7

keunp2 = dfp2.iloc[:,0].values
heunp2 = dfp2.iloc[:,1].values
kdtwnp2 = dfp2.iloc[:,2].values #top3
hdtwnp2 = dfp2.iloc[:,3].values
keuadp2 = dfp2.iloc[:,4].values
heuadp2 = dfp2.iloc[:,5].values #top3
pooledp2 = dfp2.iloc[:,6].values
separatep2 = dfp2.iloc[:,7].values #top3


test1 = w(separatep2, pooledp2)
test2 = w(kdtwnp2, pooledp2)
test3 = w(heuadp2, pooledp2)
print('separatep2 vs pooledp2', test1)
print('kdtwnp2 vs pooledp2', test2)
print('heuadp2 vs pooledp2', test3)


df = pd.read_csv('reward.csv')
keun = df.iloc[:,0].values
heun = df.iloc[:,1].values
kdtwn = df.iloc[:,2].values
hdtwn = df.iloc[:,3].values #top3
keuad = df.iloc[:,4].values #top3
heuad = df.iloc[:,5].values
pooled = df.iloc[:,6].values
separate = df.iloc[:,7].values #top3

testx = w(keunp2, pooledp2)
#print(testx)


test4 = w(separate, pooled)
test5 = w(hdtwn, pooled)
test6 = w(keuad, pooled)
print('separate vs pooled', test4)
print('hdtwn vs pooled', test5)
print('keuad vs pooled', test6)
