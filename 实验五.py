# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 14:31:57 2022

@author: HZAUerhanshen
"""

import dm2022exp
import pandas as pd
import time
t1=time.time()
data=dm2022exp.load_ex5_data()
t2=time.time()
print(t2-t1)
class Apriori():
    def __init__(self,min_supp):
        self.min_supp=min_supp
    
    def count_itemset(self,itemsets,X):
        count_item={}
        t1=time.time()
        itemsets=list(map(frozenset,itemsets))
        for item_set in itemsets:
            for row in X:
                if item_set.issubset(row): 
                    if item_set in count_item.keys():
                        count_item[item_set]+=1
                    else:
                        count_item[item_set]=1
        t2=time.time()
        print(t2-t1)
        data=pd.DataFrame()
        data['item_sets']=list(map(list,count_item.keys()))
        data['supp']=count_item.values()
        return data
    def join(self,list_of_items):
        itemsets=[]
        i=1
        for entry in list_of_items:
            proceding_items=list_of_items[i:]
            for item in proceding_items:
                if(type(item) is str):
                    if entry!=item:
                        tuples=(entry,item)
                        itemsets.append(tuples)
                else:#前K项相等
                    if list(entry[0:-1])==list(item[0:-1]):
                        tuples=entry+item[1:]
                        itemsets.append(tuples)
            i=i+1
        if(len(itemsets) == 0):
            return None
        return itemsets
    def count_item(self,trans_items):
        count_ind_item={}
        for row in trans_items:
            for i in range(len(row)):
                if row[i] in count_ind_item.keys():
                    count_ind_item[row[i]] += 1
                else:
                    count_ind_item[row[i]] = 1
        data=pd.DataFrame()
        data['item_sets']=count_ind_item.keys()
        data['supp']=count_ind_item.values()
        data = data.sort_values('supp')
        return data
    def prune(self,data,min_supp):
        df=data[data.supp>=min_supp*self.length] 
        return df
    def fit(self,X):
        self.length=len(X)
        freq_items=self.count_item(X)
        X=list(map(set,X))
        result=[]
        while(len(freq_items)!=0):
            freq_items=self.prune(freq_items,self.min_supp)
            for i,j in zip(list(freq_items.item_sets),list(freq_items.supp)):
                if type(i) is list:
                    result.append((frozenset(set(i)),float(j)/self.length))
                else:
                    result.append((frozenset({i}),float(j)/self.length))
            new_items=self.join(freq_items.item_sets)
            if new_items is None:
                return result
            freq_items=self.count_itemset(new_items,X)
apriori=Apriori(0.02)
import time
t1=time.time()
print(apriori.fit(data))
t2=time.time()
print(t2-t1)
