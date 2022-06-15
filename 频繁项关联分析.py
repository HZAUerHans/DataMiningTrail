# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 14:31:57 2022

@author: HZAUerhanshen
"""

import dm2022exp
import pandas as pd
from sklearn.preprocessing import LabelEncoder
data=dm2022exp.load_ex5_data()

class Apriori():
    def __init__(self,min_supp):
        self.min_supp=min_supp
        self.le=LabelEncoder()
    def count_itemset(self,itemsets,X):
        count_item={}
        t1=time.time()
        for item_set in itemsets:
            set_A=set(item_set)
            for row in X:
                set_B=set(row)
                if set_B.intersection(set_A)==set_A:
                    if item_set in count_item.keys():
                        count_item[item_set]+=1
                    else:
                        count_item[item_set]=1     
        t2=time.time()
        #print(t2-t1)
        data=pd.DataFrame()
        data['item_sets']=count_item.keys()
        data['supp']=count_item.values()
        
        return data
    def join(self,list_of_items):
        #print(list_of_items)
        itemsets=[]
        i=1
        for entry in list_of_items:
            proceding_items=list_of_items[i:]
            for item in proceding_items:
                if(type(item) is str):
                    if entry!=item:
                        tuples=(entry,item)
                        itemsets.append(tuples)
                else:#前K-1项相等
                    if entry[0:-1]==item[0:-1]:
                        tuples=entry+item[1:]
                        itemsets.append(tuples)
            i=i+1
        if(len(itemsets) == 0):
            return None
        #return
        return itemsets
    def count_item(self,trans_items):
        count_ind_item={}
        for row in trans_items:
            for i in range(len(row)):
                if row[i] in count_ind_item.keys():
                    count_ind_item[row[i]] += 1
                else:
                    #print(row[i])
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
        freq=pd.DataFrame()
        freq_items=self.count_item(X)
        result={}
        while(len(freq_items)!=0):
            freq_items=self.prune(freq_items,self.min_supp)
            for i,j in zip(list(freq_items.item_sets),list(freq_items.supp)):
                result[i]=float(j)/self.length
            new_items=self.join(freq_items.item_sets)
            if new_items is None:
                return result
            freq_items=self.count_itemset(new_items,X)
    def strong_rules(self,freq_item_sets, threshold):
        #print(freq_item_sets)
        confidences={}
        for itemset in freq_item_sets:
            if type(itemset) is str:#如果频繁项集只有一项，跳过
                continue
            i=1
            for item in itemset:
                if len(itemset[:i])>1 and float(freq_item_sets[itemset])/freq_item_sets[tuple(itemset[:i])]>threshold:
                    
                    print("关联规则:",itemset[:i],"--->",itemset[i:])
                    print("置信度:",float(freq_item_sets[itemset])/freq_item_sets[tuple(itemset[:i])],'\n')
                else:
                    if float(freq_item_sets[itemset])/freq_item_sets[itemset[:i][0]]>threshold:
                        print("关联规则:",itemset[:i],"--->",itemset[i:])
                        print("置信度:",float(freq_item_sets[itemset])/freq_item_sets[itemset[:i][0]],'\n')
                        
                if len(itemset[i:])>1 and float(freq_item_sets[itemset])/freq_item_sets[tuple(itemset[i:])]>threshold:
                    print("关联规则:",itemset[i:],"--->",itemset[:i])
                    print("置信度:",float(freq_item_sets[itemset])/freq_item_sets[tuple(itemset[i:])],'\n')
                else:
                    if float(freq_item_sets[itemset])/freq_item_sets[itemset[i:][0]]>threshold:
                        print("关联规则:",itemset[i:],"--->",itemset[:i])
                        print("置信度:",float(freq_item_sets[itemset])/freq_item_sets[itemset[i:][0]],'\n')
                i+=1
                if i==len(itemset):
                    break
apriori=Apriori(0.02)
import time
t1=time.time()
result=apriori.fit(data)
print(result)
print(apriori.strong_rules(result,0.4))
t2=time.time()
print(t2-t1)
