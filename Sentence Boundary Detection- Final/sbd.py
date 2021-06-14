

# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 18:57:14 2020

@author: Seema S Kanaje
"""

from __future__ import division, print_function
from sklearn import tree
import numpy as np 
import sys
import os
from sklearn.metrics import classification_report, confusion_matrix


def dataparser(filename,vocab_list=False):


    filehandle = open(filename, 'r')
    a=[]
    while not is_eof(filehandle):
        textline =filehandle.readline().split('\n')
        word=textline[0].split()[2]
        if(word!='TOK'):
            if vocab_list:
                a.append(textline[0].split()[1])
            else:
                a.append(textline[0].split())
    filehandle.close()  
    return a 
	
def is_eof(f):
    current = f.tell()    # save current position
    f.seek(0, os.SEEK_END)
    end = f.tell()    # find the size of file
    f.seek(current, os.SEEK_SET)
    return current == end
def uniqvocablist():
    trainpath= sys.argv[1]#r"D:\Fall2020\NLP\Assignment 1\SBD.train"
    testpath=sys.argv[2]#r"D:\Fall2020\NLP\Assignment 1\SBD.test"
    traindata=dataparser(trainpath,True)
    testdata=dataparser(testpath,True)
    vocablist=traindata+testdata
    uniqvocablist=list(set(vocablist))
    return uniqvocablist
	
	
def generatefeature(data):
    word_to_right=[]
    word_to_left=[]
    featurevectors=[]
    target=[]
    lwisshort=[]
    lwiscap=[]
    rwiscap=[]
    lwisalphanum=[]
    rwisalphanum=[]
    lwisdigit=[]
    uniqvocab= uniqvocablist()
    for num,dataline in enumerate(data):
        uniq_line_no=dataline[0]
        dataword=dataline[1].strip()
        targettok=dataline[2]
        target.append(targettok)
        
        
        splitword=dataword.split('.')
        wordleft=splitword[0]
        wordright=splitword[1]
        word_to_left.append(1 if wordleft else 0)
        word_to_right.append(1    if wordright  else 0)
        lwisshort.append(1   if len(wordleft) < 3   else 0)
        lwiscap.append(1  if wordleft.isupper()  else 0)
        rwiscap.append(1 if wordright.isupper() else 0)
        rwisalphanum.append(1 if wordright.isalnum() else 0)
        lwisalphanum.append(1 if wordleft.isdigit() else 0)
        lwisdigit.append(1 if wordleft.isalnum() else 0)
        
        
        vector = [0] * len(uniqvocab)
        
        vocabpos=uniqvocab.index(dataword)
        vector[vocabpos] = 1
        
        vecfeature=vector
        vecfeature.append(word_to_left[num]) 
        vecfeature.append(word_to_right[num])
        vecfeature.append(lwisshort[num])
        vecfeature.append(lwiscap[num])
        vecfeature.append(rwiscap[num])
        vecfeature.append(rwisalphanum[num])
        vecfeature.append(lwisalphanum[num])
        vecfeature.append(lwisdigit[num])
        
        featurevectors.append(vecfeature)
    return(featurevectors,target)

        
	
def main():
    trainpath= sys.argv[1]#r"D:\Fall2020\NLP\Assignment 1\SBD.train"
    testpath= sys.argv[2]#r"D:\Fall2020\NLP\Assignment 1\SBD.test"
    training_set     = dataparser(trainpath)
    test_set         = dataparser(testpath)
    training_features, train_targets = generatefeature(training_set)
    test_features, testtargets     = generatefeature(test_set)
    
    classifier = tree.DecisionTreeClassifier()

    
    print('Decision Tree classifier')
    classifier.fit(training_features, train_targets) 
    
    

    print('Prediciton of test values')
    pred = classifier.predict(test_features)
    print(confusion_matrix(testtargets, pred))
    print(classification_report(testtargets, pred))
    sbd_test_out=open("sbd.test.out","w")
    for i, row_data in enumerate(test_features):
        actual_values = testtargets[i]
        predvalues = classifier.predict(np.array(row_data).reshape(1,-1))
        line=str(test_set[i])+'  '+'Predicted:'+ predvalues[0]+' '+'Actual:'+actual_values
        sbd_test_out.write(line)
        sbd_test_out.write("\n")
    sbd_test_out.close() 

		
    


if __name__ == '__main__':
    main()	