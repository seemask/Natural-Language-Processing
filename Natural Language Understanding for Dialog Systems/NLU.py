# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 20:12:22 2020

@author: Seema S Kanaje
"""

from string import punctuation
import re
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from sklearn.preprocessing import LabelEncoder
import spacy
import random


sp = spacy.load('en_core_web_sm')
all_stopwords = sp.Defaults.stop_words

bin_conv = []


def stringtobinary(word):
    for char in word:
        ascii_val = ord(char)
        binary_val = bin(ascii_val)
        bin_conv.append(binary_val[2:])
    return (''.join(bin_conv))


def parse_data(filepath):
    f = open(filepath)
    file = f.read().split('\n\n')
    new_lst = []
    list_class = []
    for i in file:
        
        k = i.split('\n')
        try:
            # print("k value 0",k[0])
            idvalue = None
            namevalue = None
            for i in range(1, len(k)):
                # flag=0
                if (k[i].split('=')[0] == 'id'):
                    idvalue = k[i].split('=')[1]
                if (k[i].split('=')[0] == 'name'):
                    namevalue = k[i].split('=')[1]
            if (idvalue is None and namevalue is None):
                continue;
            elif (idvalue is None):
                list_class.append([k[0], '', namevalue])
            elif (namevalue is None):
                list_class.append([k[0], idvalue, ''])
            else:
                list_class.append([k[0], idvalue, namevalue])


        except:
            pass
    # print(list_class)
    # print(i.split('\n')[2])
    return list_class


def IOB_tagging(tokens, id, name):
    tok_label = []
    for tok in tokens:
        # print('tok value is: '+tok+' ID value is:'+id+' name: '+name)
        # print('Name value is ',name)
        tok = tok.strip(punctuation)
        # print("After stripping punc",tok)

        try:
            if (tok == id):
                label = 'B'
            elif (tok in name) and (tok == name.split()[0]):
                label = 'B'
            elif (tok in name) and (tok != name.split()[0]):
                label = 'I'
            else:
                label = 'O'
        except:
            if (id == ''):
                if (tok in name) and (tok == name.split()[0]):
                    label = 'B'
                elif (tok in name) and (tok != name.split()[0]):
                    label = 'I'
                else:
                    label = 'O'
            elif (name == ''):
                if (tok == id):
                    label = 'B'
                else:
                    label = 'O'

        # print("Label is",label)
        tok_label.append([tok, label])
    return tok_label


def labelencoder(trained_word_tok, test_word_tok):
    uniquewordset = ()
    uniquewordlist = []
    uniquewordlist_new = []
    for wordtag in trained_word_tok:
        for word in wordtag:
            # print(word)
            if word[0] not in uniquewordlist:
                uniquewordlist.append(word[0])
    for wordtagtest in test_word_tok:
        for wordtest in wordtagtest:
            if wordtest[0] not in uniquewordlist:
                uniquewordlist.append(word[0])

    uniquewordset = set(uniquewordlist)
    uniquewordlist_new = list(uniquewordset)

    le = LabelEncoder()
    le.fit(uniquewordlist_new)
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    # print(le_name_mapping['thrilling'])
    return le_name_mapping


def featureextraction(trained_word_tok, codedlabels):
    target = []
    testing = []
    featurevectors = []
    targetword = []
    count = 0
    uniquewordlist = []

    for i in trained_word_tok:
        for j in i:
            if (len(j[0]) > 2):
                # print(j)
                # print(j[0])
                # print(j[1])

                count += 1
                target.append(j[1])
                binvalue = stringtobinary(j[0])
                # print(word)
                if (str(j[0]).isupper()):
                    iswordupper = 1
                else:
                    iswordupper = 0
                # print("I am in error:(",j[0])
                if (str(j[0][0]).isupper()):
                    iswordcap = 1
                else:
                    iswordcap = 0
                if ((j[0]).isdigit()):
                    isworddigit = 1
                else:
                    isworddigit = 0
                if (str(j[0]).islower()):
                    iswordlower = 1
                else:
                    iswordlower = 0
                if (str(j[0]).islower()):
                    iswordnotcap = 1
                else:
                    iswordnotcap = 0
                if re.match("^[a-zA-Z0-9_]*$", str(j[0])):
                    wordhaspunc = 0
                else:
                    wordhaspunc = 1
                wordlend = len(str(j[0]))
                # print(codedlabels['thrilling'])
                vecfeature = []
                try:
                    vecfeature.append(codedlabels[j[0]])
                except:
                    vecfeature.append(random.randint(3000,30000))
                vecfeature.append(iswordupper)
                vecfeature.append(iswordcap)
                vecfeature.append(isworddigit)
                vecfeature.append(iswordlower)
                vecfeature.append(iswordnotcap)
                vecfeature.append(wordhaspunc)
                vecfeature.append(wordlend)
                featurevectors.append(vecfeature)
    # print(featurevectors)
    return (featurevectors, target)


filepath_train = sys.argv[1]#"D:\Fall2020\NLP\Exam\NLU.train"
filepath_test = sys.argv[2]#r"D:\Fall2020\NLP\Exam\NLU.test"

parsed_data_train = parse_data(filepath_train)
parsed_data_test = parse_data(filepath_test)

trained_word_tok = []
test_word_tok = []

for line in parsed_data_train:
    
    tokens = line[0].split()

    id = line[1]
    name = line[2]
    
    trained_word_tok.append(IOB_tagging(tokens, id, name))

for line_test in parsed_data_test:
    testtokens = line_test[0].split()
    testid = line_test[1]
    testname = line_test[2]
    test_word_tok.append(IOB_tagging(testtokens, testid, testname))
    
codedlabels = labelencoder(trained_word_tok, test_word_tok)

train_featurevectors, train_target = featureextraction(trained_word_tok, codedlabels)
test_featurevectors, test_target = featureextraction(test_word_tok, codedlabels)


classifier = tree.DecisionTreeClassifier()

print('Decision Tree classifier')
classifier.fit(train_featurevectors, train_target)

print('Prediciton of test values')
pred = classifier.predict(test_featurevectors)
print(confusion_matrix(test_target, pred))
print(classification_report(test_target, pred))

NLU_test_out = open("NLU.test.out", "w")
for i, row_data in enumerate(test_featurevectors):
    actual_values = test_target[i]
    predvalues = classifier.predict(np.array(row_data).reshape(1, -1))
    line = str('  ' + 'Predicted:' + predvalues[0] + ' ' + 'Actual:' + actual_values)
    NLU_test_out.write(line)
    NLU_test_out.write("\n")
NLU_test_out.close()












