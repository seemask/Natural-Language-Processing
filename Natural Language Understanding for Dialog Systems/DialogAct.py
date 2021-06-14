import sys

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import numpy as np

filepath = sys.argv[1]#r"DialogAct.train"
filepath1 = sys.argv[2]#r"DialogAct.test"
#Parses Train data and calculates all the required counts wrt to labels
def parse_data_train(filepath):
    with open(filepath, encoding="utf8") as data:
        count = 0
        flag = -1
        label_flag = False
        newstatement = ''
        sense_vocab = dict()
        countofwords_label = dict()
        dis_word_count=dict()
        labelcount_list=[]
        for line in data.read().split('\n'):
            if (line.startswith('Advisor')):# Whenever line starts with Advisor, collects the label
                labelcount_temp = line.split(']')[0]
                labelcount = labelcount_temp.split('[')[1]
                labelcount_list.append(labelcount)




            if (count > 0):
                #Calculates for each statement what is the label for it along with counts
                if (line.startswith('Advisor')):
                    label_temp = line.split(']')[0]
                    label = label_temp.split('[')[1]
                    label_flag = True
                    tokens = word_tokenize(newstatement.lower())
                    for tok in tokens:
                        sense_vocab.setdefault(label, []).append(tok)

                if (line.startswith('Student')):
                    student = line.split(':')[1]

                    if (label_flag == False and count == 1):
                        newstatement = student

                    elif (label_flag):
                        newstatement = student
                        label_flag = False

                    else:
                        newstatement += student


            count += 1

        for i in sense_vocab.keys():

            countofwords_label.setdefault(i, []).append(Counter(sense_vocab[i]))
        #Calculates distinct count of words in each label
        for i in sense_vocab.keys():
            dis_word_count.setdefault(i, len(countofwords_label[i][0]))


        count_total_labels = len(labelcount_list)
        counter_labellist = Counter(labelcount_list)

    return sense_vocab.keys(),count_total_labels,counter_labellist,countofwords_label,dis_word_count
#Parsing test data and it return student line along with its label
def parse_test_data(filepath1):
    with open(filepath1,encoding="utf8") as data:
        testlabel_flag=False
        test_data=dict()
        test_Count=0


        for testline in data.read().split('\n'):

            if(test_Count>0):
                if (testline.startswith('Advisor')):
                    testlabel_temp = testline.split(']')[0]

                    testlabel = testlabel_temp.split('[')[1]
                    testlabel_flag = True
                    test_data.setdefault(testnewstatement, testlabel)

                if (testline.startswith('Student')):
                    teststudent = testline.split(':')[1]

                    if (testlabel_flag == False and test_Count == 1):
                        testnewstatement = teststudent

                    elif (testlabel_flag):
                        testnewstatement = teststudent
                        testlabel_flag = False

                    else:
                        testnewstatement += teststudent


            test_Count+=1
        return test_data
label_keys,count_total_labels,counter_labellist,countofwords_label,dis_word_count=parse_data_train(filepath)
test_data=parse_test_data(filepath1)

actual=0
total=0
DialogAct_test_out = open("DialogAct.test.out", "w")
#Implementing Naive Baye's Algorithm
for i in test_data:
    tokenizedword=word_tokenize(i.lower())

    score=0
    score_dict=dict()
    max_score = -9999999999999
    for label in label_keys:


        total_prob=0
        for toks in tokenizedword:

            num=countofwords_label[label][0][toks]+1
            deno=counter_labellist[label]+dis_word_count[label]
            prob_features_sense1 =np.log2( num / deno)
            total_prob += prob_features_sense1

        prob_label=counter_labellist[label]/count_total_labels
        score= total_prob+np.log2(prob_label)

        if(score>max_score):
            max_score=score
            final_label=label


    if(final_label==test_data[i]):
        actual+=1
        total+=1
    else:
        total+=1

    score_dict.setdefault(label, score)
    DialogAct_test_out.write("Student: "+i)
    DialogAct_test_out.write("\n")
    line = str('Predicted:' + final_label + ' ' + 'Actual:' + test_data[i])
    DialogAct_test_out.write(line)
    DialogAct_test_out.write("\n")


print(" Accuracy of Dialog Act is :")
print(np.round((actual/total),2)*100)
DialogAct_test_out.close()



