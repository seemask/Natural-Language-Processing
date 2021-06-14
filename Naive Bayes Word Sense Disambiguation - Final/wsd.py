import sys

from bs4 import BeautifulSoup as bs
import xml.etree.ElementTree as ET
import numpy as np
import statistics
import spacy


def naivebayes_algo(uniq_vocab_test,sense_vocab,count_senses,count_senses_prob,uniq_vocab):
    folds_dict = dict()
    for ins_id in  uniq_vocab_test:
        global predicted_dict
        score_max=-999999
        for sense in sense_vocab:
            totalprob=0
            score=0
            for keys in uniq_vocab_test[ins_id]:
                prob=np.log2((sense_vocab[sense].count(keys) + 1) / (
                            count_senses[sense] + len(uniq_vocab[sense])))
                totalprob = totalprob + prob
            score=totalprob+np.log2(count_senses_prob[sense])
            if(score_max<score):
                score_max=score
                final_sense=sense
        if(ins_id not in folds_dict):
            folds_dict[ins_id]=final_sense

    if predicted_dict is not None:
        predicted_dict.update(folds_dict)
    else:
        predicted_dict.copy()
    return folds_dict






predicted_dict=dict()
pred_correct = 0
total_count = 0
instance_count=0
accuracy_list=[]
list_contents=[]
sense_lists=[]
sense_context_mapping={}
count_senses={}
count_senses_prob = {}
sense_vocab=dict()
test_results_acc=dict()
sense_vocab_test= dict()
count_living=0
count_factory=0
sense_list_distinct = []


sp = spacy.load('en_core_web_sm')
all_stopwords = sp.Defaults.stop_words

filepath=sys.argv[1]#"plant.wsd"
print("Predicting Sense for ",filepath)

f = open(filepath)
file = f.read()


soup = bs(file,features='lxml')

instance_list=soup.find_all('instance')
answers=soup.find_all('answer')

for i in instance_list:
    list_contents.append(str(i))


instance_count=len(list_contents)
fold_len=int(instance_count/5)

outputfile_name=filepath+".out"
output=open(outputfile_name,"w")

for i in range(5):
    if(i==0):
        test_fold=list_contents[:fold_len]
        train_fold=list_contents[fold_len+1:]
    else:
        test_fold = list_contents[fold_len*(i):fold_len * (i + 1)]
        train_fold = list_contents[0:fold_len * (i)]+list_contents[fold_len * (i+1):]


    train_counts=len(train_fold)
    for tr_obj in train_fold:
        attr = ET.fromstring(tr_obj)
        distinct_sense_id = (attr[0].attrib)["senseid"]

        if (distinct_sense_id not in sense_list_distinct):
            sense_list_distinct.append(distinct_sense_id)

    for train_object in train_fold:

            attr = ET.fromstring(train_object)

            s_id = (attr[0].attrib)["senseid"]
            #print(attr[0].attrib)
            ctext = attr[1].text.strip()
            doc = sp(ctext)
            tokens = [token.text for token in doc]

            for tok in tokens:
                sense_vocab.setdefault(s_id, []).append(tok)

            if(s_id==sense_list_distinct[0]):
                count_living+=1
            else:
                count_factory+=1

            sense_context_mapping[s_id] = sense_context_mapping.get(s_id, []) + [ctext]


    count_senses[sense_list_distinct[0]]=count_living
    count_senses[sense_list_distinct[1]] = count_factory

    for sense in count_senses:
        count_senses_prob[sense]=count_senses[sense]/len(train_fold)

    uniq_vocab = {sen: list(set(word)) for sen, word in sense_vocab.items()}
    {k.lower(): [i.lower() for i in v] for k, v in uniq_vocab.items()}

    for test_object in test_fold:
        root = ET.fromstring(test_object)
        instance_id_test = (root[0].attrib)["instance"]

        ctext_test = root[1].text.strip()

        doc_test = sp(ctext_test)
        tokens_test = [token.text for token in doc_test]

        for tok_test in tokens_test:
            sense_vocab_test.setdefault(instance_id_test, []).append(tok_test)
    uniq_vocab_test = {sen: list(set(word)) for sen, word in sense_vocab_test.items()}
    {k.lower(): [i.lower() for i in v] for k, v in uniq_vocab_test.items()}
    results=naivebayes_algo(uniq_vocab_test,sense_vocab,count_senses,count_senses_prob,uniq_vocab)

    for acc_object in test_fold:
        root = ET.fromstring(acc_object)  # xml elmentary,
        instance_id_accuracy = (root[0].attrib)["instance"]
        sense_id_accuracy = (root[0].attrib)["senseid"]
        test_results_acc.setdefault(instance_id_accuracy,sense_id_accuracy)

    output.write("Fold " + str(i + 1)+"\n")

    for ins_id in results:
        if results[ins_id]==test_results_acc[ins_id]:
            pred_correct+=1
            total_count+=1
        else:
            total_count += 1
            print(ins_id,results[ins_id])
            print(ins_id,test_results_acc[ins_id])
        output.write(ins_id+" "+results[ins_id]+"\n")



    accuracy_list.append((pred_correct/total_count)*100)

    print("Accuracy of Fold "+str(i+1)+" :",np.round(accuracy_list[i],2))
print("Accuracy mean of all 5 folds:",np.round(statistics.mean(accuracy_list),2))






