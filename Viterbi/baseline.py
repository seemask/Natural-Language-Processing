import nltk
from nltk.util import ngrams
import operator

def file_parser(filename):
    word_tag_list = []
    prev_present_tag_list = []
    tag_count = {}
    word_count = {}
    bigram_freq = {}
    with open(filename, 'r') as file:
        train_data = file.read().replace('\n', '')
    for word_tag in train_data.split():
        word, char, tag = word_tag.partition('/')
        word_tag_list.append((word, tag))

    for word, tag in word_tag_list:
        if word not in word_count:
            word_count[word] = {tag: 1}
        else:
            if tag not in word_count[word]:
                word_count[word][tag] = 1
            else:
                word_count[word][tag] += 1
        if tag not in tag_count:
            tag_count[tag] = 1
        else:
            tag_count[tag] += 1
    for word_tag in train_data.split():
        word, char, tag = word_tag.partition('/')
        prev_present_tag_list.append((tag))
    bgs = list(ngrams(prev_present_tag_list, 2))

    for bigram in bgs:
        if bigram not in bigram_freq:
            bigram_freq[bigram] = 1
            bigram_freq[bigram] += 1

    return word_count,tag_count,bigram_freq



def strip_test_file():
    filename =sys.argv[2] #r'D:\Fall2020\NLP\Assignment 2\POS.test'
    test_tag_list = []
    test_word_list = []
    test_sentence_list = []
    with open(filename, 'r') as file:
        for sentence in file.readlines():
            for test_word_tag in sentence.split():
                test_word, test_char, test_tag = test_word_tag.partition('/')
                test_tag_list.append(test_tag)
                test_word_list.append((test_word))
                # print(test_word)
            test_sentence_list.append(test_word_list)

    return test_word_list, test_tag_list, test_sentence_list







def baseline(tag_count,word_count,test_word_list):
    pred_data = []
    pred_tag = []
    keyMax = max(tag_count.items(), key=operator.itemgetter(1))[0]
    for i in test_word_list:
        if i in word_count:
            pred_data.append(i + '/' + max(word_count[i]))
            pred_tag.append(max(word_count[i]))
        else:
            pred_data.append(i + '/' + keyMax)
            pred_tag.append(keyMax)
    #print(pred_data)
    return pred_data, pred_tag


def accuracy(p_test_tag_list,pred_tag):
    correct_tags=0
    total_tags=0
    for i,tag in enumerate(pred_tag):
        if(tag==p_test_tag_list[i]):
            correct_tags+=1
            total_tags+=1
        else:
            total_tags+=1
    accuracy=correct_tags/total_tags
    return accuracy


file_name=sys.argv[1]#r'D:\Fall2020\NLP\Assignment 2\POS.train.large'
word_count,tag_count,bigram_freq=file_parser(file_name)
test_word_list,test_tag_list,test_sentence_list=strip_test_file()
train_tag_list=tag_count.keys()
train_tag_list_new=[]
for i in train_tag_list:
    if(len(i)<=3):
        train_tag_list_new.append(i)

predicted_data,pred_tag=baseline(tag_count,word_count,test_word_list)
baseline_accuracy=accuracy(test_tag_list,pred_tag)
print('Accuracy of Baseline Algorithm is:',baseline_accuracy*100)

