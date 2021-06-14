import nltk
from nltk.util import ngrams

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

def transition_prob(tag_prev, tag_cur,bigram_freq,tag_count):
    try:
        count_tag_prev = tag_count[tag_prev]
        count_current_prev_tag = bigram_freq[(tag_prev, tag_cur)]
        trans_prob = count_current_prev_tag / count_tag_prev
    except:
        trans_prob = 1 / len(bigram_freq.keys())
    return trans_prob




def emission_prob(word, tag,word_count,tag_count):
    try:
        count_word_tag = word_count[word][tag]
        # print(count_word_tag)
        count_tag = tag_count[tag]
        # print(count_tag)
        prob = count_word_tag / count_tag
    except:
        prob = 1 / sum(tag_count.values())
    return prob


def strip_test_file():
    filename = sys.argv[2]#r'D:\Fall2020\NLP\Assignment 2\POS.test'
    test_tag_list = []

    test_sentence_list = []
    with open(filename, 'r') as file:
        for sentence in file.readlines():
            test_word_list = []
            for test_word_tag in sentence.split():
                test_word, test_char, test_tag = test_word_tag.partition('/')
                test_tag_list.append(test_tag)
                test_word_list.append((test_word))
                # print(test_word)
            test_sentence_list.append(test_word_list)

    return test_word_list, test_tag_list, test_sentence_list


def Viterbi(test_sentence_list,T,bigram_freq,tag_count,word_count):
    statetag= []
    for key, line in enumerate(test_sentence_list):
        prob = []
        for tag in T:
            if key == 0:
                trans_prob = transition_prob('.', tag,bigram_freq,tag_count)
            else:
                trans_prob = transition_prob(statetag[-1], tag,bigram_freq,tag_count)

                # compute emission and state probabilities
            emi_prob = emission_prob(line, tag,word_count,tag_count)
            probability = emi_prob * trans_prob
            prob.append(probability)

        probmax = max(prob)
        maxstate = T[prob.index(probmax)]
        statetag.append(maxstate)
    return list(zip(test_sentence_list, statetag))


def accuracy(list_pred,test_tag_list):
    correct = 0
    total = 0
    count = 0
    for i, pred in enumerate(list_pred):
        for j in range(len(pred)):
            if (list_pred[i][j][1] == test_tag_list[count]):
                correct += 1
                total += 1
            else:
                total += 1
            count = count + 1
    accuracy = correct / total
    return accuracy

list_pred=[]
file_name=sys.argv[1]#r'D:\Fall2020\NLP\Assignment 2\POS.train'
word_count,tag_count,bigram_freq=file_parser(file_name)
test_word_list,test_tag_list,test_sentence_list=strip_test_file()
T=list(tag_count.keys())
for i in test_sentence_list:
    list_pred.append(Viterbi(i,T,bigram_freq,tag_count,word_count))
sbd_test_out=open("POS.test.out","w")
for i in list_pred:
    for tup in i:
        #print(tup[0])
        sbd_test_out.write(str(tup[0]+'/'+tup[1]+' '))
    sbd_test_out.write("\n")
accuracy_viterbi=accuracy(list_pred,test_tag_list)
print('Accuracy of Viterbi Algorithm:', accuracy_viterbi*100)
