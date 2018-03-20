import pandas as pd
import codecs
from nltk.tokenize import word_tokenize
from sklearn.metrics.classification import accuracy_score
from sklearn.metrics import f1_score
import re

data = pd.read_csv("hindi.txt", delimiter=' ')

fields = ['POS_TAG', 'ID', 'POS', 'NEG', 'LIST_OF_WORDS']

words_dict = {}
for i in data.index:
    # print (data[fields[0]][i], data[fields[1]][i], data[fields[2]][i], data[fields[3]][i], data[fields[4]][i])

    words = data[fields[4]][i].split(',')
    for word in words:
        words_dict[word] = (data[fields[0]][i], data[fields[2]][i], data[fields[3]][i])


# print(len(words_dict))
# for key in words_dict:
#    print(key,words_dict[key])

def sentiment(text):
    words = word_tokenize(text)
    votes = []
    pos_polarity = 0
    neg_polarity = 0
    allowed_words = ['a','v','r','n']
    for word in words:
        if word in words_dict:
            pos_tag, pos, neg = words_dict[word]
            # print(word, pos_tag, pos, neg)
            if pos_tag in allowed_words:
                if pos > neg:
                    pos_polarity += pos
                    votes.append(1)
                elif neg > pos:
                    neg_polarity += neg
                    votes.append(0)
    pos_votes = votes.count(1)
    neg_votes = votes.count(0)
    if pos_votes > neg_votes:
        return 1
    elif neg_votes > pos_votes:
        return 0
    else:
        if pos_polarity < neg_polarity:
            return 0
        else:
            return 1


pred_y = []
actual_y = []
pos_reviews = codecs.open("pos_final.txt", "r", encoding='utf-8', errors='ignore').read()
for line in pos_reviews.split('$'):
    data = line.strip('\n')
    if data:
        pred_y.append(sentiment(data))
        actual_y.append(1)
#print(accuracy_score(actual_y, pred_y) * 100)
print(len(actual_y))
neg_reviews = codecs.open("neg_final.txt", "r", encoding='utf-8', errors='ignore').read()
for line in neg_reviews.split('$'):
    data=line.strip('\n')
    if data:
        pred_y.append(sentiment(data))
        actual_y.append(0)
print(len(actual_y))
print(accuracy_score(actual_y, pred_y) * 100)
print('F-measure:  ',f1_score(actual_y,pred_y))
# if __name__ == '__main__':
    #print(sentiment("मैं इस उत्पाद से बहुत खुश हूँ  यह आराम दायक और सुन्दर है  यह खरीदने लायक है "))
