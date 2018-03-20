#logistic regrssiom, naive bayes, svm
from dbn_outside.dbn.tensorflow import SupervisedDBNClassification
from sentiment_analysis import create_feature_set_and_labels
import numpy as np
from statistics import mean
import pickle
from sklearn.metrics.classification import accuracy_score
from tfidf_vectorizer import get_features
from sklearn.metrics import f1_score

# train_x,train_y,test_x,test_y = pickle.load(open("attr.pickle","rb"))
# train_x = np.array(list(train_x))
train_x, train_y, test_x, test_y = get_features('dbn')
#train_x, train_y, test_x, test_y = create_feature_set_and_labels('pos_final.txt', 'neg_final.txt')
#print(type(train_x))
print(len(train_x),len(train_y),len(test_x),len(test_y))
train_x = np.array(train_x,dtype=np.float32)
#print(type(train_x))
train_y = np.array(train_y,dtype=np.int32)
test_x = np.array(test_x,dtype=np.float32)
test_y = np.array(test_y,dtype=np.int32)
print(type(train_x))
classifier = SupervisedDBNClassification(hidden_layers_structure=[256,256,256],
                                         learning_rate_rbm=0.05,
                                         learning_rate=0.1,
                                         n_epochs_rbm=10,
                                         n_iter_backprop=100,
                                         batch_size=32,
                                         activation_function='relu',
                                         dropout_p=0.2)
classifier.fit(train_x, train_y)
# classifier = SupervisedDBNClassification.load('model.pkl')
# classifier.save('model.pkl')
accuracies = []
f_measures = []
for i in range(1):
    y_pred = classifier.predict(test_x)
    accuracy = accuracy_score(test_y, y_pred)
    f_measure = f1_score(test_y, y_pred)
    accuracies.append(accuracy)
    f_measures.append(f_measure)
print(accuracies)
print('Accuracy ', mean(accuracies))
print('F-measure', mean(f_measures))
