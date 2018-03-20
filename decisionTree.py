from time import time
# from train_test_dataset import preprocess
import pickle
from sentiment_analysis import create_feature_set_and_labels
from tfidf_vectorizer import get_features
### Change here
features_train, labels_train, features_test, labels_test = get_features()
#features_train, labels_train, features_test, labels_test = create_feature_set_and_labels('pos_final.txt','neg_final.txt')
###
#features_train = np.array(features_train)
# features_test = np
# Decision Trees
from sklearn.tree import DecisionTreeClassifier as DTC

clf = DTC()

####time to train
t0 = time()

# train
clf.fit(features_train, labels_train)

print("training time:", round(time() - t0, 3), "s")

###time to predict
t1 = time()

# predict
pred = clf.predict(features_test)

print("testing time:", round(time() - t1, 3), "s")

# accuracy
from sklearn.metrics import accuracy_score
print(labels_test)
print(pred)
accuracy = accuracy_score(labels_test, pred)
print("accuracy: ", accuracy)
