from sklearn.svm import SVC  # "Support Vector Classifier"
import numpy as np
from Evaluation import evaluation


def Model_SVM(train_data,train_target,test_data,test_target, m):
    clf = SVC(kernel='rbf')
    # fitting x samples and y classes
    predict = np.zeros((test_target.shape)).astype(('int'))
    for i in range(train_target.shape[1]):
        clf.fit(train_data.tolist(), train_target[:, i].tolist())
        predict[:, i] = clf.predict(test_data.tolist())

    Eval = evaluation(predict, test_target)
    return np.asarray(Eval).ravel(),predict