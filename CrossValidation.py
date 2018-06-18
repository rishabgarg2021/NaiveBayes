import Final as tt
import numpy as np

from sklearn.model_selection import train_test_split

def cross_validate (n):
    accuracy_list = []
    df=tt.preProcess()
    for i in range (n):
        X_train, X_test = train_test_split(df, test_size=0.1)
        dict,temp,classDict = tt.train_supervised(X_train)
        accuracy = tt.predict_supervised(dict, X_test, classDict)
        accuracy_list.append(accuracy)
    mean_accuracy = np.mean(accuracy_list)
    print ("mean accuracy is: ", mean_accuracy)
    return np.mean(accuracy_list)


cross_validate(10)
