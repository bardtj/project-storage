import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from Engine import ForesightEngine

# Define
TEST_CNT = 1 
threshold_list = np.linspace(0, 1, 101)

# GET Data 
data = pd.read_csv('Data/df_final.csv')
engine = ForesightEngine(data)

# Get Score
def get_score(y_label, y_pred):
    cnf_matrix = confusion_matrix(y_label, y_pred, labels=[1, 0])
    score = cnf_matrix[0][0] * 2 + cnf_matrix[1][1] * 1 - (cnf_matrix[0][1] + cnf_matrix[1][0])
    return score

# Get Train 
for test_case in range(TEST_CNT):
    max_rf_score = 0
    max_svc_score = 0
    max_rf_threshold = 0
    max_svc_threshold = 0
    
    train_data = engine.train[test_case]
    y = train_data['너울성파도발생여부']
    x = train_data.drop(['너울성파도발생여부', '기상불량발생여부'], axis=1)
    
    x_train = x[:-100]
    y_train = y[:-100]
    print("test_case : {} [{}]".format(test_case, len(x_train)))
    
    #TODO 
    x_val = x[-100:]
    y_val = y[-100:]
    print(train_data.columns)    
    
    # RF
    clf = RandomForestClassifier(n_estimators=10)
    clf.fit(x_train, y_train)
    y_prob = clf.predict_proba(x_val)[:,1]
    
    for threshold in threshold_list:
        y_th_pred = y_prob > threshold
        rf_score = get_score(y_val, y_th_pred)
        if (rf_score > max_rf_score): 
            max_rf_score = rf_score
            max_rf_threshold = threshold
            print("Score [{}]  Threshold {}".format(max_rf_score, max_rf_threshold))

    # SVC
    for c in [0.1, 1, 5, 10, 15]:
        for g in [0.1, 1, 5]:

            svc = SVC(C=c, gamma=g, class_weight='balanced', probability=True) # 10, 1
            svc.fit(x_train, y_train)
            y_prob = svc.predict_proba(x_val)[:,1]
            
            for threshold in threshold_list:
                y_th_pred = y_prob > threshold
                svc_score = get_score(y_val, y_th_pred)
                if(svc_score > max_svc_score) : 
                    max_svc_score = svc_score
                    max_svc_threshold = threshold
                    print("Score [{}]  C: {} gamma: {} Threshold {}".format(svc_score, c, g, threshold))
            
            

#cnf_matrix = confusion_matrix(y_val, y_pred, labels=[1, 0])
#print(cnf_matrix)
#print(classification_report(y_val, y_pred, labels=[1, 0]))
 
# Prediction 
#test_data = engine.train[test_case]