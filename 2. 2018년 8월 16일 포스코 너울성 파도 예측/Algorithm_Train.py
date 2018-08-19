import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from Engine import ForesightEngine
import pickle
import os

# DEFINE
TEST_CNT = 25
threshold_list = np.linspace(0, 1, 101)
MODEL_PATH = './model_last'
os.makedirs(MODEL_PATH)

# GET Data 
data = pd.read_csv('Data/df_last.csv')
engine = ForesightEngine(data)
th_df = pd.DataFrame(columns=[ 'TC', 'RF', 'SVC', 'RF_score', 'SVC_score' ])

# Get Score
def get_score(y_label, y_pred):
    cnf_matrix = confusion_matrix(y_label, y_pred, labels=[1, 0])
    score = cnf_matrix[0][0] * 2 + cnf_matrix[1][1] - cnf_matrix[0][1] * 2 - cnf_matrix[1][0]
    return score

# Get Train 
for test_case in range(TEST_CNT):
    max_rf_score = 0
    max_svc_score = 0
    max_rf_threshold = 0
    max_svc_threshold = 0
    
    x = engine.x_train[test_case]
    y = engine.y_train[test_case]
    
    data_size = len(x)
    sample = int(data_size * 0.8)
    
    x_train = x[:sample]
    y_train = y[:sample]    
    x_val = x[sample:]
    y_val = y[sample:]
    print("test_case[{}] Train {} Validation {}".format(test_case, len(x_train), len(x_val)))
    
    # RF
    print("[Train : {}] RandomForest".format(test_case))
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(x_train, y_train)
    y_prob = clf.predict_proba(x_val)[:,1]
    
    for threshold in threshold_list:
        y_th_pred = y_prob > threshold
        rf_score = get_score(y_val, y_th_pred)
        if (rf_score > max_rf_score): 
            max_rf_score = rf_score
            max_rf_threshold = threshold
    print("Score [{}]  Threshold {}".format(max_rf_score, max_rf_threshold))

    # Save TH
    th_df.loc[test_case, 'TC'] = test_case
    th_df.loc[test_case, 'RF'] = max_rf_threshold
    th_df.loc[test_case, 'RF_score'] = max_rf_score

    # RF Fitting
    clf.fit(x, y)
    
    # SVC
    print("[Train : {}] SVM".format(test_case))
    scaler = StandardScaler()
    x_svc = scaler.fit_transform(x)
    x_train_svc = scaler.transform(x_train)
    x_val_svc  = scaler.transform(x_val)
    
    svc = SVC(C=10, gamma=1, class_weight='balanced', probability=True) # 10, 1
    svc.fit(x_train_svc, y_train)
    y_prob = svc.predict_proba(x_val_svc)[:,1]
    
    for threshold in threshold_list:
        y_th_pred = y_prob > threshold
        svc_score = get_score(y_val, y_th_pred)
        if(svc_score > max_svc_score) : 
            max_svc_score = svc_score
            max_svc_threshold = threshold
    print("Score [{}] Threshold {}".format(svc_score, threshold))
            
    # Save TH
    th_df.loc[test_case, 'SVC'] = max_svc_threshold
    th_df.loc[test_case, 'SVC_score'] = max_svc_score
    
    # SVC Fitting 
    svc.fit(x_svc, y)

    with open('{}/RF_TC{}.pkl'.format(MODEL_PATH, test_case), 'wb') as f:
        pickle.dump(clf, f)
        print("SAVE RF Model")

    with open('{}/SVC_TC{}.pkl'.format(MODEL_PATH, test_case), 'wb') as f:
        pickle.dump(svc, f)
        print("SAVE SVC Model")
        
    with open('{}/SVC_scaler{}.pkl'.format(MODEL_PATH, test_case), 'wb') as f:
        pickle.dump(scaler, f)
        print("SAVE scaler")
    
th_df.to_csv("{}/Train_result.csv".format(MODEL_PATH), index=False)
        