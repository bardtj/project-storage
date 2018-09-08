import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from Engine import ForesightEngine
import pickle

# Define
TEST_CNT = 25
MODEL_PATH = './model_last'

# GET Data 
data = pd.read_csv('Data/df_last.csv')
engine = ForesightEngine(data)

# Get Score
def get_score(y_label, y_pred):
    cnf_matrix = confusion_matrix(y_label, y_pred, labels=[1, 0])
    score = cnf_matrix[0][0] * 2 + cnf_matrix[1][1] - cnf_matrix[0][1] * 2 - cnf_matrix[1][0]
    return score

# Read Dataframe
th_df = pd.read_csv("{}/Train_result.csv".format(MODEL_PATH))
for test_case in range(TEST_CNT):
    
    x_test = engine.x_test[test_case]
    y_test = engine.y_test[test_case]
  
    if (th_df['RF_score'].iloc[test_case] > th_df['SVC_score'].iloc[test_case]):
        
        with open('{}/RF_TC{}.pkl'.format(MODEL_PATH, test_case), 'rb') as f:
            clf = pickle.load(f)
            
        y_prob = clf.predict_proba(x_test)[:,1]    
        y_th_pred = y_prob > th_df['RF'].iloc[test_case]
        rf_score = get_score(y_test, y_th_pred)
        cnf_matrix = confusion_matrix(y_test, y_th_pred, labels=[1, 0])
        print("Testcase [{}] RandomForest".format(test_case))
        print(cnf_matrix)
        print(classification_report(y_test, y_th_pred, labels=[1, 0]))

    else :
        
        with open('{}/SVC_TC{}.pkl'.format(MODEL_PATH, test_case), 'rb') as f:
            svc = pickle.load(f)
            
        with open('{}/SVC_scaler{}.pkl'.format(MODEL_PATH, test_case), 'rb') as f:
            scaler = pickle.load(f)

        x_test_svc = scaler.transform(x_test)
        
        y_prob = svc.predict_proba(x_test_svc)[:,1]    
        y_th_pred = y_prob > th_df['SVC'].iloc[test_case]
        svc_score = get_score(y_test, y_th_pred)
        cnf_matrix = confusion_matrix(y_test, y_th_pred, labels=[1, 0])
        print("Testcase [{}] SVM".format(test_case))
        print(cnf_matrix)
        print(classification_report(y_test, y_th_pred, labels=[1, 0]))