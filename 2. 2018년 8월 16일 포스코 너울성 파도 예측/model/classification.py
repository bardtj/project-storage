import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.grid_search import GridSearchCV

class model:
    
    def __init__(self, train_X, train_y, test_X, test_y):
        self.train_X = train_X
        self.train_y = train_y
        self.test_X  = test_X
        self.test_y  = test_y
        self.rf_clf  = RandomForestClassifier()
        self.mlp_clf = MLPClassifier()
        self.ada_clf = AdaBoostClassifier()
        self.gb_clf  = GradientBoostingClassifier()

    def split_train_dataset(self, test_size=0.2):
        data_size = len(self.train_X)
        sample = int(data_size * (1-test_size))
    
        train_split_X = self.train_X[:sample]
        train_split_y = self.train_y[:sample]    
        val_split_X = self.train_X[sample:]
        val_split_y = self.train_y[sample:]
        print("Train {} Validation {}".format(len(train_split_X), len(val_split_X)))
        return train_split_X, train_split_y, val_split_X, val_split_y

    def get_score(self, y_label, y_pred):
        cnf_matrix = confusion_matrix(y_label, y_pred, labels=[1, 0])
        score = cnf_matrix[0][0] * 2 + cnf_matrix[1][1] - cnf_matrix[0][1] * 2 - cnf_matrix[1][0]
        return score

    def Voting(self):
        
        models = {
            "RandomForest"  : GridSearchCV(self.rf_clf,  {'n_estimators'  : [10, 30, 50, 70], 
                                                          'max_features'  : [5, 10, 15], 
                                                          'class_weight'  : ['balanced', {0: 1, 1: 3}, {0: 3, 1: 1}]} ),
            "MLP"           : GridSearchCV(self.mlp_clf, {'learning_rate_init' : [0.1, 0.05, 0.01],  
                                                          'early_stopping'     : [True],
                                                          'hidden_layer_sizes' : [5, 10, 15, 20]}),
            "AdaBoost"      : GridSearchCV(self.ada_clf, {'learning_rate' : [0.1, 0.05, 0.01]}),
            "GBoost"        : GridSearchCV(self.gb_clf,  {'learning_rate' : [0.1, 0.05, 0.01],
                                                          'n_estimators'  : [10, 30, 50, 70],
                                                          'max_depth'     : [3, 5, 7]}),   # 'max_features'  : [5, 10, 15]
        }
        
        for key, model in models.items():
            print("[{}]".format(key))
            model.fit(self.train_X, self.train_y)
            print(model.best_estimator_)
            print(model.best_score_)
            
            y_pred = model.predict(self.test_X)
            cnf_matrix = confusion_matrix(self.test_y, y_pred, labels=[1, 0])
            print(cnf_matrix)

            if (key == 'RandomForest') : self.rf_clf = model.best_estimator_
            if (key == 'MLP') : self.mlp_clf = model.best_estimator_
            if (key == 'AdaBoost') : self.ada_clf = model.best_estimator_
            if (key == 'GBoost') : self.gb_clf = model.best_estimator_
        
        
        tr_X, tr_y, val_X, val_y = self.split_train_dataset()
        voting_clf = VotingClassifier(estimators=[('rf', self.rf_clf), ('mlp', self.mlp_clf), ('ada', self.ada_clf), ('gb', self.gb_clf)], voting='soft')
        voting_clf.fit(tr_X, tr_y)        
        
        print("Validation Result")
        y_pred = voting_clf.predict(val_X)
        cnf_matrix = confusion_matrix(val_y, y_pred, labels=[1, 0])
        print(cnf_matrix)
        print(classification_report(val_y, y_pred, labels=[1, 0]))
        
        max_score = 0
        max_threshold = 0 
        y_prob = voting_clf.predict_proba(val_X)[:,1]   
        for threshold in np.linspace(0.1, 1, 51):
            y_th_pred = y_prob > threshold
            cnf_matrix = confusion_matrix(val_y, y_th_pred, labels=[1, 0])
            score = self.get_score(val_y, y_th_pred)
            if (score > max_score):
                print("[Threshold {}] Update Score {}->{}".format(threshold, max_score, score))
                max_score = score
                max_threshold = threshold
                
        print("Best threshold [{}]".format(max_threshold))
        y_th_pred = y_prob > max_threshold
        cnf_matrix = confusion_matrix(val_y, y_th_pred, labels=[1, 0])
        print(cnf_matrix) 
        print(classification_report(val_y, y_th_pred, labels=[1, 0]))
        
        # Model Fit
        voting_clf.fit(self.train_X, self.train_y)

        # Test Result
        print("TEST Result", max_threshold)
        y_prob = voting_clf.predict_proba(self.test_X)[:,1]
        y_th_pred = y_prob > max_threshold
        cnf_matrix = confusion_matrix(self.test_y, y_th_pred, labels=[1, 0])
        print(cnf_matrix)