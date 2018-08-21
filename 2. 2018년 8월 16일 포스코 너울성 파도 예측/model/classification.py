import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.grid_search import GridSearchCV

class model:
    
    def __init__(self, train_X, train_y, test_X, test_y):
        self.train_X = train_X
        self.train_y = train_y
        self.test_X  = test_X
        self.test_y  = test_y
        self.rf_clf  = RandomForestClassifier()
        self.ada_clf = AdaBoostClassifier()
        self.gb_clf  = GradientBoostingClassifier()

    def split_train_dataset(self, test_size=0.2):
        data_size = len(self.train_X)
        sample = int(data_size * (1-test_size))
    
        x_train = self.train_X[:sample]
        y_train = self.train_y[:sample]    
        x_val = self.train_X[sample:]
        y_val = self.train_y[sample:]
        print("Train {} Validation {}".format(len(x_train), len(x_val)))
        return x_train, y_train, x_val, y_val

    def get_score(self, y_label, y_pred):
        cnf_matrix = confusion_matrix(y_label, y_pred, labels=[1, 0])
        score = cnf_matrix[0][0] * 2 + cnf_matrix[1][1] - cnf_matrix[0][1] * 2 - cnf_matrix[1][0]
        return score

    def Voting(self):
        
        models = {
            "RandomForest"  : GridSearchCV(self.rf_clf,  {'n_estimators' : [10, 20, 30], 
                                                          'max_features' : [5, 10, 15], 
                                                          'class_weight' : ['balanced', {0: 1, 1: 3}, {0: 3, 1: 1}]} ),
            "AdaBoost"      : GridSearchCV(self.ada_clf, {'learning_rate' : [1, 0.1, 0.01]}),
            "GBoost"        : GridSearchCV(self.gb_clf,  {'learning_rate' : [1, 0.1, 0.01], 
                                                          'max_depth' : [3, 5, 8],
                                                          'max_features' : [5, 10, 15]}),
        }
        
        for key, model in models.items():
            print(key)
            model.fit(self.train_X, self.train_y)
            print(model.best_estimator_)
            
            y_pred = model.predict(self.test_X)
            cnf_matrix = confusion_matrix(self.test_y, y_pred, labels=[1, 0])
            print(cnf_matrix)

            if (key == 'RandomForest') : self.rf_clf = model.best_estimator_
            if (key == 'AdaBoost') : self.ada_clf = model.best_estimator_
            if (key == 'GBoost') : self.gb_clf = model.best_estimator_
        
        x_train, y_train, x_val, y_val = self.split_train_dataset()
        voting_clf = VotingClassifier(estimators=[('rf', self.rf_clf), ('ada', self.ada_clf), ('gb', self.gb_clf)], voting='soft')
        voting_clf.fit(x_train, y_train)
        print("X-Val Result")
        y_pred = voting_clf.predict(x_val)
        cnf_matrix = confusion_matrix(y_val, y_pred, labels=[1, 0])
        print(cnf_matrix)
        print(classification_report(y_val, y_pred, labels=[1, 0]))
        
        max_score = 0
        max_threshold = 0 
        y_prob = voting_clf.predict_proba(x_val)[:,1]   
        for threshold in np.linspace(0, 1, 51):
            y_th_pred = y_prob > threshold
            cnf_matrix = confusion_matrix(y_val, y_th_pred, labels=[1, 0])
            score = self.get_score(y_val, y_th_pred)
            if (score > max_score):
                print(cnf_matrix) 
                print(classification_report(y_val, y_th_pred, labels=[1, 0]))
                print("[Threshold {}] Update Score {}->{}".format(threshold, max_score, score))
                max_score = score
                max_threshold = threshold

        print("TEST Result", max_threshold)
        y_prob = voting_clf.predict_proba(self.test_X)[:,1]
        y_th_pred = y_prob > max_threshold
        cnf_matrix = confusion_matrix(self.test_y, y_th_pred, labels=[1, 0])
        print(cnf_matrix)