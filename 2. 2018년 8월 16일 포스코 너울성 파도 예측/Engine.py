import pickle
from copy import deepcopy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


class ForesightEngine(object):
    def __init__(self, data):
        self.raw_data = data
        self.data = None
        self.change_data_format()

        self.test_dates = []
        self.make_test_dates()

        self.train = {}
        self.test = {}
        self.reference_train = {}

        self.train_test_split()

        self.x_train = {}
        self.y_train = {}

        self.x_reference = {}
        self.y_reference = {}

        self.x_test = {}
        self.y_probs = {}
        self.y_test = {}

        self.train_xy_split()
        self.test_xy_split()

        self.threshold = {}
        self.model = {}

    def make_test_dates(self):
        test_dates = \
            ['2014-05-18 07:00:00', '2014-07-06 07:00:00', '2014-09-25 07:00:00', '2014-10-23 07:00:00',
             '2014-12-21 07:00:00', '2015-01-13 07:00:00', '2015-04-04 07:00:00', '2015-06-27 07:00:00',
             '2015-07-18 07:00:00', '2015-11-13 07:00:00', '2015-12-13 07:00:00', '2016-01-22 07:00:00' ,
             '2016-03-04 07:00:00', '2016-08-30 07:00:00', '2016-10-20 07:00:00', '2016-11-30 07:00:00',
             '2016-12-26 07:00:00', '2017-02-12 07:00:00', '2017-03-15 07:00:00', '2017-03-31 07:00:00',
             '2017-06-03 07:00:00', '2017-09-16 07:00:00', '2017-10-12 07:00:00', '2017-10-23 07:00:00',
             '2017-12-11 07:00:00']

        for date in test_dates:
            self.test_dates.append(pd.to_datetime(date))

    def change_data_format(self):
        if 'ds' not in self.raw_data.columns:
            raise(TypeError, "YOU SHOULD HAVE 'DS")

        if 'ds' in self.raw_data.columns:
            data = deepcopy(self.raw_data)
            data['ds'] = pd.to_datetime(self.raw_data['ds'])
            data.set_index('ds', inplace=True)

        self.data = data[pd.to_datetime('2014-01-01'):]

    def train_test_split(self):
        for i, date in enumerate(self.test_dates):
            self.train[i] = self.data[:(date - pd.DateOffset(hours=1))]
            self.test[i] = self.data[date:date+pd.DateOffset(days=1) - pd.DateOffset(hours=1)]
            self.reference_train[i] = self.data[:(date + pd.DateOffset(days=2))]

    def train_xy_split(self):
        for key, value in self.train.items():
            self.x_train[key] = self.train[key][self.train[key].columns[3:]]
            self.y_train[key] = self.train[key]['너울성파도발생여부']
            self.x_reference[key] = self.reference_train[key][self.reference_train[key].columns][3:]
            self.y_reference[key] = self.reference_train[key]['너울성파도발생여부']

    def test_xy_split(self):
        for key, value in self.test.items():
            self.x_test[key] = self.test[key][self.test[key].columns[3:]]
            self.y_test[key] = self.test[key]['너울성파도발생여부']
            self.y_test[key].apply(lambda x: 0)

    def predict_test_x(self):
        for i, value in self.train.items():
            for column in self.x_train[i].columns[:-6]:

                print("==========================")
                print("TEST " + str(i) + " COLUMN " + column)

                end_date = self.x_train[i].index.max()
                one_day = pd.DateOffset(days=1)
                one_hour = pd.DateOffset(hours=1)

                if '_일' not in column:
                    models = []

                    model1 = HoltWinters(frequency=24)
                    #model2 = STL(frequency=24)
                    #model3 = ExponentialMovingAverage(frequency=24)
                    #model4 = AutoArima(frequency=24, seasonal=False)
                    #model5 = AutoRegressiveNN(frequency=[24])

                    models.append(model1)
                    #models.append(model2)
                    #models.append(model3)
                    #models.append(model4)
                    #models.append(model5)

                    fit1, forecast1 = model1.fit_and_predict(self.x_train[i][column][:end_date - one_day], horizon=24)
                    #fit2, forecast2 = model2.fit_and_predict(self.x_train[i][column][:end_date - one_day], horizon=24)
                    #fit3, forecast3 = model3.fit_and_predict(self.x_train[i][column][:end_date - one_day], horizon=24)
                    #fit4, forecast4 = model4.fit_and_predict(self.x_train[i][column][:end_date - one_day], horizon=24)
                    #fit5, forecast5 = model5.fit_and_predict(self.x_train[i][column][:end_date - one_day], horizon=24)

                    scores = []

                    score1 = self.calculate_test_x_score(self.x_train[i][column][end_date - one_day + one_hour:], forecast1['Point.Forecast'])
                    #score2 = self.calculate_test_x_score(self.x_train[i][column][end_date - one_day + one_hour:], forecast2['Point.Forecast'])
                    #score3 = self.calculate_score(self.x_train[i][column][end_date - one_day + one_hour:], forecast3['Point.Forecast'])
                    #score4 = self.calculate_score(self.x_train[i][column][end_date - one_day + one_hour:], forecast4['Point.Forecast'])
                    #score5 = self.calculate_score(self.x_train[i][column][end_date - one_day + one_hour:], forecast5['forecast.mean'])

                    scores.append(score1)
                    #scores.append(score2)
                    #scores.append(score3)
                    #scores.append(score4)
                    #scores.append(score5)

                    index = np.argmin(scores)
                    model = models[index]

                    fit, forecast = model.fit_and_predict(self.x_train[i][column], horizon=24)

                    if model.__class__.__name__ == 'AutoRegressiveNN':
                        self.x_test[i][column] = forecast['forecast.mean'].values
                    else:
                        self.x_test[i][column] = forecast['Point.Forecast'].values

                    print("SELECTED_MODEL IS ", model.__class__.__name__ + " SCORE " + str(scores[index]))
                else:
                    model = ExponentialMovingAverage(frequency=24)
                    fit, forecast = model.fit_and_predict(self.x_train[i][column][:end_date - one_day], horizon=24)
                    score = self.calculate_test_x_score(self.x_train[i][column][end_date - one_day + one_hour:], forecast['Point.Forecast'])
                    fit, forecast = model.fit_and_predict(self.x_train[i][column], horizon=24)
                    self.x_test[i][column] = forecast['Point.Forecast'].values

                    print("SELECTED_MODEL IS ", model.__class__.__name__ + " SCORE " + str(score))

                print("==========================")
            self.x_test[i].to_csv('./engine_x_test_' + str(i) + '.csv')

    def calculate_test_x_score(self, y, yhat):
        return np.sum(np.abs(np.array(y)-np.array(yhat)))

    def load_test_x(self, folder_path):
        columns = self.x_train[0].columns

        for i, value in self.test.items():
            self.x_test[i] = pd.read_csv(folder_path +'/' + 'engine_x_test_' + str(i) + '.csv').set_index('ds').round(2)
            self.x_test[i] = self.x_test[i][columns]
        print("X TEST LOADED")

    def train_model(self, max_depth=None, class_weight={0:1, 1:1}):
        for i, value in self.train.items():
            self.model[i] = RandomForestClassifier(n_estimators=100, max_depth=max_depth, class_weight=class_weight, random_state=2018)
            self.model[i].fit(self.x_train[i], self.y_train[i])
            print("TRAIN COMPLETED " + str(i) + " th RANDOMFOREST MODEL")

    def show_feature_importance(self, index):
        plt.rcParams["font.family"] = 'AppleGothic'

        features = self.x_train[index].columns
        importances = self.model[index].feature_importances_
        indices = np.argsort(importances)

        plt.figure(figsize=(12, 12))
        plt.title('Feature Importances')
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.show()

    def show_model_performance(self, index):
        prediction = self.model[index].predict(self.x_train[index])
        cnf_matrix = confusion_matrix(self.y_train[index], prediction)
        score = cnf_matrix[1][1] * 2 + cnf_matrix[0][0] - cnf_matrix[1][0] * 2 - cnf_matrix[0][1]
        print(cnf_matrix, "\n")
        print(score, "\n")

    def load_threshold(self):
        result = pd.read_csv("./model/Train_result.csv")

        for i, value in self.test.items():
            self.threshold[i] = result.loc[i, 'RF']
        print("THRESHOLD LODADED")

    def load_model(self):
        MODEL_PATH = './model'

        for i, value in self.test.items():
            with open('{}/RF_TC{}.pkl'.format(MODEL_PATH, i), 'rb') as f:
               self.model[i] = pickle.load(f)
        print("MODELS LODAD")

    def predict_test_y(self):
        for i in range(len(self.x_test)):
            self.y_probs[i] = self.model[i].predict_proba(self.x_test[i])[:, 1]

    def postprocess(self):
        pass


class StandardModel(object):
    def __init__(self, frequency):
        self.frequency = frequency

    def fit_and_predict(self):
        pass


class AutoRegressiveNN(StandardModel):
    def __init__(self, frequency):
        super(AutoRegressiveNN, self).__init__(frequency)

    def fit_and_predict(self, train, horizon):
        r_string = """
            function(data, frequency, horizon){
                library(forecast)

                if(length(frequency) == 1){
                    ts_data <- ts(data, frequency=frequency)
                }else{
                    ts_data <- msts(data, seasonal.periods=frequency)
                }

                fit <- nnetar(ts_data)
                fitted_df <- data.frame(fit$fitted)

                forecast <- forecast(fit, h = horizon)
                forecast_df <- data.frame(forecast$mean)

                output <- list(fitted_df, forecast_df)
                return(output)
            }
        """

        r_func = robjects.r(r_string)

        pandas2ri.activate()
        output_list = r_func(train, robjects.IntVector(self.frequency), horizon)
        fit = pandas2ri.ri2py(output_list[0])
        forecast = pandas2ri.ri2py(output_list[1])
        pandas2ri.deactivate()

        return fit, forecast


class AutoArima(StandardModel):
    def __init__(self, seasonal, frequency):
        super(AutoArima, self).__init__(frequency)
        self.seasonal = seasonal

    def fit_and_predict(self, train, horizon):
        r_string = """
            function(data, frequency, seasonal, horizon){
                library(forecast)
                ts_data <- ts(data, frequency=frequency)

                fit <- auto.arima(ts_data, seasonal=seasonal)
                fitted_df <- data.frame(fit$fitted)

                forecast <- forecast(fit, h = horizon)
                forecast_df <- data.frame(forecast)

                output <- list(fitted_df, forecast_df)
                return(output)
            }
        """

        r_func = robjects.r(r_string)

        pandas2ri.activate()
        output_list = r_func(train, self.frequency, self.seasonal, horizon)
        fit = pandas2ri.ri2py(output_list[0])
        forecast = pandas2ri.ri2py(output_list[1])
        pandas2ri.deactivate()

        return fit, forecast


class TBATS(StandardModel):
    def __init__(self, frequency):
        super(TBATS, self).__init__(frequency)

    def fit_and_predict(self, train, horizon):
        r_string = """
            function(data, frequency, horizon){
                library(forecast)

                if(length(frequency) == 1){
                    ts_data <- ts(data, frequency=frequency)
                }else{
                    ts_data <- msts(data, seasonal.periods=frequency)
                }

                fit <- tbats(ts_data)
                fitted_df <- data.frame(fit$fitted.values)

                forecast <- forecast(fit, h = horizon)
                forecast_df <- data.frame(forecast)

                output <- list(fitted_df, forecast_df)
                return(output)
            }
        """

        r_func = robjects.r(r_string)

        pandas2ri.activate()
        output_list = r_func(train, robjects.IntVector(self.frequency), horizon)
        fit = pandas2ri.ri2py(output_list[0])
        forecast = pandas2ri.ri2py(output_list[1])
        pandas2ri.deactivate()

        return fit, forecast


class ExponentialMovingAverage(StandardModel):
    def __init__(self, frequency):
        super(ExponentialMovingAverage, self).__init__(frequency)

    def fit_and_predict(self, train, horizon):
        r_string = """
            function(data, frequency, horizon){
                library(forecast)
                ts_data <- ts(data, frequency=frequency)

                fit <- HoltWinters(ts_data, beta=FALSE, gamma=FALSE)
                fitted_df <- data.frame(fit$fitted)

                forecast <- forecast(fit, h = horizon)
                forecast_df <- data.frame(forecast)

                output <- list(fitted_df, forecast_df)
                return(output)
            }
        """

        r_func = robjects.r(r_string)

        # Run R
        pandas2ri.activate()
        output_list = r_func(train, self.frequency, horizon)
        fit = pandas2ri.ri2py(output_list[0])
        forecast = pandas2ri.ri2py(output_list[1])
        pandas2ri.deactivate()

        return fit, forecast


class STL(StandardModel):
    def __init__(self, frequency):
        super(STL, self).__init__(frequency)

    def fit_and_predict(self, train, horizon):
        r_string = """
            function(data, frequency, horizon){
                library(forecast)
                ts_data <- ts(data, frequency=frequency)

                fit <- stl(ts_data, s.window="periodic")
                fitted_df <- data.frame(fit$time.series)

                forecast <- forecast(fit, h = horizon)
                forecast_df <- data.frame(forecast)

                output <- list(fitted_df, forecast_df)
                return(output)
            }
        """

        r_func = robjects.r(r_string)

        # Run R
        pandas2ri.activate()
        output_list = r_func(train, self.frequency, horizon)
        fit = pandas2ri.ri2py(output_list[0])
        forecast = pandas2ri.ri2py(output_list[1])
        pandas2ri.deactivate()

        return fit, forecast


class HoltWinters(StandardModel):
    def __init__(self, frequency):
        super(HoltWinters, self).__init__(frequency)

    def fit_and_predict(self, train, horizon):
        r_string = """
            function(data, frequency, horizon){
                library(forecast)
                ts_data <- ts(data, frequency=frequency)

                fit <- HoltWinters(ts_data)
                fitted_df <- data.frame(fit$fitted)

                forecast <- forecast(fit, h = horizon)
                forecast_df <- data.frame(forecast)

                output <- list(fitted_df, forecast_df)
                return(output)
            }
        """

        r_func = robjects.r(r_string)

        # Run R
        pandas2ri.activate()
        output_list = r_func(train, self.frequency, horizon)
        fit = pandas2ri.ri2py(output_list[0])
        forecast = pandas2ri.ri2py(output_list[1])
        pandas2ri.deactivate()

        return fit, forecast