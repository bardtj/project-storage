
import pandas as pd
import classification
from Engine import ForesightEngine

# GET Data 
data = pd.read_csv('Data/df_last.csv')
engine = ForesightEngine(data)

TEST_CASE = 3
x_train = engine.x_train[TEST_CASE]
y_train = engine.y_train[TEST_CASE]
 
x_test = engine.x_test[TEST_CASE]
y_test = engine.y_test[TEST_CASE]

clf = classification.model(x_train, y_train, x_test, y_test)
clf.Voting()