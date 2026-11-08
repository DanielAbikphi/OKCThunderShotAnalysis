# DANIEL ABIKPHI | 989-954-1238

#imports
from narwhals._compliant import column
import polars as pl
import time
import numpy as np #this handles number arrays
#split the data into train and test pieces
from sklearn.model_selection import train_test_split
#allows us to groups categories that have the same type 
from sklearn.compose import ColumnTransformer
#StandardScaler: put numbers on same scale
#OneHotEncoder: turns categories of same thing to 0/1's column
from sklearn.preprocessing import OneHotEncoder, StandardScaler
#pipeline helps clean out the data and fit model as one object
from sklearn.pipeline import Pipeline
#logisticRegression: outputs the probability (1's) 1's s if shots are made
from sklearn.linear_model import LogisticRegression
#log_loss: keeps track of score for probability predictions (being low is good)
from sklearn.metrics import log_loss


#step 1: read the traning and testing csv files 
print("Reading training data...") #this is the data for shots that do have outcome and we use to predict
train_df = pl.read_csv("training.csv.gz")
print(train_df.head()) #this is for debugging purposes
print(train_df.shape) #this gives us the number of rows and columns in the training dataframe
time.sleep(2)

print("Reading testing data...") #these are shots we must predict because it has no outcome column
test_df = pl.read_csv("testing.csv.gz")
print(test_df.head()) #this is for debugging purposes
print(test_df.shape) #this gives us the number of rows and column in the test dataframe

#lets build the model x and y selecring only columns that are numeric
#first we need to select the columns that are numeric and categrorical 
num_cols = ['distance','closestdefdist','num_contesters','dribblesbefore','shotclock','shooterspeed']
cat_cols = ['three','contested']

#DEBUGGING
print(train_df[num_cols].head())
print(train_df[cat_cols].head())

#prepare x and y
x = train_df.select(num_cols + cat_cols).to_pandas() #this changes the dataframe from polars to pandas since scikit expects pandas
y = train_df['outcome'].to_numpy().astype(int) #this converts true/false values into 1/0

#ColumnTransformer -- understand this part
preprocessor = ColumnTransformer(
    transformers=[
        ('num',StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ]
)

#Pipeline -- understadn this part too 
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=2000))
])

#now lets split our model into testing and training
x_train, x_val, y_train, y_val = train_test_split(
    x,y,test_size=0.2,random_state = 43, stratify= y
)

model.fit(x_train, y_train)
y_prob = model.predict_proba(x_val)
print(f'Validating Log Loss: {log_loss(y_val, y_prob)}')

#now lets work with the test dataframe
model.fit(x,y)
x_test = test_df.select(num_cols + cat_cols).to_pandas()

#now lets predict the probability outcome 
test_probs = model.predict_proba(x_test)[:,1]

#Now we have to write to the file 
pl.DataFrame({
    'shot_id':test_df['shot_id'],
    'make_prob':test_probs
}).write_csv('submission.csv')

print('Successfully filled')