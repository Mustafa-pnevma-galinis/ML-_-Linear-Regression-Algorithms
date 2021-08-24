#                                                                        بسمك يارب توكلت عليك ف إجعل عاقبة أمري رشدا  

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor




# ◘ Using the traditional method.
'''with open ("C:\\Users\\Mustafa Muhammad\\Documents\\Python Scripts\\Machine Learning\\melb_data.csv") as file:
        file_data = file.read()

print(file_data)
'''


# ◘ Using pandas to read the file.
melb_data_path = "C:\\Users\\Mustafa Muhammad\\Documents\\Python Scripts\\Machine Learning\\melb_data.csv"
melb_to_read_data = pd.read_csv(melb_data_path)



# ◘ Describe the dataFrame basic calculations of the provided file.
'''print(melb_to_read_data.describe())'''



# ◘ Get the dataFrame columns
'''print(melb_to_read_data.columns)'''




# ◘ Using {dropna()} function which is mainly used to get data and ignore missing data. <(na) means not available>.
# » Will return the same dataFrame but with ignoring missing data.
'''print(melb_to_read_data.dropna(axis = 0))'''

# ◘ Determinig the {Price} column in dataFrame.
Y = melb_to_read_data.Price



# ◘ Choosing features of data; from the dataFrame columns.
melbourne_features = ["Rooms","Bathroom","Landsize","Lattitude","Longtitude"]


X = melb_to_read_data[melbourne_features]
'''print(X)'''


# ◘◘ Building the model.

        # ┐└┴┬┬├─┼╞ Use [scikit-learn library] to implement new models;
        # » Steps for implementing machine learning model are:
        # α) Define: Define the model; the decision tree.
        # ß) Fit: Capture patterns from the provided data.
        # Γ) Predict
        # π) Evaluate: Determining how accurate the model is.


#─────────────────────────────────────────────────────────────────────────────────────────
# ◘◘ For the X and Y:                                                                         |
# in the model provided X > represents the pattern to be worked in, and Y represents the main |
# Factor for predictions...         
# # X » PATTERN.
# # Y » FACTOR.                                                          
#─────────────────────────────────────────────────────────────────────────────────────────


#───────────────────────────────   ... DOCUMENTATION FOR DECISION_TREE_CALSSIFIER ...  ─────────────────────────────────

'''
melb_data_model = DecisionTreeClassifier(random_state = 1)
melb_data_model.fit(X,Y)
'''

# » That will return {Y} which indicates prices according to the provided columns of {X}.
'''predictions = melb_data_model.predict(X)'''


# » Get the average value of error in the provided values
'''
mean_of_error = mean_absolute_error(Y,predictions)
print(mean_of_error) # returns » 1151.270324005891
'''



# ◙ Using model validation and training. <DecisionTreeRegression>

# ◘ to get more accurate values involved with the provided data; Training the model with 
#   supportive data will be used to solve this.


#───────────────────────────────   ... DOCUMENTATION FOR DECISION_TREE_REGRESSOR ...  ─────────────────────────────────
# ◘ » [train_test_split] is maily allocated for splitting data into data for <training> and data for <validation>;
#     random_state = value >> This numeric value guarantees to get the same split for the provided data set.

# ◘ Training the model.
train_X, val_X, train_Y, val_Y = train_test_split(X,Y,random_state = 0)

'''
This Testing Model will be splitted into {2} data sets; one of them is allocated for data training {train_X} & {train_Y}
         and other one for testing and validation {val_X} & {val_Y} __val » abbreviation for [validation].
'''



# » Define the model 
'''melbourne_model = DecisionTreeRegressor()
'''
# » Fit The Model > with the pattern and factor.
'''melbourne_model.fit(X,Y)
'''
'''
pred_vals = melbourne_model.predict(val_X)
print(mean_absolute_error(val_Y, pred_vals)) # returns 1114.2368188512519 {after solving some errors}.
'''

#───────────────────────────────   ... DOCUMENTATION FOR Random_Forests_Algorithms ...  ─────────────────────────────────
forest_model = RandomForestRegressor(random_state = 1)
forest_model.fit(train_X,train_Y)
pred_vals = forest_model.predict(val_X)
print(mean_absolute_error(val_Y, pred_vals))









