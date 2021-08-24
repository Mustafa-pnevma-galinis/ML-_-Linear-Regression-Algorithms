#بسمك يارب نسبحك و نُقدِّس أمرك العظيم
import pandas as pd


import numpy as np


# Applying Decision tree Algorithm
from sklearn.tree import DecisionTreeClassifier


from sklearn.model_selection import train_test_split


# ◘◘@Note: This module is used to measure the accuracy of the applied algorithm
#          regarding the data set.
from sklearn.metrics import accuracy_score


from sklearn import tree


import joblib


from sklearn.datasets import load_iris, load_digits


from sklearn.linear_model import LinearRegression


from sklearn import neighbors , datasets


from sklearn.naive_bayes import GaussianNB


from sklearn.decomposition import PCA

from matplotlib import pyplot as plt
#data = pd.read_csv('C:\\Users\\Mustafa Muhammad\\Documents\\Python Scripts\\
    #ΓMachine Learning\\music.csv')
#print(data)


# ◘◘@Note: that <.drop()> function is used to implement new data set 
#          with the specified column dropped / deleted. 
'''X = data.drop(columns = ['genre'])
Y = data['genre']
'''
#model = DecisionTreeClassifier()


# ◘◘@Note: That method (.predict()) is used to guess the result for
#          the input value; for instance in the example below,
#          The (age and gender) have been passed in separate array
#          which involve the {age and gender} Columns.
'''pred = model.predict([[21, 1], [22, 0]])'''


# ◘◘@Note: use (train_test_split) module might be used to split 
#          the data set for testing..
#          in the example below the data set has been fragmented into 20% of data
#          for testing purposes and 80% for execution
#        model training purposes.
#  ╚>@ Note that the module (train_test_split) ouputs 4 different tuples.
#  ▬▬ First (2) are for input (training and test) input values;
#     and Second (2) are for (training and testing) for output values.
#  ╚>@Note that the test_size of data gets on a very important rule 
#          whereas that directly involved with the accuracy of the result;
#  ▬▬ While supporting the module (train_test_split) 
#     with sufficient proportion of the data set will conclude more accurate result;
#     and vice versa...
#   @Additional Note: That the proportion of data that determined to get the
#                     accuracy might provide different numbers as a data set for
#                     test, while That affects directly to the result, as for each
#                     iteration accuracy might differs.



'''X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
'''
# Fitting the training data sets for both input and output values.
'''model.fit(X_train, Y_train)
'''
# ◘◘@Note: instead of passing the required values as input to get the associated
#       output; passing the new [(X_test)] is basic and an alternative solution.
'''pred = model.predict(X_test)

print(pred)

print(tree.export_graphviz(model, out_file='music-recommender.dot',
                           feature_names=['age','gender'],
                           class_names=sorted(Y.unique()), label='all',
                           rounded=True,filled=(True)))'''

# ◘◘@Note: measure the accuracy; That illustrates mainly comparing values of 
#          the predictions and the result of the output.
#          it returns (1) if it accurate and (0) if not.
'''score = accuracy_score(Y_test, pred)
#print(score)
'''



# ◘◘@Note: using joblib, which is ready-to-use library for model training..
'''trained_model = joblib.load(model, 'music-recommender.joblib')
predic = model.predict([[21,1]])
print(predic)'''




#───────────────── SCIKIT_Machine_Learning_Documentations_──────────────────── 
'''
Machine Learning model is being trained by providing training data <train_X>,
train_Y to predict then value of Y depending on value of {val_X}.
'''

# α) general implementation.
#──────────────────────────

# ▬▬ iris is considered as a flower.
'''iris = load_iris()

n_samples,n_features = iris.data.shape'''
# ◘ iris data.
'''print(iris.data)'''

# ◘ iris targetted data.
'''print(iris.target)'''

# ◘ returns the dataset that will be used to discover and precisely dedicate
#   new data set.
'''print(iris.target_names)'''

# ◘ » plotting the iris dataset.
'''x_index = 0
y_index = 1
formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])
plt.figure(figsize = (5,4))
plt.scatter(iris.data[:, x_index], iris.data[:, y_index], c=iris.target)
plt.colorbar(ticks=[0, 1, 2], format=formatter)
plt.xlabel(iris.feature_names[x_index])
plt.ylabel(iris.feature_names[y_index])

plt.tight_layout()
plt.show()
'''


# ◘ ► Implement new Linear model to apply the dataset on.
'''linear_model = LinearRegression(copy_X=(True),fit_intercept=(True),
                                n_jobs=(True),normalize = True)
print(linear_model)
'''
# ◘ ► Declare the dataset and fitting into the model.
'''x = np.array([0,1,2])
y = np.array([0,1,2])
'''

# » Converting the x-array into 2D- Array.
'''X = x[:,np.newaxis]'''

# ◘ ► Fitting data into the model.
'''linear_model.fit(X,y)'''

# ▬ print(linear_model.coef_)

# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# ß) Supervised learning classification implementation
#────────────────────────────────────────────────────
'''iris = datasets.load_iris()'''

# ◘ ► Identifying data as Features and Samples for <X,Y>..
'''X, Y = iris.data, iris.target'''

# ◘ ► Declaring the model type.
'''knn = neighbors.KNeighborsClassifier(n_neighbors = 1)'''

# ◘ ► Fitting data to the model.
'''knn.fit(X,Y)'''

# What kind of iris has 3cm x 5cm sepal and 4cm x 2cm petal?
'''print(iris.target_names[knn.predict([[3, 5, 4, 2]])])'''


# Γ) LinearRegression and plotting dataset implementation.
#────────────────────────────────────────────────────
'''
x = 30 * np.random.randint((20,1))
y = 0.5 * x + 1.0 + np.random.normal(size = x.shape)

linear_regression_model = LinearRegression()
linear_regression_model.fit(x,y)


x_valu = np.linspace(0, 30, 100)
y_valu = linear_regression_model.predict(x_valu[:, np.newaxis])



plt.figure(figsize = (4, 3))
plotting_axis = plt.axes()
plotting_axis.scatter(x, y)
plotting_axis.plot(x_valu, y_valu)

plotting_axis.set_xlabel("X")
plotting_axis.set_ylabel("Y")

plotting_axis.axis("tight")

plt.show()
'''

# Σ) Supervised learning: Classification of handwritten digits.
#───────────────────────────────────────────────────────────────
'''
digits = load_digits()

fig = plt.figure(figsize = (6,6))
fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1
                    , hspace = 0.05, wspace = 0.05)

for i in range(64):
    data_vig_axis = fig.add_subplot(8, 8, i+1, xticks = [], yticks = [])
    data_vig_axis.imshow(digits.images[i], cmap = plt.cm.binary, interpolation = "nearest")
    
    data_vig_axis.text(0, 7, str(digits.target[i]))




plt.figure()


pca = PCA(n_components = 2)
proj = pca.fit_transform(digits.data)
plt.scatter(proj[:, 0], proj[:, 1], c = digits.target, cmap = "Paired")
plt.colorbar()
'''

# σ) Classify with Gaussian naive Bayes...
#────────────────────────────────────────

# ◘ » Declare and randomised load data.
digits = load_digits()

# ◘ » Split dataset into both training and validation sets.
train_X, test_X, train_Y, test_Y = train_test_split(digits.data, digits.target)


# ◘ » Training and fitting data to the model.
gaussian_data_classifier = GaussianNB()
gaussian_data_classifier.fit(train_X, train_Y)

# ◘ » Getting the deducted values...
pred_valus = gaussian_data_classifier.predict(test_X)


expec_valus = train_Y
print(train_Y)
fig = plt.figure(figsize = (6, 6)) # fig size in inches
fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1, 
                    hspace = 0.05, wspace = 0.05)

for i in range(64):
    data_axis = fig.add_subplot(8, 8, i+1, xticks = [], yticks = [])
    data_axis.imshow(test_X.reshape(-1,8,8)[i], cmap = plt.cm.binary,
                     interpolation = "nearest")
    
    # label images with target values 
    if pred_valus[i] == expec_valus[i]:
        data_axis.text(0, 7, str(pred_valus[i]), color = "green")
    else:
        data_axis.text(0,7, str(pred_valus[i]), color = "red")
        
        
        
        