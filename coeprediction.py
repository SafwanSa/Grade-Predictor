import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib as mat
mat.use('TkAgg')
from matplotlib import pyplot as plt
import pickle



data = pd.read_csv("coe202.csv", sep=";")

data = data[["absences","dic","classw","G1","G2","G3"]]

predict = "G3"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

'''
best = 0
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)

    if acc > best:
        best = acc
        # save the model
        # with open("coe202marks.pickle", "wb") as f:
        #     pickle.dump(linear, f)
        print(acc)'''

#
pickle_in = open("coe202marks.pickle", "rb")
linear = pickle.load(pickle_in)

# predictions = linear.predict(x_test)
# for x in range(len(predictions)):
#     print("Predicted: ",np.round(predictions[x],2), x_test[x], "Actual: ",y_test[x])

# ["absences","dic","classw","G1","G2","G3"]
inp = np.array([4, 5, 20, 25, 25]).reshape(1, -1)
predictions = linear.predict(inp)
for x in range(len(predictions)):
    print("Predicted: ",np.round(predictions[x],2), inp)

# p = "dic"
# style.use("ggplot")
# pyplot.scatter(data[p],data["G3"])
# pyplot.xlabel(p)
# pyplot.ylabel("Final Grade")
# pyplot.show()
