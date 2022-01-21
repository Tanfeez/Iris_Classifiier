import seaborn as sns
import pickle
import matplotlib.pyplot as pit
from sklearn import datasets 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

iris=datasets.load_iris()
x=iris.data
y=iris.target
x_train,x_test,y_train,y_test=train_test_split(x,y)
lin_reg=LinearRegression()
lin_reg=lin_reg.fit(x_train,y_train)

pickle.dump(lin_reg,open('lin_model.pkl','wb'))  