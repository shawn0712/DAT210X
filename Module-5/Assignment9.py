import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import preprocessing

matplotlib.style.use('ggplot') # Look Pretty
%matplotlib inline

def drawLine(model, X_test, y_test, title, R2):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X_test, y_test, c='g', marker='o')
    ax.plot(X_test, model.predict(X_test), color='orange', linewidth=1, alpha=0.7)

    title += " R2: " + str(R2)
    ax.set_title(title)
    print (title)
    print ("Intercept(s): ", model.intercept_)

    plt.show()

def drawPlane(model, X_test, y_test, title, R2):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_zlabel('prediction')

    X_test = np.array(X_test)
    col1 = X_test[:,0]
    col2 = X_test[:,1]

    x_min, x_max = col1.min(), col1.max()
    y_min, y_max = col2.min(), col2.max()
    x = np.arange(x_min, x_max, (x_max-x_min) / 10)
    y = np.arange(y_min, y_max, (y_max-y_min) / 10)
    x, y = np.meshgrid(x, y)

    z = model.predict(  np.c_[x.ravel(), y.ravel()]  )
    z = z.reshape(x.shape)

    ax.scatter(col1, col2, y_test, c='g', marker='o')
    ax.plot_wireframe(x, y, z, color='orange', alpha=0.7)
  
    title += " R2: " + str(R2)
    ax.set_title(title)
    print (title)
    print ("Intercept(s): ", model.intercept_)
    plt.show()

X=pd.read_csv("../Module5/Datasets/College.csv",index_col =0)
X.Private = X.Private.map({'Yes':1, 'No':0})
X.dtypes()#print(X.dtypes)# check whether contains missing value etc

column=X.columns
model = linear_model.LinearRegression()

def process(scale,data):
    T=scale.fit_transform(data)
    X=pd.DataFrame(T,columns=column)
    return X
# funtion used for differenet processing method

def draw(scale,data,X_,y_):
    print (scale)
    X=process(scale,data)[[X_]].copy()
    y=process(scale,data)[[y_]].copy()
    
    from sklearn import cross_validation
    X_train, X_test,y_train, y_test= cross_validation.train_test_split(X,y,test_size=0.3,random_state=7)
    model.fit(X_train,y_train)
    score=model.score(X_test, y_test)
    drawLine(model, X_test, y_test, X_, score)
    
# function for different column relationship
    

    
'''
draw(preprocessing.Normalizer(),X,"Room.Board","Accept")
will return using Normalizer processing, the linaer relationship between room.board and accept
'''
    
   
