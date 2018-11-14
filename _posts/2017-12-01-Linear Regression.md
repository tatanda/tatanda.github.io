---
title: "Developing a Linear Regression Algorithm"
data: 2017-12-20
tags: [Linear_Regression]
header:
  #image: "/assets/images/optics.jpg"
  teaser: /assets/images/cost converge.png
  excerpt: "Machine Learning, Linear Regression, Gradient Descent, Data Science"

classes: wide

---


<div align = "justify"> Linear Regression is a key Machine Learning model that helps to predict a continous variable given one or more independent variables (features). When there is one continous variable, we have a Single Variable Linear Regression. On the other hand, two or more independent variables is called a Multivariable Linear Regression.</div>

<div align = "justify"> While there are out-of-the box algorithms for Linear regression from libraries such as Scikit-Learn, in this post, I attempt to develop a Multivariable Linear Regression from scratch in order to understand the intuition behind such models.</div>

<div align = "justify"> LinearRegression models a linear relationship between one dependent and one or more independent variables. This is done by Parametric Learning.
Parametric learning entails finding the optimal parameter values that fits the linear relationship between variables.</div>

The Linear Regression is specified as;

 $$Y = \theta_0X_0 + \theta_1X_1 + \theta_2X_2 + ... + \theta_nX_n$$
 where $X_0 = 1$

where our prediction hypothesis is $h_{(\theta)}(X)  = \theta_0X_0 + \theta_1X_1 + \theta_2X_2 + ... + \theta_nX_n$ and $Y$ is our actual.

<div align = "justify"> The goal of parametric learning is to compute the optimal parameter values that fits the linear relationship between $Y$ and $X_i$.</div>

There are two ways to compute these optimized parameters;
- Gradient Descent
- Normal Equation



A general overview of the process;
- Import required libraries
- Develop model using both Gradient Descent and Normal Equation
- Test Model using both methods on a dataset and compare with Sklearn out-of-the box model using
the Root Mean Square Error (RMSE)

<div align = "justify"> RMSE is the average of the square differences between our hypothesis predicted values and the actual values.
It's used as a means to test for model accuracy.</div>

Let's import the important libraries:


```python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
```

### 1. Gradient Descent

<div align = "justify"> Optimization processes entail one of two things. Either maximize payoff or minimize cost.

Gradient Descent is a general process used to minimize several functions including the cost function. It works by
initializing the parameter values, calculate the partial derivatives of cost function with respect to the parameters, subtract a
scaled value of the partial derivative from the initial parameter until we arrive at the parameter values that makes cost function converge at a global minimum.</div>

<div align = "justify"> To compute optimized parameter values using Gradient Descent, we have to minimize the cost function. In statistics, there are several cost function but the one we'd use is the Squared Error Function which is specified as;</div>

$$J_{(\theta)} = \frac{1}{2m} \sum_{i=1}^m (h_{(\theta)}(X^{(i)}) - Y^{(i)})^2$$


<div align = "justify"> Since we don't know what the optimized $\theta$ values are, we have to initialize theta values by giving them a value of zeros</div>


```python
def cost_function(X, y, init0s): #Returns the cost value given the theta values.
  m = len(X) #m is the length of X i.e the number of samples
  cost = 1/(2*m)*np.sum((np.dot(init0s, np.transpose(X)) - y)**2)
  return cost
```

<div align = "justify"> The goal is to Minimize $J_{(\theta)}$ to give optimized $\theta_j = (\theta_1, \theta_2,..., \theta_n)$ values.</div>

<div align = "justify"> To do this, we need the partial derivatives of the cosy function wrt each $\theta_j$ value.</div>

$$\theta_j := \theta_j - \alpha \frac{\delta J(\theta_0, \theta_1,...,\theta_n)}{\delta\theta_j}$$

<div align = "justify"> $\alpha$ is the learning rate which specifies how big a step to take downshill when creating descent. IT must not be too large else the cost won't converge at minimum and it must not be too small else the the model take long to converge.</div>


```python
def partial_derivs(X, y, init0s): #Returns a list of updated thetas
  m = len(X)
  predicted = (np.transpose(init0s) * X).sum(axis = 1)
  diff = (predicted - y)
  diff_transpose = np.transpose(diff[:,np.newaxis])
  #Add a new axis to make it vectorizable for dot product
  updated0s = init0s - alpha*((1/m))* np.dot(diff_transpose, X)[0,:]
  #Remove the extra dimension/axis
  return updated0s
```

<div align = "justify"> We need to keep calculating the partial derivarives of cost function with respect to the parameters, subtract a scaled value of the partial derivative from the initial parameter until we arrive at the parameter values that makes
cost function converge at a global minimum. Once the cost converges at minimum, the theta values will remain constant. </div>


```python

def gradient_descent(X, y,init0s):
  #Returns the optimal theta value and a list of costs for each iteration
  costs = [cost_function(X, y,init0s)]
  for i in range(epochs):
    init0s = partial_derivs(X, y,init0s)
    c = cost_function(X, y, init0s)
    costs.append(c)
  return init0s, costs
```


```python
def predict(X_test, y_test, optimal0s):
  predicted = (np.transpose(optimal0s) * X_test).sum(axis = 1)
  rmse = mean_squared_error(y_test, predicted)
  return predicted, rmse
```



Let's test out our Linear Regression model on a dataset


```python
data = datasets.make_regression(100,2, noise = 1) #100 samples with two features
print(data)
```

    (array([[-2.37759907,  0.4415468 ],
           [ 1.35768791,  0.62366801],
           [ 0.46415043, -0.60795592],
           [ 0.11919051,  0.03099776],
           [-0.25396912, -0.43361524],
           [ 0.82882596, -1.03243575],
           [-0.28876997, -0.387609  ],
           [ 0.5714437 ,  1.05848605],
           [ 0.12303767, -0.67112459],
           ...
           [-0.25638847, -0.50759324],
           [-0.25334618, -1.46977272],
           [-0.25473333,  0.41628009],
           [-0.2367325 , -0.38696975],
           [-0.93864887, -1.20648181],
           [ 1.02867051,  0.69952625],
           [-1.10693866,  0.82899393],
           [ 0.41280868,  0.67398247],
           [-0.82242226, -1.46872768]]), array([-128.32612235,  156.62936098,  
            -23.00157862,   13.79748273,
            -59.15950308,  -37.38216146,  -58.42959904,  139.89493792,
            -54.37098522, -155.89579496,  -41.78211416, -197.90237397,
            139.70894371,  -60.19288674,  168.64414418,   85.5834497 ,
            135.45948204,  -41.43222537,  -19.23625686,  -66.77387176,
            ...
             61.18294922,  -69.82805268, -280.76019724,  297.69689676,
            -64.27834335,  334.9217973 ,   54.22484719,  225.38827061,
            -18.70734055,  -82.97630182,  150.42763414,   28.80524967,
           -121.28762166,  -16.01779866,    3.33038114,  -64.08131134,
           -156.47444545,   19.18896421,  -52.56071383, -180.88100031,
            140.14608223,    0.4524311 ,   93.43954001, -196.05294509]))


We need to carry out some preprocessing on data.
- Add the $X_0$ column with 1's
- Split the data into Train and Split for Validation




```python
def preprocess_data(data):
  X = data[0]
  y = data[1]
  #copy = data.copy
  ones = np.ones(len(X))
  X = np.c_[ones, X]
  return train_test_split(X, y, test_size=0.33, random_state=1)

X_train, X_test, y_train, y_test = preprocess_data(data)
#y_train
```

*Plot the Cost values*

<div align = "justify"> To check that our gradient descent theta values are actually optimized, we have to plot the
cost values to check for convergence</div>


```python
epochs = 1000 #Number of Iterations
alpha = 0.04
init0s = np.zeros(3)

y = gradient_descent(X_train, y_train, init0s)[1]
x = np.arange(len(y))
plt.xlabel("Number of Epochs")
plt.ylabel("Cost")
plt.plot(x,y)
```




    [<matplotlib.lines.Line2D at 0x7f2d50903ac8>]




![png](/images/Developing a Linear Regression Model/cost converge.png)


*Optimal Theta Values*


```python
gradient_descent(X_train, y_train, init0s)[0]
```




    array([-6.50232840e-02,  7.15272503e+01,  9.40884410e+01])

 So our multivariable linear regression model is; <br>
 $Y$ = -0.0650232840$X_0$ + 71.5272503$X_1$ + 94.0884410$X_2$

*Compare our model's RMSE with Sklearn's*


```python
epochs = 1000 #Number of Iterations
alpha = 0.04
init0s = np.zeros(3)
predict(X_test, y_test, gradient_descent(X_train, y_train, init0s)[0])[1]
#Returns the model RMSE
```




    1.088651983784035




```python
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
prediction = lr.predict(X_test)
rmse = mean_squared_error(y_test, prediction)
rmse
```




    1.088651983784131



Our model's RMSE value (1.0886519) is comparable to Sklearn's RMSE value (1.0886519)





### 2. Normal Equation

<div align = "justify"> The normal equation uses linear algebra to compute the optimal theta values at once.</div>

where $\theta_j = (X^TX)^{-1}X^TY$

<div align = "justify"> The normal equation is useful when number of features is not very large and $(X^TX)$ is invertible. Otherwise, Gradient Descent should be used. With Normal Equation, there is no iteration or initializing of theta values.</div>



```python
def normal_equation(X, y):
  xtrans = np.transpose(X)
  inv = np.linalg.pinv(np.dot(xtrans, X)) #use pinv for nonivertible singular matrices
  and use inv for normal invertible matrices
  optimal0s = np.dot(np.dot(inv, xtrans), y)
  return optimal0s
normal_equation(X_train, y_train)
```




    array([1.96367929e-02, 9.18139647e+01, 3.07754066e+01])



<div align = "justify"> We can see that the values for the normal equation are similar to that of gradient descent.</div>
