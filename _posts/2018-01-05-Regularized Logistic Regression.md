---
title: "Regularized Multiclass Logistic Regression Algorithm"
data: 2018-01-05
tags: [Linear Regression]
layout: single
header:
  #image: "/assets/images/optics.jpg"
  teaser: /assets/images/logReg.png
  excerpt: ""

classes: wide

---


<div align = "justify"> Logistic Regression is a key Classification Machine Learning model  that helps to predict a discrete variable given one or more independent variables. Classification entails assigning a class to each sample based on the learned parameters. </div>

A real life example is; <br>
Given a symmetrical college classroom where each student sitted in the room is identified by a row and column value, we can develop a logistic regression model to classify whether each student is male or female. This type of classification is called Binary Classification. i.e

$$Y \in 0,1 $$


Alternatively, if the class room contains first year, second year, third-year, and fourth year student, we can classify whether each student is year 1, 2, 3, or 4. This type is called Multiclass Logistic Regression. i.e
$$Y \in 1,2,3,4 $$

While there are out-of-the box algorithms for Logistic regression from libraries such as Scikit-Learn, in this post, I'd develop a Multivariable Logistic Regression model (that will also work for Binary Classification) from scratch in order to understand the intuition behind such models.

The Logistic Regression is specified as;



 $$h_{(\theta)}(X) = g(\theta_0X_0 + \theta_1X_1 + \theta_2X_2 + ... + \theta_nX_n )= g(\theta^TX) \text {  (using vectorized implementation)} $$
 where $X_0 = 1$

 We can see that $g(X)$ is similar to a Linear Regression. But our Logistic regression model must output probability values between 0 and 1 which specify probaility of each sample belonging to each class.  To do this, we pass the $g(X)$ into a Sigmoid function $g(z)$;

 $$ g(z)= \frac{1}{1 + e^{-z}}$$
 <br>

 $$h_{(\theta)}(X) = \frac{1}{1 + e^{-\theta^TX}}$$

 For a binary classification, $h_{(\theta)}(X)$ will output the probability that the sample instance is in Class 1. We state a decision boundary that says that if this probability value is $>0.5$, sample is in class 1. Else if the probability value is $<0.5$, sample is in Class 0.

 This same method is applied in Multi class classification using the One-vs-All. For example, given a data with 4 classes, we'd train 4 different binary classification model, where for each model one class is assigned 1 and the others are assigned 0. At the end of the iteration, each sample instance will output an array of probability values that sum to 1, stating the probability of each sample instance belonging to class 1, 2, 3, and 4 so that the class with the higest probability is predicted.

 A key aspect of logistic regressison is defining the decision boundary. The decision boundary is delineated by;

 $$\theta_0X_0 + \theta_1X_1 + \theta_2X_2 + ... + \theta_nX_n = \theta^TX\text {  (using vectorized implementation)} $$
Without properly defining the decision boundary, our sigmoid function will output probability values that do not properly classify our sample instances.

Gradient Descent is used to arive at the optimal parameters $\theta_j = (\theta_0, \theta_1, \theta_2,..., \theta_n)$ values which fits our data. Gradient descent works by initializing the parameter values (with zero values), calculate the partial derivarive of cost function with respect to the parameters, subtract a scaled value of the partial derivative from the initial parameter until we arrive at the parameter values that makes cost function converge at a global minimum.





A general overview of the process
- Import required libraries
- Develop model using both Gradient Descent
- Test Model using both methods on a dataset and compare with Sklearn out-of-the box model using prediction accuracy


Lets import the important libraries:


```python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
```
<
We need a cost function that gives a cost of zero if our model $h_{(\theta)}(X)$
prediction equals $Y$ and gives a cost that ranges to infinity if our model $h_{(\theta)}(X)$ prediction is unequal to the actual $Y$ value.
A problem that may arise with logistic regression is overfitting. This is when our logistic model predicts well on the training dataset but is unable to perfom well on previously unseen data. This result when there are many features. With regularization, our goal here is to determine the best decision boundary and also avoid overfitting. To do this we'll add a regularization parameter that helps to control the tradeoff between our two different goals.

Such cost function is stated as;


$$J_{(\theta)} = \frac{1}{m} [\sum_{i=1}^m Y^{(i)}log h_{(\theta)}(X^{(i)}) +(1-Y^{(i)}) log (1 - h_{(\theta)}(X^{(i)}))  ] + \lambda \sum_{i=1}^n\theta_j ^2$$

m = Number of Samples

n = Number of features


```python
def logistic_cost(X_train, y_train, init0s):
  #Returns the cost of our Logistic Regression model given a set of theta parameters (init0s)
  m = len(X_train)
  product = (np.transpose(init0s) * X_train).sum(axis = 1)
  #print(product)
  sigmoid = 1 / (1+ (np.exp(-product)))
  a = (y_train * np.log(sigmoid))
  b = ((1 - y_train) * np.log(1 - sigmoid))
  c = (lambdaa/2*m)*((init0s[1:]**2).sum())
  cost = (-1/m)*np.sum(a+b) + c
  return cost
```

The goal is to Minimize $J_{(\theta)}$ to output optimized $\theta_j = (\theta_0,\theta_1, \theta_2,..., \theta_n)$ values.

To do this, we need the partial derivatives of the cost function wrt each $\theta_j$ value.
Note that the second part of our cost function sums from i = 1 to n. So we'll have a different equation for $\theta_0$ and $(\theta_1, \theta_2,..., \theta_n)$

$$\theta_j := \theta_j - \alpha \frac{\delta J(\theta_0, \theta_1,...,\theta_n)}{\delta\theta_j}$$
$$\theta_0 := \theta_0 - \alpha  \frac {1}{m}\sum_{i=1}^m(h_{(\theta)}(X^{(i)}) - Y^{(i)}))X_0^{(i)}$$
$$\theta_j := \theta_j (1 - \alpha\frac{\lambda}{m})- \alpha  \frac {1}{m}\sum_{i=1}^m(h_{(\theta)}(X^{(i)}) - Y^{(i)}))X_j^{(i)}$$


```python

def partial(X_train, y_train, init0s):
  #Returns the partial derivate of the cost function with respect to each theta value
  m = len(X_train)
  prod = (X_train * init0s).sum(axis = 1)
  sigmoid = 1 / (1+ (np.exp(-prod)))
  diff = (sigmoid - y_train)
  update0 = np.asarray(init0s[0] - (alpha/m)*np.dot(diff.transpose(), X_train[:,0]))
  updatej = init0s[1:]*(1 - alpha*(lambdaa/m)) - (alpha/m)*np.dot(diff.transpose(), X_train[:,1:])
  return  np.concatenate([np.expand_dims(update0, axis=0), updatej], axis = 0)




```

We need to keep calculating the partial derivarives of cost function wrt the parameters, subtract a scaled value of the partial derivative from the initial parameter until we arrive at the paramter values that makes cost function converge at a global minimum. Once the cost converges at minimum, the theta values will remain constant.


```python
def gradient_descent(X_train, y_train, init0s):
  #Returns the optimal theta value using 3000 iterations
  for i in range(3000):
    init0s = partial(X_train, y_train,init0s)
  return init0s
```


```python

```

To implement One-vs-All classification, we have to iterate over each class and replace with 1's and 0's.
1 for the chosen class and 0 for others


```python
def oneVsAll(y_array, class_val):
  #Seperates Y into 1's and 0's for each chosen class
  for i in range(len(y_array)):
    if y_array[i] == class_val:
      y_array[i] = 1
    elif y_array[i] != class_val:
      y_array[i] = 0
  return y_array



def multiClassThetas(X_train, y_train, init0s):
  #Returns  a matrix of the optimal theta values with dimension k * n
  #where k is the number of classes and n is number of features
  y_copy = y_train.copy()
  thetas_list = []
  for i in set(y_train):
    copy = y_train.copy()
    y_copy = oneVsAll(copy, i) #Seperate target into 0s and 1s
    class_thetas = gradient_descent(X_train, y_copy, init0s)
    thetas_list.append(class_thetas)
  all_thetas = np.c_[thetas_list]
  return all_thetas
```


```python
def p_values(thetas, X_train, y_train):
  #Returns each sample instances' probability values for each class with dimension k * m
  #Where k is the number of classes and m is number of samples
  x_tranpose = X_train.transpose()
  p_values = []
  for theta in thetas:
    dot_product = np.dot(theta, x_tranpose)
    p_values.append(1 / (1+ (np.exp(-dot_product))))
  return np.c_[p_values]



def predict(p_values, y_test):
  #Returns predictions
  prediction = np.argmax(p_values, axis = 0)
  return prediction

def accuracy_score(prediction, actual):
  #Returns accuracy score (%)
  accuracy_score = sum(prediction == actual) / len(actual)
  return accuracy_score

```


```python

```

Let's test our model on a dataset


```python
data = datasets.make_classification(n_samples=300, n_features=10, n_informative = 6, n_redundant=3, n_classes=5, random_state = 1)
data
```




    (array([[-1.81458019,  2.33394777, -2.3936822 , ...,  3.36414033,
             -3.73943818, -1.88755868],
            [-0.16650515,  1.35463263,  3.63192178, ...,  0.01524995,
             -0.3123429 ,  2.83485591],
            [ 0.07446529,  0.03744541, -2.53456456, ...,  0.61316887,
              2.26427936, -4.93067066],
            ...,
            [-1.03925447, -3.3815413 ,  1.26568454, ...,  0.0138977 ,
             -3.57490573,  0.23015243],
            [-1.04548304, -1.92299224, -1.4837808 , ..., -1.01732038,
             -1.43899743, -0.19823519],
            [ 0.58937621, -3.83200668, -0.20003948, ...,  1.13090516,
              2.50903518, -4.00841809]]),
     array([1, 4, 2, 3, 0, 1, 3, 0, 0, 3, 0, 1, 4, 0, 2, 1, 3, 1, 2, 2, 1, 4,
            0, 0, 0, 1, 0, 1, 3, 4, 0, 3, 0, 3, 2, 3, 3, 3, 1, 1, 1, 2, 1, 2,
            2, 4, 1, 0, 3, 1, 1, 0, 2, 0, 3, 1, 2, 2, 3, 2, 3, 1, 4, 4, 4, 4,
            4, 0, 1, 3, 1, 1, 4, 4, 4, 0, 1, 3, 1, 4, 2, 4, 4, 3, 1, 0, 2, 1,
            3, 1, 1, 0, 2, 0, 2, 2, 1, 1, 3, 0, 3, 3, 4, 4, 3, 2, 3, 0, 2, 1,
            4, 4, 0, 0, 2, 3, 2, 3, 2, 4, 2, 0, 2, 1, 0, 4, 2, 2, 1, 3, 1, 3,
            1, 0, 0, 3, 4, 2, 4, 3, 3, 4, 0, 1, 0, 0, 2, 2, 3, 4, 1, 2, 0, 4,
            4, 2, 1, 2, 1, 4, 2, 3, 3, 2, 1, 3, 1, 3, 1, 1, 0, 0, 4, 2, 4, 0,
            0, 0, 0, 1, 3, 1, 2, 2, 4, 3, 4, 4, 1, 3, 2, 0, 4, 2, 1, 0, 0, 1,
            1, 2, 0, 3, 1, 0, 4, 2, 0, 1, 1, 2, 2, 3, 4, 2, 1, 4, 3, 1, 3, 2,
            4, 4, 0, 1, 0, 3, 2, 2, 4, 3, 2, 2, 0, 4, 1, 1, 0, 4, 3, 4, 0, 2,
            2, 0, 3, 0, 4, 3, 0, 0, 4, 2, 4, 1, 4, 1, 1, 3, 0, 2, 4, 4, 3, 3,
            2, 4, 0, 3, 3, 1, 2, 2, 1, 3, 2, 0, 1, 1, 3, 4, 3, 0, 4, 0, 3, 4,
            0, 3, 0, 2, 4, 1, 3, 4, 0, 4, 4, 3, 4, 0]))



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

```

*Define our parameters*


```python
k = set(y_train)
n = X_train.shape[1]
alpha = 0.04
lambdaa = 0.05
init0s = np.zeros(n)

```

*Compute Matrix of Optimal Theta Values*


```python
multiClassThetas(X_train, y_train, init0s)
```




    array([[-2.88422479,  0.24309291,  0.16414952, -0.33379612, -0.88075859,
            -0.12240557, -0.62048425, -0.09045171, -0.48629003, -0.59670919,
            -0.31908036],
           [-2.18916809,  0.17476873, -0.42731814,  0.59707005, -0.04991429,
            -0.2136668 ,  0.43408559,  0.41920152,  0.39079001,  0.11236469,
             0.29704426],
           [-2.00092841, -0.31148633,  0.00506626, -0.4361685 ,  0.41469982,
            -0.29536442,  0.041249  , -0.05702246,  0.35759855,  0.09703339,
             0.02012853],
           [-1.98322291, -0.15334435, -0.06941377,  0.62106743, -0.07424716,
             0.56739005,  0.02879656, -0.08823448, -0.51015856,  0.13945534,
             0.04863167],
           [-1.82202513, -0.04592405, -0.01566563, -0.35337018,  0.29177393,
            -0.10437302,  0.02210771, -0.09507905,  0.25969456,  0.0625971 ,
            -0.11330846]])



*Compute Probability values using optimal thetas on the Test set*


```python
optimal_thetas = multiClassThetas(X_train, y_train, init0s)
p_values(optimal_thetas, X_test, y_test)
```




    array([[1.92591934e-03, 2.28051018e-02, 5.35273941e-01, 3.24470478e-01,
            9.46326943e-01, 8.58800510e-02, 8.36377790e-01, 8.49673280e-02,
            3.50325636e-01, 1.08293676e-02, 4.60323480e-01, 2.88200927e-03,
            ...
            1.93714158e-03, 1.23331739e-01, 3.66535955e-03, 5.49014403e-03,
            6.07531548e-01, 4.40306594e-01, 1.11807034e-01, 1.26273503e-02,
            3.00040097e-03, 4.16965008e-02, 1.11211979e-02],
           [4.66432386e-01, 7.64619017e-01, 3.73136432e-02, 9.60473829e-02,
            1.32174215e-03, 1.97554336e-01, 1.24355413e-03, 5.70955459e-03,
            1.12411934e-01, 1.70764345e-01, 3.84620641e-02, 1.07588009e-01,
            ...
            9.32832551e-01, 2.34115405e-04, 1.17651446e-01, 6.08453805e-02,
            2.93824231e-03, 8.85236532e-03, 2.85146965e-02, 3.55560379e-01,
            9.05189594e-02, 3.28891732e-02, 1.58408211e-01],
           [2.14887293e-01, 9.52559044e-02, 5.23400986e-02, 3.63717365e-01,
            6.13793024e-03, 1.70601443e-01, 8.94826503e-02, 4.58306271e-01,
            1.33104179e-01, 2.76977552e-01, 3.39891641e-02, 4.96051384e-01,
            ...
            2.03301006e-01, 2.73880364e-01, 1.61025555e-01, 3.10146733e-01,
            2.03046409e-01, 4.55456159e-02, 9.34140553e-02, 5.99287503e-02,
            5.68466414e-01, 7.76459104e-02, 2.46325287e-02],
           [2.97056461e-01, 1.13920813e-01, 1.53460084e-01, 3.53527233e-03,
            7.18539986e-01, 3.25096460e-02, 1.01642798e-01, 2.44163925e-02,
            2.06456172e-02, 4.43463006e-02, 3.07865722e-01, 9.03378712e-02,
            ...
            2.47016041e-02, 3.21693632e-01, 2.75577498e-01, 5.54969116e-02,
            1.50858994e-02, 3.12367081e-01, 1.73687281e-01, 2.39342614e-01,
            2.67569335e-02, 2.21932048e-01, 6.85279453e-01],
           [1.56779619e-01, 3.74992762e-02, 8.25264788e-02, 2.60208062e-01,
            2.64202429e-02, 1.22376417e-01, 1.79311391e-01, 2.51799282e-01,
            1.61436334e-01, 1.76635627e-01, 5.15799802e-02, 4.29335632e-01,
           ...
            1.30160724e-01, 5.57220485e-01, 1.57596732e-01, 2.73677817e-01,
            3.48417721e-01, 8.42799605e-02, 1.03391112e-01, 7.49187209e-02,
            3.40438836e-01, 1.59124127e-01, 5.46260346e-02]])



*Compute Predictions*


```python
p_values = p_values(optimal_thetas, X_test, y_test)
predict(p_values, y_test)
```




    array([1, 1, 0, 2, 0, 1, 0, 2, 0, 2, 0, 2, 1, 2, 0, 1, 1, 0, 2, 4, 1, 4,
           2, 1, 2, 2, 0, 4, 2, 4, 3, 4, 3, 2, 0, 3, 4, 1, 2, 4, 3, 2, 0, 0,
           1, 4, 0, 0, 3, 1, 0, 3, 4, 0, 0, 0, 4, 1, 1, 1, 1, 1, 2, 4, 4, 2,
           1, 0, 0, 0, 0, 1, 4, 3, 1, 2, 3, 1, 1, 1, 4, 0, 1, 0, 4, 1, 4, 3,
           1, 4, 3, 2, 0, 0, 3, 1, 2, 3, 3])



*Check for Accuracy*


```python
prediction = predict(p_values, y_test)
accuracy_score(prediction, y_test)
```




    0.6565656565656566



*Compare With Sklearn's Logistic Regression Module*


```python
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr.predict(X_test)
```




    array([1, 1, 0, 2, 0, 1, 0, 2, 0, 2, 0, 2, 1, 2, 0, 1, 1, 0, 2, 4, 1, 4,
           2, 1, 2, 2, 0, 4, 2, 4, 3, 0, 3, 2, 0, 3, 4, 1, 2, 4, 3, 2, 0, 0,
           1, 4, 0, 0, 3, 1, 0, 3, 4, 0, 0, 0, 4, 1, 1, 1, 1, 1, 2, 4, 4, 2,
           1, 0, 0, 0, 0, 1, 4, 3, 1, 2, 3, 1, 1, 1, 4, 0, 1, 0, 4, 1, 4, 3,
           1, 4, 3, 2, 0, 0, 3, 1, 2, 3, 3])




```python
sk_prediction = lr.predict(X_test)
accuracy_score(sk_prediction, y_test)
```




    0.6666666666666666




```python

```

**Our model's accuracy score of 65.6% is comparable with Sklearn's 66.6%**


```python

```
