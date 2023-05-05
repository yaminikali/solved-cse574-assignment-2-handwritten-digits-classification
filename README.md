Download Link: https://assignmentchef.com/product/solved-cse574-assignment-2-handwritten-digits-classification
<br>
In this assignment, your task is to implement a Multilayer Perceptron Neural Network and evaluate its performance in classifying handwritten digits. You will use the same network to analyze a more challenging face dataset and compare the performance of the neural network against a <em>deep </em>neural network and a convolutional neural network using the TensorFlow library.

After completing this assignment, you are able to understand:

<ul>

 <li>How Neural Network works and use Feed Forward, Back Propagation to implement Neural Network?</li>

 <li>How to setup a Machine Learning experiment on real data?</li>

 <li>How <em>regularization </em>plays a role in the <em>bias-variance </em>tradeoff?</li>

 <li>How to use TensorFlow library to deploy deep neural networks and understand how having multiple hidden layers can improve the performance of the neural network?</li>

 <li>How to use TensorFlow library to deploy convolutional neural networks and understand the benefit of convolutional neural network compared to fully connected neural network?</li>

</ul>

To get started with the exercise, you will need to download the supporting files and unzip its contents to the directory you want to complete this assignment.

<strong>Warning: </strong>In this project, you will have to handle many computing intensive tasks such as training a neural network. Our suggestion is to use the CSE server Metallica (this server is dedicated to intensive computing tasks) and CSE server Springsteen (this <em>boss </em>server is dedicated to running TensorFlow) to run your computation. YOU MUST USE PYTHON 3 FOR IMPLEMENTATION. In addition, training such a big dataset will take a very long time, maybe many hours or even days to complete. Therefore, we suggest that you should start doing this project as soon as possible so that the computer will have time to do heavy computational jobs.

<h2>1.1         File included in this exercise</h2>

<ul>

 <li><em>mnist all.mat</em>: original dataset from MNIST. In this file, there are 10 matrices for testing set and 10 matrices for training set, which corresponding to 10 digits. You will have to split the training data into training and validation data.</li>

 <li><em>face all.pickle</em>: sample of face images from the CelebA data set. In this file there is one data matrix and one corresponding labels vector. The preprocess routines in the script files will split the data into training and testing data.</li>

 <li><em>py</em>: Python script for this programming project. Contains function definitions –

  <ul>

   <li><em>preprocess()</em>: performs some preprocess tasks, and output the preprocessed train, validation and test data with their corresponding labels. <em>You need to make changes to this function.</em></li>

   <li><em>sigmoid()</em>: compute sigmoid function. The input can be a scalar value, a vector or a matrix. <em>You need to make changes to this function.</em></li>

   <li><em>nnObjFunction()</em>: compute the error function of Neural Network. <em>You need to make changes to this function.</em></li>

   <li><em>nnPredict()</em>: predicts the label of data given the parameters of Neural Network. <em>You need to make changes to this function.</em></li>

   <li><em>initializeWeights()</em>: return the random weights for Neural Network given the number of unit in the input layer and output layer.</li>

  </ul></li>

 <li><em>py</em>: Python script for running your neural network implementation on the CelebA dataset. This function will call your implementations of the functions sigmoid(), nnObjFunc() and nnPredict() that you will have to copy from your nnScript.py. <em>You need to make changes to this function.</em></li>

 <li><em>py</em>: Python script for calling the TensorFlow library for running the deep neural network. <em>You need to make changes to this function.</em></li>

 <li><em>py</em>: Python script for calling the Tensorflow library for a convolutional neural network. <em>You need to make changes to this function.</em></li>

</ul>

<h2>1.2         Datasets</h2>

Two data sets will be provided. Both consist of images.

<h3>1.2.1         MNIST Dataset</h3>

The MNIST dataset [1] consists of a training set of 60000 examples and test set of 10000 examples. All digits have been size-normalized and centered in a fixed image of 28×28 size. In original dataset, each pixel in the image is represented by an integer between 0 and 255, where 0 is black, 255 is white and anything between represents different shade of gray.

You will need to split the training set of 60000 examples into two sets. First set of 50000 randomly sampled examples will be used for training the neural network. The remainder 10000 examples will be used as a validation set to estimate the hyper-parameters of the network (regularization constant <em>λ</em>, number of hidden units).

<h3>1.2.2         CelebFaces Attributes Dataset (CelebA)</h3>

CelebFaces Attributes Dataset (CelebA) [3] is a large-scale face attributes dataset with more than 200K celebrity images. CelebA has large diversities, large quantities, and rich annotations, including:

<ul>

 <li>10,177 number of identities,</li>

 <li>202,599 number of face images, and</li>

 <li>5 landmark locations, 40 binary attributes annotations per image.</li>

</ul>

For this programming assignment, we will have provided a subset of the images. The subset will consist of data for 26407 face images, split into two classes. One class will be images in which the individual is wearing glasses and the other class will be images in which the individual is not wearing glasses. Each image is a 54 × 44 matrix, flattened into a vector of length 2376.

Figure 1: Neural network

<h1>2           Your tasks</h1>

<ul>

 <li>Implement <strong>Neural Network </strong>(forward pass and back propagation)</li>

 <li>Incorporate regularization on the weights (<em>λ</em>)</li>

 <li>Use validation set to tune hyper-parameters for Neural Network (number of units in the hidden layer and <em>λ</em>).</li>

 <li>Run the deep neural network code we provided and compare the results with normal neural network.</li>

 <li>Run the convolutional neural network code and print out the results, for example the confusion matrix.</li>

 <li>Write a report to explain the experimental results.</li>

</ul>

<h1>3           Some practical tips in implementation</h1>

<h2>3.1         Feature selection</h2>

In the dataset, one can observe that there are many features which values are exactly the same for all data points in the training set. With those features, the classification models cannot gain any more information about the difference (or variation) between data points. Therefore, we can ignore those features in the pre-processing step.

Later on in this course, you will learn more sophisticated models to reduce the dimension of dataset (but not for this assignment).

<em>Note: </em>You will need to save the indices of the features that you use and submit them as part of the submission.

<h2>3.2         Neural Network</h2>

<h3>3.2.1         Neural Network Representation</h3>

Neural network can be graphically represented as in Figure 1.

As observed in the Figure 1, there are totally 3 layers in the neural network:

<ul>

 <li>The first layer comprises of (<em>d </em>+ 1) units, each represents a feature of image (there is one extra unit representing the bias).</li>

 <li>The second layer in neural network is called the hidden units. In this document, we denote <em>m </em>+ 1 as the number of hidden units in hidden layer. There is an additional bias node at the hidden layer as well. Hidden units can be considered as the learned features extracted from the original data set. Since number of hidden units will represent the dimension of learned features in neural network, it’s our choice to choose an appropriate number of hidden units. Too many hidden units may lead to the slow training phase while too few hidden units may cause the the under-fitting problem.</li>

 <li>The third layer is also called the output layer. The value of <em>l<sup>th </sup></em>unit in the output layer represents the probability of a certain hand-written image belongs to digit <em>l</em>. Since we have 10 possible digits, there are 10 units in the output layer. In this document, we denote <em>k </em>as the number of output units in output layer.</li>

</ul>

The parameters in Neural Network model are the weights associated with the hidden layer units and the output layers units. In our standard Neural Network with 3 layers (input, hidden, output), in order to represent the model parameters, we use 2 matrices:

<ul>

 <li><em>W</em><sup>(1) </sup>∈ R<em><sup>m</sup></em><sup>×(<em>d</em>+1) </sup>is the weight matrix of connections from input layer to hidden layer. Each row in this matrix corresponds to the weight vector at each hidden layer unit.</li>

 <li><em>W</em><sup>(2) </sup>∈ R<em><sup>k</sup></em><sup>×(<em>m</em>+1) </sup>is the weight matrix of connections from hidden layer to output layer. Each row in this matrix corresponds to the weight vector at each output layer unit.</li>

</ul>

We also further assume that there are <em>n </em>training samples when performing learning task of Neural Network. In the next section, we will explain how to perform learning in Neural Network.

<h3>3.2.2         Feedforward Propagation</h3>

In Feedforward Propagation, given parameters of Neural Network and a feature vector <strong>x</strong>, we want to compute the probability that this feature vector belongs to a particular digit.

Suppose that we have totally <em>m </em>hidden units. Let <em>a<sub>j </sub></em>for 1 ≤ <em>j </em>≤ <em>m </em>be the linear combination of input data and let <em>z<sub>j </sub></em>be the output from the hidden unit <em>j </em>after applying an activation function (in this exercise, we use sigmoid as an activation function). For each hidden unit <em>j </em>(<em>j </em>= 1<em>,</em>2<em>,</em>··· <em>,m</em>), we can compute its value as follow:

<em>d</em>+1

<em>a</em><em>j </em>= X<em>w</em><em>jp</em>(1)<em>x</em><em>p                                                                                                                                       </em>(1)

<em>p</em>=1

(2)

where] is the weight of connection from the <em>p<sup>th </sup></em>input feature to unit <em>j </em>in hidden layer. Note that we do not compute the output for the bias hidden node (<em>m </em>+ 1); <em>z<sub>m</sub></em><sub>+1 </sub>is directly set to 1.

The third layer in neural network is called the output layer where the learned features in hidden units are linearly combined and a sigmoid function is applied to produce the output. Since in this assignment, we want to classify a hand-written digit image to its corresponding class, we can use the one-vs-all binary classification in which each output unit <em>l </em>(<em>l </em>= 1<em>,</em>2<em>,</em>··· <em>,</em>10) in neural network represents the probability of an image belongs to a particular digit. For this reason, the total number of output unit is <em>k </em>= 10. Concretely, for each output unit <em>l </em>(<em>l </em>= 1<em>,</em>2<em>,</em>··· <em>,</em>10), we can compute its value as follow:

<em>m</em>+1

<em>b</em><em>l </em>= X <em>w</em><em>lj</em>(2)<em>z</em><em>j                                                                                                                                        </em>(3)

<em>j</em>=1

(4)

Now we have finished the <strong>Feedforward pass</strong>.

<h3>3.2.3         Error function and Backpropagation</h3>

The error function in this case is the negative log-likelihood error function which can be written as follow:

))                                            (5)

where <em>y<sub>il </sub></em>indicates the <em>l<sup>th </sup></em>target value in 1-of-K coding scheme of input data <em>i </em>and <em>o<sub>il </sub></em>is the output at <em>l<sup>th </sup></em>output node for the <em>i<sup>th </sup></em>data example (See (4)).

Because of the form of error function in equation (5), we can separate its error function in terms of error for each input data <strong>x</strong><em><sub>i</sub></em>:

)                                                                 (6)

where

))                                                  (7)

One way to learn the model parameters in neural networks is to initialize the weights to some random numbers and compute the output value (feed-forward), then compute the error in prediction, transmits this error backward and update the weights accordingly (error backpropagation).

The feed-forward step can be computed directly using formula (1), (2), (3) and (4).

On the other hand, the error backpropagation step requires computing the derivative of error function with respect to the weight.

Consider the derivative of error function with respect to the weight from the hidden unit <em>j </em>to output unit <em>l </em>where <em>j </em>= 1<em>,</em>2<em>,</em>··· <em>,m </em>+ 1 and <em>l </em>= 1<em>,</em>··· <em>,</em>10:

(8)

(9)

where

Note that we are dropping the subscript <em>i </em>for simplicity. The error function (log loss) that we are using in (5) is different from the the squared loss error function that we have discussed in class. Note that the choice of the error function has “simplified” the expressions for the error!

On the other hand, the derivative of error function with respect to the weight from the <em>p<sup>th </sup></em>input feature to hidden unit <em>j </em>where <em>p </em>= 1<em>,</em>2<em>,</em>··· <em>,d </em>+ 1 and <em>j </em>= 1<em>,</em>··· <em>,m </em>can be computed as follow:

(10)

(11)

(12)

Note that we do not compute the gradient for the weights at the bias hidden node.

After finish computing the derivative of error function with respect to weight of each connection in neural network, we now can write the formula for the gradient of error function:

)                                                           (13)

We again can use the gradient descent to update each weight (denoted in general as <em>w</em>) with the following rule:

<em>w</em><em>new </em>= <em>w</em><em>old </em>− <em>γ</em>∇<em>J</em>(<em>w</em><em>old</em>)                                                                               (14)

<h3>3.2.4         Regularization in Neural Network</h3>

In order to avoid overfitting problem (the learning model is best fit with the training data but give poor generalization when test with validation data), we can add a regularization term into our error function to control the magnitude of parameters in Neural Network. Therefore, our objective function can be rewritten as follow:

(15)

where <em>λ </em>is the regularization coefficient.

With this new objective function, the partial derivative of new objective function with respect to weight from hidden layer to output layer can be calculated as follow:

!

(16)

Similarly, the partial derivative of new objective function with respect to weight from input layer to hidden layer can be calculated as follow:

!

(17)

With this new formulas for computing objective function (15) and its partial derivative with respect to weights (16) (17) , we can again use gradient descent to find the minimum of objective function.

<h3>3.2.5         Python implementation of Neural Network</h3>

In the supporting files, we have provided the base code for you to complete. In particular, you have to complete the following functions in Python:

<ul>

 <li><em>sigmoid</em>: compute sigmoid function. The input can be a scalar value, a vector or a matrix.</li>

 <li><em>nnObjFunction</em>: compute the objective function of Neural Network <em>with regularization </em>and the gradient of objective function.</li>

 <li><em>nnPredict</em>: predicts the label of data given the parameters of Neural Network.</li>

</ul>

Details of how to implement the required functions is explained in Python code.

<strong>Optimization: </strong>In general, the learning phase of Neural Network consists of 2 tasks. First task is to compute the value and gradient of error function given Neural Network parameters. Second task is to optimize the error function given the value and gradient of that error function. As explained earlier, we can use gradient descent to perform the optimization problem. In this assignment, you have to use the Python scipy function: <strong>scipy.optimize.minimize </strong>(using the option <em>method=’CG’ </em>for conjugate gradient descent), which performs the conjugate gradient descent algorithm to perform optimization task. In principle, conjugate gradient descent is similar to gradient descent but it chooses a more sophisticated learning rate <em>γ </em>in each iteration so that it will converge faster than gradient descent. Details of how to use <em>minimize </em>are provided here: http://docs.scipy.org/doc/scipy-0.

14.0/reference/generated/scipy.optimize.minimize.html.

We use regularization in Neural Network to avoid overfitting problem (more about this will be discussed in class). You are expected to change different value of <em>λ </em>to see its effect in prediction accuracy in validation set. Your report should include diagrams to explain the relation between <em>λ </em>and performance of Neural Network. Moreover, by plotting the value of <em>λ </em>with respect to the accuracy of Neural Network, you should explain in your report how to choose an appropriate hyper-parameter <em>λ </em>to avoid both underfitting and overfitting problem. You can vary <em>λ </em>from 0 (no regularization) to 60 in increments of 5 or 10.

You are also expected to try different number hidden units to see its effect to the performance of Neural Network. Since training Neural Network is very slow, especially when the number hidden units in Neural Network is large. You should try with small hidden units and gradually increase the size and see how it effects the training time. Your report should include some diagrams to explain relation between number of hidden units and training time. Recommended values: 4<em>,</em>8<em>,</em>12<em>,</em>16<em>,</em>20.

<h1>4           TensorFlow Library</h1>

In this assignment you will only implement a single layer Neural Network. You will realize that implementing multiple layers can be a very cumbersome coding task. However, additional layers can provide a better modeling of the data set. The analysis of the challenging CelebA data set will show how adding more layers can improve the performance of the Neural Network. To experiment with Neural Networks with multiple layers, we will use Google’s TensorFlow library (https://www.tensorflow.org/).

Your experiments should include the following:

<ul>

 <li>Evaluate the accuracy of single hidden layer Neural Network on CelebA data set (test data only), to distinguish between two classes – <em>wearing glasses </em>and <em>not wearing glasses</em>. Use <em>py </em>to obtain these results.</li>

 <li>Evaluate the accuracy of deep Neural Network (try 3, 5, and 7 hidden layers) on CelebA data set (test data only). Use <em>py </em>to obtain these results.</li>

 <li>Compare the performance of single vs. deep Neural Networks in terms of accuracy on test data and learning time.</li>

</ul>