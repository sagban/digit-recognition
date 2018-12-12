import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from PIL import Image

print("Imorting Done")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR,"dataset/train.csv")
TEST_PATH = os.path.join(BASE_DIR,"dataset/test.csv")


LABELS = 10           # Number of labls(1-10)
IMAGE_WIDTH = 28      # Width/height if the image
COLOR_CHANNELS = 1    # Number of color channels

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

print("Data Done")

train_y = np.asarray(train.pop('label'))
train_x = np.asarray(train)
# test_y = np.asarray(test.pop('label'))
test_x = np.asarray(test)


#Exploring the Dataset
m_train = train_x.shape[0]
m_test = test_x.shape[0]
print("Number of Training Exapmles: " + str(m_train))
print("Number of test Exapmles: " + str(m_test))



# Optional: Checking the random image
# index = int(abs((np.random.randn()*100000)%m_train))
# train_image = train_x[index].reshape(IMAGE_WIDTH, IMAGE_WIDTH)
# print("Size Of the Image: "train_image.shape)
# plt.imshow(train_image)
# plt.show()


#Standardizing the data
train_x = train_x.T/225.
train_y = train_y.reshape(1, m_train)
temp = np.zeros((m_train, LABELS))
temp[np.arange(m_train), train_y] = 1
train_y = temp.T
test_x = test_x.T/225.

print ("train_x's shape: " + str(train_x.shape))
print ("train_y's shape: " + str(train_y.shape))
print ("test_x's shape: " + str(test_x.shape))


#Architect Of the Neural Network
# 2-Layer Neural Network

# Helper Function: Softmax
def softmax(z):

    # print(z)
    s = np.exp(z)
    sum = np.sum(s, axis=0)

    s /= np.squeeze(sum)
    return s

# Helper Function: Relu
def relu(z):
    z[z<0] = 0
    return z


def sigmoid(z):
    return 1/(1 + np.exp(-z))


# Initialize Parameters
def initialize_parameter(n_x, n_h1, n_h2, n_h3, n_y):

    W1 = np.random.randn(n_h1, n_x) * 0.01
    b1 = np.zeros((n_h1, 1))
    W2 = np.random.randn(n_h2, n_h1) * 0.01
    b2 = np.zeros((n_h2, 1))
    W3 = np.random.randn(n_h3, n_h2) * 0.01
    b3 = np.zeros((n_h3, 1))
    W4 = np.random.randn(n_y, n_h3) * 0.01
    b4 = np.zeros((n_y, 1))

    assert(W1.shape == (n_h1, n_x))
    assert(b1.shape == (n_h1, 1))
    assert(W2.shape == (n_h2, n_h1))
    assert(b2.shape == (n_h2, 1))
    assert (W3.shape == (n_h3, n_h2))
    assert (b3.shape == (n_h3, 1))
    assert (W4.shape == (n_y, n_h3))
    assert (b4.shape == (n_y, 1))

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2,
        "W3": W3,
        "b3": b3,
        "W4": W4,
        "b4": b4
        }
    return parameters

#Forward Propagation
def linear_forward(A, W, b, activation):


    Z = np.dot(W, A) + b
    linear_cache = (A, W, b)

    if activation == "softmax":
        A = softmax(Z)
        activation_cache = A

    elif activation == "relu":
        A = relu(Z)
        activation_cache = A
    elif activation == "sigmoid":
        A = sigmoid(Z)
        activation_cache = A
    cache = (linear_cache, activation_cache)
    return A, cache


#calculating cost
def compute_cost(AL, Y):


    m = Y.shape[1]

    # cost = -(1/m) * np.sum(np.sum(Y * np.log(AL), axis = 0, keepdims=True), axis = 1)
    cost = -(1/m) * np.sum(np.sum(Y * np.log(AL) + (1-Y)*np.log(1-AL), axis = 0, keepdims=True), axis = 1)

    # print(cost)

    cost = np.squeeze(cost)
    assert(cost.shape == ())

    return cost


def linear_backward(dZ, cache):


    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = (1/m) * np.dot(dZ, A_prev.T)
    db = (1/m) * np.sum(dZ, axis=1, keepdims= True)

    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db

def softmax_backward(dA, activation_cache):

    dZ = dA * ((activation_cache) * (1 - activation_cache))
    return dZ

def sigmoid_backward(dA, activation_cache):
    dZ = dA * (activation_cache) * (1 - activation_cache)
    return dZ

def relu_backward(dA, activation_cache):

    (x, y) = activation_cache.shape

    activation_cache = activation_cache.reshape(x*y)

    activation_cache[activation_cache>=0] = 1
    activation_cache[activation_cache < 0] = 0
    activation_cache = activation_cache.reshape(x, y)
    return activation_cache*dA

# Backward Propagation
def linear_activation_backward(dA, cache, activation):

    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "softmax":
        dZ = softmax_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


# Updating Parameter
def update_parameter(parameters, grads, alpha):

    L = len(parameters) // 2

    for l in range(L):
        parameters["W" + str(l+1)] -= alpha * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] -= alpha * grads["db" + str(l+1)]



    return parameters


def random_mini_batches(X, Y, mini_batch_size, seed=0):

    np.random.seed(seed)  # To make your "random" minibatches the same as ours
    m = X.shape[1]  # number of training examples
    mini_batches = []

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = int(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):

        mini_batch_X = shuffled_X[:, k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size:(k + 1) * mini_batch_size]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:

        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size:m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size:m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def cal_accuracy(A2, Y):

    m = Y.shape[1]
    p = A2*np.ones((LABELS, m))
    p[ p == np.max(p, axis=0)] = 1
    p[p != np.max(p, axis=0)] = 0

    p[p != Y]=0
    val = np.sum(p)
    val /= m
    val *= 100
    # print("acu: " + str(val))
    return val

def cal_accuracy_sigmoid(A2, Y):

    m = Y.shape[1]
    p = A2*np.ones((LABELS, m))

    p = np.argmax(p, axis=0)
    y = np.argmax(Y*np.ones((LABELS, m)), axis =0)

    p[ p == y]=1
    p[p != y]=0
    val = np.sum(p)
    val /= m
    val *= 100
    # print("acu: " + str(val))
    return val



def two_layer_model(X, Y, layer_dims, alpfa, num_itertions, mini_batch = 600):

    (n_x, n_h1, n_h2, n_h3, n_y) = layer_dims
    m = X.shape[1]
    costs = []

    num_minibatches = int(m/mini_batch)

    parameters = initialize_parameter(n_x, n_h1, n_h2, n_h3, n_y)

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    W4 = parameters["W4"]
    b4 = parameters["b4"]

    for i in range(0, num_itertions):


        epoch_cost = 0
        epoch_accuracy = 0
        minibatches = random_mini_batches(X, Y, mini_batch)

        for minibatch in minibatches:

            (minibatch_X, minibatch_Y) = minibatch

            # print("Shape of MINI X: "+str(minibatch_X.shape))
            # print("Shape of MINI Y: " + str(minibatch_Y.shape))

            A1, cache1 = linear_forward(minibatch_X, W1, b1, "relu")
            A2, cache2 = linear_forward(A1, W2, b2, "relu")
            A3, cache3 = linear_forward(A2, W3, b3, "relu")
            A4, cache4 = linear_forward(A3, W4, b4, "sigmoid")

            # print("A2 after softmax \n" + str(A2))

            cost = compute_cost(A4, minibatch_Y)
            accu = cal_accuracy_sigmoid(A4, minibatch_Y)
            epoch_cost += cost
            epoch_accuracy += accu

            # print("A2 after Computation \n" + str(A2))
            # A2 = A2.reshape(LABELS, mini_batch)
            dA4 = -((minibatch_Y/A4) - (1-minibatch_Y)/(1-A4))

            dA3, dW4, db4 = linear_activation_backward(dA4, cache4, "sigmoid")
            dA2, dW3, db3 = linear_activation_backward(dA3, cache3, "relu")
            dA1, dW2, db2 = linear_activation_backward(dA2, cache2, "relu")
            dA0, dW1, db1 = linear_activation_backward(dA1, cache1, "relu")


            grads = {
                "dW1": dW1,
                "dW2": dW2,
                "dW3": dW3,
                "dW4": dW4,
                "db1": db1,
                "db2": db2,
                "db3": db3,
                "db4": db4
            }

            parameters = update_parameter(parameters, grads, alpfa)

            W1 = parameters["W1"]
            b1 = parameters["b1"]
            W2 = parameters["W2"]
            b2 = parameters["b2"]
            W3 = parameters["W3"]
            b3 = parameters["b3"]
            W4 = parameters["W4"]
            b4 = parameters["b4"]



        epoch_cost /= num_minibatches
        epoch_accuracy /= num_minibatches
        # if i % 100 == 0:
        costs.append(epoch_cost)
        print("Cost After " + str(i) + " interations: " + str(np.squeeze(epoch_cost)) + " | Accuracy: " + str(epoch_accuracy))

    # plot the cost

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(alpfa))
    plt.show()

    return parameters


if __name__ == "__main__":

    layer_dims = (train_x.shape[0], 512, 256, 128, 10)
    parameters = two_layer_model(train_x, train_y, layer_dims, 0.01, 10, 10)
    print("Completed")

