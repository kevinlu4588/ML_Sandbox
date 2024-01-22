#!/Users/KevinLu/Downloads/AI_PA04/venv/bin/python
import sys
import numpy as np
import matplotlib.pyplot as plt
import math
import util

class Model:
    """
    Abstract class for a machine learning model.
    """
    
    def get_features(self, x_input):
        pass

    def get_weights(self):
        pass

    def hypothesis(self, x):
        pass

    def predict(self, x):
        pass

    def loss(self, x, y):
        pass

    def gradient(self, x, y):
        pass

    def train(self, dataset):
        pass


# PA4 Q1
class PolynomialRegressionModel(Model):
    """
    Linear regression model with polynomial features (powers of x up to specified degree).
    x and y are real numbers. The goal is to fit y = hypothesis(x).
    """

    def __init__(self, degree = 1, learning_rate = 1e-3):
        "*** YOUR CODE HERE ***"
        self.degree = degree
        self.learning_rate = learning_rate
        self.weights = np.zeros(degree + 1)
        self.losses = []
        self.numIterations = 1000
    def get_features(self, x):
        features = []
        for i in range(self.degree + 1):
            features.append(math.pow(x,i))
        return features

    def get_weights(self):
        return self.weights

    def hypothesis(self, x):
        guess = 0
        features = self.get_features(x)
        for i in range(len(self.weights)):
            guess+= self.weights[i] * features[i]
        return guess

    def predict(self, x):
        return self.hypothesis(x)

    def loss(self, x, y):
        features = self.get_features(x)
        return np.power((y - np.dot(self.weights, features)), 2)

    def gradient(self, x, y):
        features = self.get_features(x)
        gradient = -2 * (y - np.dot(self.weights, features)) * np.array(features)
        return gradient
    
    def train(self, dataset, evalset=None):
        #Data shuffling
        datasetX = dataset.xs
        datasetY = dataset.ys
        temp = list(zip(datasetX, datasetY))
        np.random.shuffle(temp)
        res1, res2 = zip(*temp)
        dataset.xs, dataset.ys = list(res1), list(res2)
        for iteration in range(self.numIterations):
            total_loss = 0
            #print("weights: ", self.weights)
            for i in range(dataset.get_size()):
                x, y = dataset.xs[i], dataset.ys[i]
                gradients = self.gradient(x, y)
                self.weights -= self.learning_rate * gradients
                total_loss += self.loss(x,y) 
            self.losses.append(total_loss/self.numIterations)   

# PA4 Q2
def linear_regression():

    #A)
    # Load the sine train dataset
    sine_train = util.get_dataset("sine_train")
    # Create a PolynomialRegressionModel with degree 1 and learning rate 10^(-4)
    model = PolynomialRegressionModel(1, 1e-4)
    model.train(sine_train)

    # Report the final hypothesis and average loss
    final_hypothesis = model.get_weights()
    average_loss = sine_train.compute_average_loss(model)

    print("Final Hypothesis:", final_hypothesis)
    print("Average Loss on Training Dataset:", average_loss)

    #  # Plot the data and the hypothesis
    # sine_train.plot_data(model=model)

    #  #b)
    # sine_train.plot_loss_curve(range(model.numIterations), model.losses, "Loss Curve")

    #c)

    sine_val = util.get_dataset("sine_val")
    models = []
    combo1 = PolynomialRegressionModel(1, 1e-4)
    combo2 = PolynomialRegressionModel(1, 1e-3)
    combo3 = PolynomialRegressionModel(1, 1e-2)
    combo4 = PolynomialRegressionModel(2, 1e-4)
    combo5 = PolynomialRegressionModel(2, 1e-5)
    combo6 = PolynomialRegressionModel(2, 1e-6)
    combo7 = PolynomialRegressionModel(3, 1e-5)
    combo8 = PolynomialRegressionModel(3, 1e-6)
    combo9 = PolynomialRegressionModel(3, 1e-7)
    combo10 = PolynomialRegressionModel(4, 1e-7)
    models.append(combo1)
    models.append(combo2)
    models.append(combo3)
    models.append(combo4)
    models.append(combo5)
    models.append(combo6)
    models.append(combo7)
    models.append(combo8)
    models.append(combo9)
    models.append(combo10)
    for i in range(len(models)):
        linear_model = models[i]
        linear_model.train(sine_train)
        final_hypothesis = linear_model.get_weights()
        average_loss = sine_train.compute_average_loss(linear_model)
        combo_name = "Combo " + str(i) + " "
        validation_loss = sine_val.compute_average_loss(linear_model)
        print(combo_name + "Final Hypothesis:", final_hypothesis)
        print(combo_name + "Average Loss on Training Dataset:", average_loss)
        print(combo_name + "Validation Loss: ", validation_loss)
    # "*** YOUR CODE HERE ***"
    # Examples
    # sine_train = util.get_dataset("sine_train")
    # sine_val = util.get_dataset("sine_val")
    # sine_model = PolynomialRegressionModel()
    # sine_model.train(sine_train)

# PA4 Q3
class BinaryLogisticRegressionModel(Model):
    """
    Binary logistic regression model with image-pixel features (num_features = image size, e.g., 28x28 = 784 for MNIST).
    x is a 2-D image, represented as a list of lists (28x28 for MNIST). y is either 0 or 1.
    The goal is to fit P(y = 1 | x) = hypothesis(x), and to make a 0/1 prediction using the hypothesis.
    """

    def __init__(self, num_features, learning_rate = 1e-2):
        "*** YOUR CODE HERE ***"
        self.num_features = num_features
        self.learning_rate = learning_rate
        self.weights = np.zeros(num_features + 1)
        self.num_iterations = 100
        self.train_accuracy = []
        self.test_accuracy = []
    def get_features(self, x):
        return np.concatenate([np.array(x).flatten(), [1]])

    def get_weights(self):
        return self.weights
    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))
    def hypothesis(self, x):
        z = np.dot(self.weights, self.get_features(x))  # Insert bias term
        return self.sigmoid(z)
        "*** YOUR CODE HERE ***"

    def predict(self, x):
        if(self.hypothesis(x) >= 0.5): 
            return 1
        else:
            return 0
        "*** YOUR CODE HERE ***"

    def loss(self, x, y):
        h = self.hypothesis(x)
        return -y * np.log(h) - (1-y) * np.log(1-h)
        "*** YOUR CODE HERE ***"

    def gradient(self, x, y):
        h = self.hypothesis(x)
        error = h - y
        gradients = error * self.get_features(x)
        return gradients
        "*** YOUR CODE HERE ***"

    def train(self, dataset, evalset = None):
        "*** YOUR CODE HERE ***"
        for iteration in range(self.num_iterations):
            if(iteration % 10 == 0):
                print("Iteration #: ", iteration)
                #train_accuracy = dataset.compute_average_accuracy(self)
                test_accuracy = evalset.compute_average_accuracy(self)
                #print("train accuracy", train_accuracy)
                #self.train_accuracy.append(train_accuracy)
                self.test_accuracy.append(test_accuracy)
            #print("weights: ", self.weights)
            for i in range(dataset.get_size()):
                x, y = dataset.xs[i], dataset.ys[i]
                gradients = self.gradient(x, y)
                self.weights -= self.learning_rate * gradients
                # if(self.predict(x) != y):
                #     features = self.get_features(x)[:-1]
                #     im = np.reshape(features, (28,28))
                #     plt.imshow(im)
                #     plt.colorbar()
                #     plt.show()

            

# PA4 Q4
def binary_classification():
    #load the mnist dataset
    binary_mnist = util.get_dataset("mnist_binary_train")
    slice_index = binary_mnist.get_size() // 5
    #binary_mnist.xs = binary_mnist.xs[:slice_index]
    #binary_mnist.ys = binary_mnist.ys[:slice_index]

    binary_mnist_eval = util.get_dataset("mnist_binary_test")
    slice_index_eval = binary_mnist_eval.get_size() // 5
    #binary_mnist_eval.xs = binary_mnist_eval.xs[:slice_index_eval]
    #binary_mnist_eval.ys = binary_mnist_eval.ys[:slice_index_eval]
    binary_model = BinaryLogisticRegressionModel(28*28, 1e-2)
    binary_model.train(binary_mnist, binary_mnist_eval)

    #binary_mnist.plot_accuracy_curve(range(10, 501, 10), binary_model.train_accuracy, "Train Accuracy Curve")
    #binary_mnist_eval.plot_accuracy_curve(range(10, 501, 10), binary_model.test_accuracy, "Test Accuracy Curve")
    #b) Plotting the confusion matrix
    binary_mnist.plot_confusion_matrix(binary_model)

    #c) Plotting 
    weights = binary_model.get_weights()
    binary_mnist.plot_image(weights[:-1])
# PA4 Q5
class MultiLogisticRegressionModel(Model):
    """
    Multinomial logistic regression model with image-pixel features (num_features = image size, e.g., 28x28 = 784 for MNIST).
    x is a 2-D image, represented as a list of lists (28x28 for MNIST). y is an integer between 1 and num_classes.
    The goal is to fit P(y = k | x) = hypothesis(x)[k], where hypothesis is a discrete distribution (list of probabilities)
    over the K classes, and to make a class prediction using the hypothesis.
    """

    def __init__(self, num_features, num_classes, learning_rate = 1e-2):
        self.num_features = num_features
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.num_iterations = 500
        self.train_accuracy = []
        self.test_accuracy = []
    # Initialize weights and biases
        self.weights = np.zeros((num_classes, num_features))
        self.biases = np.zeros(num_classes)

    def get_features(self, x):
        return np.array(x).flatten()

    def get_weights(self):
        return self.weights, self.biases

    def hypothesis(self, x):
        # Calculate probabilities using softmax function
        logits = np.dot(self.weights, self.get_features(x)) + self.biases
        exp_logits = np.exp(logits - np.max(logits))  # for numerical stability
        probabilities = exp_logits / np.sum(exp_logits)
        return probabilities.tolist()

    def predict(self, x):
        probabilities = self.hypothesis(x)
        return np.argmax(probabilities)  # Add 1 to make predictions in the range [1, K]

    def loss(self, x, y):
        probabilities = self.hypothesis(x)
        return -np.log(probabilities[y])  # y is in the range [1, K]

    def gradient(self, x, y):
        probabilities = self.hypothesis(x)
        error = probabilities.copy()
        error[y] -= 1  # y is in the range [1, K]
        dw = np.outer(error, x)
        db = np.array(error)
        return dw, db
    
    def train(self, dataset, evalset = None):
        for iteration in range(self.num_iterations):
            if(iteration % 10 == 0):
                print("Iteration #: ", iteration)
                train_accuracy = dataset.compute_average_accuracy(self)
                test_accuracy = evalset.compute_average_accuracy(self)
                print("train accuracy", train_accuracy)
                
                self.train_accuracy.append(train_accuracy)
                self.test_accuracy.append(test_accuracy)
            for i in range(dataset.get_size()):
                x, y = dataset.xs[i], dataset.ys[i]
                    
                dw, db = self.gradient(x, y)
                #print("type dw: ", type(dw))
                #print("type learning:" , self.learning_rate)
                self.weights -= self.learning_rate * dw
                self.biases -= self.learning_rate * db

# PA4 Q6
def multi_classification():
#load the mnist dataset
    mnist_train = util.get_dataset("mnist_train")
    slice_index = mnist_train.get_size() // 5
    mnist_train.xs = mnist_train.xs[:slice_index]
    mnist_train.ys = mnist_train.ys[:slice_index]

    mnist_test = util.get_dataset("mnist_test")
    slice_index_eval = mnist_test.get_size() // 5
    mnist_test.xs = mnist_test.xs[:slice_index_eval]
    mnist_test.ys = mnist_test.ys[:slice_index_eval]

    multi_model = MultiLogisticRegressionModel(28*28, 10, 1e-4)
    multi_model.train(mnist_test, mnist_test)

    
    mnist_train.plot_accuracy_curve(range(10, 101, 10), multi_model.train_accuracy, "Train Accuracy Curve")
    mnist_test.plot_accuracy_curve(range(10, 101, 10), multi_model.test_accuracy, "Test Accuracy Curve")
    #b) Plotting the confusion matrix
    mnist_test.plot_confusion_matrix(multi_model)


    #c) Plotting 
    weights, biases = multi_model.get_weights()
    #print('numrows', len(weights))
    #print('numcols', len(input[0]))
    mnist_test.plot_image(weights)
def main():
    #linear_regression()
    binary_classification()
    multi_classification()

if __name__ == "__main__":
    main()
