from enum import Enum
from typing import Callable

import math
import numpy as np

# region Activation

class Activation_Softmax:
    """
    Softmax funtion is always applied as the last activation function in a network
    The derivitive is included in Loss_CCE backward calculation
    """
    def forward(self, inputs):
        """
        takes an array as argument
        """
        exp_values = np.exp(inputs - np.max(inputs, axis=1,
                                            keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1,
                                            keepdims=True)
        return probabilities

    def backward(self, x):
        raise NotImplementedError

class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        return np.maximum(0, inputs)

    def backward(self, dvalues):
        inputs = dvalues.copy()
        inputs[self.inputs <= 0] = 0
        return inputs

class Activation_Sigmoid:
    def forward(self, X):
        self.inputs = X
        return 1 / (1 + np.exp(-self.inputs))

    def backward(self, X):
        return self.inputs * self.forward(X) * (1- self.forward(X))
    
class Activation_Tanh:
    def forward(self, X):
        return np.tanh(X)
    
    def backward(self, X):
        return (1 - np.tanh(X)**2)



# endregion Activation

# region Loss

class Loss:
    def __init__(self):
        self.reset_accumulation()

    def reset_accumulation(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0
    
    def calculate(self, batch_outputs: np.ndarray, batch_targets: np.ndarray):
        batch_loss = self.forward(batch_outputs, batch_targets)
        # print(batch_loss)
        # print(np.mean(batch_loss))
        self.accumulated_sum += np.sum(batch_loss)
        self.accumulated_count += len(batch_loss)
        return np.mean(batch_loss)

    def calculate_accumulated(self):
        accumulated_loss = self.accumulated_sum / self.accumulated_count
        self.reset_accumulation()
        return accumulated_loss
        

class Loss_CCE(Loss):
    """
    Forward and backward pass of the Categorical Cross-Entropy loss function. Backward pass
    also includes the derivative of the Soft-Max function.
    """
    def forward(self, batch_outputs: np.ndarray, batch_targets: np.ndarray):
        batch_outputs = np.clip(batch_outputs, 1e-7, 1-1e-7)
        confidences = np.sum(batch_outputs*batch_targets, axis=1)
        losses = -np.log(confidences)
        return losses

    def backward(self, batch_outputs: np.ndarray, batch_targets: np.ndarray):
        num_of_samples = len(batch_outputs)
        backprop_input = batch_outputs.copy()
        backprop_input[range(num_of_samples), np.argmax(batch_targets, axis=1)] -= 1
        return backprop_input / num_of_samples

# endregion Loss

# region Accuracy

class Accuracy:
    def __init__(self):
        self.reset_accumulation()

    def reset_accumulation(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

    def calculate(self, batch_outputs: np.ndarray, batch_targets: np.ndarray):
        comparisons = self.compare(batch_outputs, batch_targets)
        accuracy = np.mean(comparisons)
        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)
        return accuracy

    def calculate_accumulated(self):
        accumulated_accuracy = self.accumulated_sum / self.accumulated_count
        self.reset_accumulation()
        return accumulated_accuracy

class Accuracy_Categorical(Accuracy):

    def compare(self, batch_outputs: np.ndarray, batch_targets: np.ndarray):
        predictions = np.argmax(batch_outputs, axis=1)
        if len(batch_targets.shape) == 2:
            batch_targets = np.argmax(batch_targets, axis=1)
        return predictions == batch_targets

# endregion Accuracy

# region Optimizer

class Optimzer:
    """
    A class all optimziers should enherit. Class must not overide all functions.
    The functions are there to prevent error incase optimzer has no need for it.
    All the functions are called in nework.backward().
    ----------------------------------------------------------------------------
    (pre_update) is called once per backpropagation before apply_traing. 
    (apply_training) is called once per layer per backpropagation
    (post_update) is called once per backpropagation after apply_traing.
    """
    def pre_update(self):
        pass

    def apply_training(self):
        pass

    def post_update(self):
        pass

class Optimizer_SGD(Optimzer):
    """
    Stochastic gradient decent - the most simple optimizer algorithm
    """
    def __init__(self, *, learn_rate, decay): # ALLA OPTIMZIERS MÅSTE HA "*"" EFTER SELF EFTERSOM DET MÅSTE VARA KWARGS. VÄNLIGET KONTAKTA CASPER FÖR MER INFORMATION OM VARFÖR DETTA ÄR. MVH. MR. BENÉ
        self.learn_rate = learn_rate
        self.current_learn_rate = learn_rate
        self.decay = decay
        self.iterations = 0

    def pre_update(self):
        if self.decay > 0:
            self.current_learn_rate = self.learn_rate * (1 / (1 + self.decay * self.iterations))
        self.iterations += 1

    def apply_training(self, layer):
        layer.weights -= self.learn_rate * layer.dweights
        layer.biases -= self.learn_rate * layer.dbiases


class Optimizer_Adam(Optimzer):
    
    def __init__(self, *, learn_rate, decay, epsilon, beta_1, beta_2):
        self.learn_rate = learn_rate
        self.current_learn_rate = learn_rate
        self.decay = decay
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

        self.t = 0

    def pre_update(self):
        if self.decay:
            self.current_learn_rate = self.learn_rate * (1 / (1 + self.decay * self.t))

    def apply_training(self, layer):
        """ Algorithm from 3.2.3 "Nätverkets inlärningsprocess" under rubriken "Vikter och biaser ändras" """

        # init momentum vectors for weights and biases
        # only called once per layer
        if not hasattr(layer, 'v_dw'):
            # 1st momentum vector
            layer.v_dw = np.zeros_like(layer.weights) # samma som np.zeros(layer.weights.shape)
            layer.v_db = np.zeros_like(layer.biases)
            # 2nd momentum vector
            layer.s_dw = np.zeros_like(layer.weights)
            layer.s_db = np.zeros_like(layer.biases)
        
        # 1st momentum vector using beta_1. m_t = beta_1 * m_(t-1) + (1-beta_1) * g
        layer.v_dw = self.beta_1 * layer.v_dw + (1 - self.beta_1) * layer.dweights
        layer.v_db = self.beta_1 * layer.v_db + (1 - self.beta_1) * layer.dbiases
        
        # g^2 indicates elementwise square.
        # 2nd momentum vector using beta_2. v_t = beta_2 * v_(t-1) + (1-beta_2) * g^2
        layer.s_dw = self.beta_2 * layer.s_dw + (1 - self.beta_2) * layer.dweights**2 # samma som np.square
        layer.s_db = self.beta_2 * layer.s_db + (1 - self.beta_2) * layer.dbiases**2 # samma som np.square

        # 1st moment vector correction. m_t_hat = m_t / (1 - beta_1^t)
        v_dw_corrected = layer.v_dw / (1 - self.beta_1 ** (self.t + 1)) # t + 1 efterson t börjar på 0
        v_db_corrected = layer.v_db / (1 - self.beta_1 ** (self.t + 1))

        # 2nd moment vector correction. v_t_hat = v_t / (1 - beta_2^t)
        s_dw_corrected = layer.s_dw / (1 - self.beta_2 ** (self.t + 1))
        s_db_corrected = layer.s_db / (1 - self.beta_2 ** (self.t + 1))
        
        # update params. theta_t = theta_(t-1) - alpha * m_t_hat / (sqrt(v_t_hat) + epsilon)
        layer.weights -= self.current_learn_rate * v_dw_corrected / (np.sqrt(s_dw_corrected) + self.epsilon)
        layer.biases -= self.current_learn_rate * v_db_corrected / (np.sqrt(s_db_corrected) + self.epsilon)

    def post_update(self):
        self.t += 1

# endregion Optimizer