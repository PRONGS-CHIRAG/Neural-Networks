import numpy as np
def sigmoid(x):
    return (1/(1+np.exp(-x)))
def sigmoid_derivative(x):
    a = sigmoid(x)
    return a*(1-a)
def mse_loss(y_true,y_pred):
    return ((y_true-y_pred)**2).mean()
class NeuralNetwork():
    def __init__(self):
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()
    def feedforward(self,x):
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        h3 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        return h3
    def train(self,data,all_y_trues):
        learn_rate = 0.1
        epochs = 50000
        for epoch in range(epochs):
            for x,y_true in zip(data,all_y_trues):
                sh1= self.w1 * x[0] + self.w2 * x[1] +self.b1
                h1 = sigmoid(sh1)
                sh2= self.w3 * x[0] + self.w4 * x[1] +self.b2
                h2 = sigmoid(sh2)
                sh3= self.w5 * h1 + self.w6 * h2 +self.b3
                h3 = sigmoid(sh1)
                y_pred = h3
                # calculating derivatives now
                L_ypred = -2 * (y_true-y_pred)
                #neuron h1
                h1_w1 = x[0] * sigmoid_derivative(sh1)
                h1_w2 = x[1] * sigmoid_derivative(sh1)
                h1_b1 = sigmoid_derivative(sh1)
                #neuron h2
                h2_w3 = x[0] * sigmoid_derivative(sh2)
                h2_w4 = x[1] * sigmoid_derivative(sh2)
                h2_b2 = sigmoid_derivative(sh2)
                #neuron h3
                h3_w5 = h1 * sigmoid_derivative(sh3)
                h3_w6 = h2 * sigmoid_derivative(sh3)
                h3_b3 = sigmoid_derivative(sh3)
                h3_h1 = self.w5 * sigmoid_derivative(sh3)
                h3_h2 = self.w6 * sigmoid_derivative(sh3)
                #update weights and bias
                #neuron h1
                self.w1 = learn_rate * L_ypred * h1_w1
                self.w2 = learn_rate * L_ypred * h1_w2
                self.b1 = learn_rate * L_ypred * h1_b1
                #neuron h2
                self.w3 = learn_rate * L_ypred * h2_w3
                self.w4 = learn_rate * L_ypred * h2_w4
                self.b2 = learn_rate * L_ypred * h2_b2
                #neuron h3
                self.w5 = learn_rate * L_ypred * h3_w5
                self.w6 = learn_rate * L_ypred * h3_w6
                self.b3 = learn_rate * L_ypred * h3_b3
            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.feedforward, 1, data)
                loss = mse_loss(all_y_trues, y_preds)
                print("Epoch %d loss: %.3f" % (epoch, loss))
data = np.array([[-2, -1],  [25, 6],[17, 4],[-15, -6]])
all_y_trues = np.array([1,0,0,1])
network = NeuralNetwork()
network.train(data,all_y_trues)
