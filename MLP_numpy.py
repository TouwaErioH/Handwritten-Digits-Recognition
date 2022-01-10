import random
import numpy as np
import os


def loadData(path):
    def loadImg(path):
        img = []

        with open(path, "r") as f:
            for row in f.readlines():
                if row.strip():
                    img += [int(i) for i in row.strip()]
        return img

    files = os.listdir(path)
    img = np.zeros([len(files), 32 * 32], dtype=np.int)
    lable = np.zeros([len(files)], dtype=np.int)

    for i, fileName in enumerate(files):
        lable[i] = int(fileName.split("_")[0])

        img[i] = loadImg(path + "/" + fileName)

    return img, lable


class Network(object):
    #init network
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.zeros((y, 1)) for y in sizes[1:]] 
        self.weights = [np.random.randn(y, x) / np.sqrt(y/2)
                        for x, y in zip(sizes[:-1], sizes[1:])] 
    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for i in range(len(self.biases)-1):
            a = np.dot(self.weights[i], a).reshape(-1, 1)
            a += self.biases[i]
            a = sigmoid(a)
        a= np.dot(self.weights[-1], a)
        a += self.biases[-1]
        a = softmax(a)
        return a
    #SGD algorithm
    # https://www.zhihu.com/question/36301367
    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        if test_data: n_test = len(test_data)
        n = len(training_data)
        #epoch
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
        print("lr:{},params:{}, wrong num:{},precision: {} loss:{}".format(eta, self.sizes[1], n_test-self.evaluate(test_data),self.evaluate(test_data)/n_test * 100, self.cost(test_data)))
    #update_mini_batch.self.weights和self.biases
    def update_mini_batch(self, mini_batch, eta):
        t = [a for (a, b) in mini_batch]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        #loss
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
    #backpropagation algorithm
    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward 
        activation = x.reshape(-1, 1) 
        activations = [x] # list to store all the activations, layer by layer
        # Activation Function
        zs = [] # list to store all the z vectors, layer by layer
        for i in range(len(self.biases)-1):
            z = np.dot(self.weights[i], activation)+self.biases[i]
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        z = np.dot(self.weights[-1], activation)
        z = z + self.biases[-1]
        zs.append(z)
        activations.append(softmax(z))
        # backward pass 
        delta = self.cost_cross(activations[-1], y) #delta
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            t = activations[-l-1].reshape(1, -1)
            # print(t.shape)
            nabla_w[-l] = np.dot(delta, t)
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]

        return sum(int(x == y) for (x, y) in test_results)
    #loss function
    def cost(self, test_data):
        loss_all = 0
        for (x, y) in test_data:
            result=self.feedforward(x)
            # assert result[y]>0

            temp = float(-np.log(result[y]+0.0000001))
            loss_all += temp
        loss = loss_all/len(test_data)
        return loss

    def cost_cross(self, output_activations, y):
        Y = np.zeros((10, 1))
        Y[y] = 1
        return softmax(output_activations) - Y

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def softmax(z):
    max_value = np.max(z)
    exp = np.exp(z - max_value)
    exp_sum = np.sum(exp)
    dist = exp / exp_sum
    return dist


if __name__ == "__main__":
    #load data
    trainImg, trainLabel = loadData("./digits/trainingDigits")
    testImg, testLabel = loadData('./digits/testDigits')
    #zip：IMG和LABEL 成pair
    training_data = list(zip(trainImg[:1920], trainLabel[:1920]))
    test_data = list(zip(testImg, testLabel))

    for lr in [0.1, 0.01, 0.001, 0.0001]:
        for h in [500, 1000, 1500, 2000]:
            learning_rate = lr
            sizes = [1024, h, 10]
            dnn = Network(sizes)
            dnn.SGD(training_data, 30, 60, learning_rate, test_data=test_data)


'''
http://neuralnetworksanddeeplearning.com/chap1.html
https://blog.csdn.net/xovee/article/details/85056983?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_title~default-0.essearch_pc_relevant&spm=1001.2101.3001.4242.1

https://github.com/MichalDanielDobrzanski/DeepLearningPython
https://github.com/mnielsen/neural-networks-and-deep-learning

'''


'''
lr:0.1,params:500 precision 95.1374 loss:0.7317185185949221
lr:0.1,params:1000 precision 95.1374 loss:0.7614327488716612
lr:0.1,params:1500 precision 95.8774 loss:0.6645566194601893
lr:0.1,params:2000 precision 95.8774 loss:0.6443917787705792
lr:0.01,params:500 precision 90.5920 loss:1.0712871293164192
lr:0.01,params:1000 precision 91.7548 loss:1.139703084593655
lr:0.01,params:1500 precision 91.9662 loss:1.200202167233782
lr:0.01,params:2000 precision 91.2262 loss:1.2745205217295519
lr:0.001,params:500 precision 87.4207 loss:0.666048284729168
lr:0.001,params:1000 precision 87.9493 loss:0.9023565256062953
lr:0.001,params:1500 precision 90.0634 loss:0.9719556982663603
lr:0.001,params:2000 precision 88.1607 loss:1.2810698761339294
lr:0.0001,params:500 precision 32.4524 loss:3.815013156549688
lr:0.0001,params:1000 precision 55.4968 loss:1.9129748608848889
lr:0.0001,params:1500 precision 63.5307 loss:2.165815821522075
lr:0.0001,params:2000 precision 73.6786 loss:1.5356950763647035
'''

'''
lr:0.1,params:500, wrong num:44,precision: 95.34883720930233 loss:0.69119362245158
lr:0.1,params:1000, wrong num:40,precision: 95.77167019027483 loss:0.6557226304423373
lr:0.1,params:1500, wrong num:44,precision: 95.34883720930233 loss:0.7496960885393562
lr:0.1,params:2000, wrong num:41,precision: 95.66596194503171 loss:0.6985594761316214
lr:0.01,params:500, wrong num:84,precision: 91.12050739957716 loss:0.9986526826360357
lr:0.01,params:1000, wrong num:86,precision: 90.9090909090909 loss:1.2004809043024656
lr:0.01,params:1500, wrong num:82,precision: 91.33192389006342 loss:1.1811307741741575
lr:0.01,params:2000, wrong num:83,precision: 91.2262156448203 loss:1.2754621068009677
lr:0.001,params:500, wrong num:135,precision: 85.7293868921776 loss:0.6276128833355958
lr:0.001,params:1000, wrong num:140,precision: 85.20084566596195 loss:1.1813047254615876
lr:0.001,params:1500, wrong num:106,precision: 88.79492600422833 loss:0.9966538320197375
lr:0.001,params:2000, wrong num:111,precision: 88.26638477801268 loss:1.1875503635886568
lr:0.0001,params:500, wrong num:717,precision: 24.207188160676534 loss:3.3413164682487517
lr:0.0001,params:1000, wrong num:677,precision: 28.43551797040169 loss:6.843246637457159
lr:0.0001,params:1500, wrong num:360,precision: 61.94503171247357 loss:2.078118673193193
lr:0.0001,params:2000, wrong num:323,precision: 65.85623678646935 loss:1.825134652917873
'''
