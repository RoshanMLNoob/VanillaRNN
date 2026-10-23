import numpy as np
import pickle

class S2S_RNN(object):

    def __init__(self , inp_r , s_len , vocab_size , opt_r=None , factor=0.01):

        if opt_r is None:
            op = inp_r

        self.ip , self.op = inp_r , op
        self.s_r = s_len

        #State transmitters
        self.Wsx = np.array(np.random.randn(self.s_r , self.ip) , ndmin=2)*factor
        self.Wss = np.array(np.random.randn(self.s_r , self.s_r) , ndmin=2)*factor
        self.bs = np.array(np.zeros((self.s_r , 1)) , ndmin=2).reshape(-1,1)

        #st = f1(Wsx@xt + Wss@st-1 + bs)

        #output generators
        self.Wo = np.array(np.random.randn(vocab_size , self.s_r) , ndmin=2)*factor
        self.bo = np.array(np.zeros((vocab_size)) , ndmin=2)

        #zt = Wo@st + bo --> yt = softmax(zt)

    def softmax(self , distribution):

        shifted = distribution - np.max(distribution)
        shifted = np.clip(shifted , -500 , 500)
        exps = np.exp(shifted)
        min_subbed = np.sum(exps , axis=0 , keepdims=True)

        return exps / (min_subbed + 1e-15)
    def same(self , z):
        return z

    def tanh(self , z):
        return np.tanh(z)
    def ReLU(self , z):
        return np.maximum(0 , z)
    def sigmoid(self , z):
        return 1/(1+np.exp(-z))

    def dReLU(self , z):
        return np.where(z>0 , 1 , 0)
    def dtanh(self , z):
        return 1-(self.tanh(z)**2)
    def dSigmoid(self , z):
        S = (self.sigmoid(z))
        return S*(1-S)
        