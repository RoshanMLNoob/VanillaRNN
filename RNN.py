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
        self.bo = np.array(np.zeros((vocab_size , 1)) , ndmin=2)

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

    def forward_pass(self , X , Y=None , s0=None , f1_0="relu" , f2_0="softmax" , loss=None):

        func_map1 = {"relu":self.ReLU , "tanh":self.tanh , "sigmoid":self.sigmoid}
        func_map2 = {"softmax":self.softmax , "same":self.same , "linear":self.same}
        loss_dict = {"cross entropy":self.Loss_CE , "mse":self.Loss_MSE , "squared":self.Loss_MSE}

        self.f1 , self.f2 = func_map1.get(f1_0 , self.ReLU) , func_map2.get(f2_0 , self.softmax)

        #Sooo X is a sequence X = {x1 , x2 , x3 , x4 ..... xT}

        if s0 is None:
            self.s = np.array(np.zeros((self.s_r , 1)) , ndmin=2) #s0
        else:
            self.s = np.array(s0 , ndmin=2).reshape(self.s_r , 1)

        self.P = []
        self.states = [self.s.copy()]
        self.Zs = []
        self.hidden_Zs = []
        self.losses_over_t = []
        self.Loss_f = loss_dict.get(loss , None)

        for t , x_t in enumerate(list(X)):

            x_t = np.array(x_t , ndmin=2).reshape(-1,1)

            #Hidden Pre Activation
            z_hidden = self.Wsx@x_t + self.Wss@self.s + self.bs
            self.hidden_Zs.append(z_hidden)

            #Now calculation transmitting sequence
            self.s = self.f1(self.Wsx@x_t + self.Wss@self.s + self.bs) #Using Recurrence Formula

            self.states.append(self.s.copy())

            #Calculating the output
            self.z = self.Wo@self.s + self.bo #This is z_t

            self.P.append(self.f2(self.z)) #This is y_t being appended to output sequence Y
            self.Zs.append(self.z)

            if self.Loss_f is not None and Y is not None:
                self.losses_over_t.append(np.sum(self.Loss_f(self.P[t],Y[t])))

        return self.P
    
    def Loss_CE(self , p , y):
        p = np.clip(p , 1e-15 , 1)
        return -y*np.log(p)
    def Loss_MSE(self , p , y):
        return (y-p)**2

    def dLoss_CE(self , p , y):
        return -y/p
    def dLoss_MSE(self , p , y):
        return 2*(y-p)
    
    def One_Hot(self , ind , l):
        arr = np.zeros((l , 1))
        arr[0 , ind]  = 1
        return arr

    def backward_pass(self , X , Y , for_pass=True , f1="relu" , f2="softmax" , loss="cross entropy" , s0=None , one_hot=False):

        deriv_map = {self.ReLU : self.dReLU , self.sigmoid:self.dSigmoid , self.tanh:self.dtanh}

        if for_pass:

            self.forward_pass(X=X , Y=Y , s0=s0 , f1_0=f1 , f2_0=f2 , loss=loss)
            self._loss_ = np.mean(self.losses_over_t)

        df1 = deriv_map[self.f1]

        if loss.lower() in ["cross entropy" , "nll" , "mse"]: #Initializing of weights

            eval_steps = min(len(self.P) , len(Y))

            if not one_hot:

                delta_0 = [self.P[t] - np.array(Y[t] , ndmin=2).reshape(-1,1) for t in range(eval_steps)] #idk why this comment is green
            
            else:

                delta_0 = [self.P[t] - self.One_Hot( ind=Y[t] , l=len(self.P[t]) ) for t in range(eval_steps)]

        else:
            raise ValueError(f"Unsupported function type : {loss}")

        
        #Initialization ARC

        dWo , dbo= np.zeros_like(self.Wo) , np.zeros_like(self.bo) #Initialization for the w,b for output func

        dWss , dWsx , dbs = np.zeros_like(self.Wss) , np.zeros_like(self.Wsx) , np.zeros_like(self.bs) #Initialization of the w,b for the recurrence formula

        ds_1 = np.zeros( (self.s_r , 1) )

        T = len(X)

        for t in range(T-1 , -1 , -1):

            x_t = np.array(X[t] , ndmin=2).reshape(-1,1)

            delta_o = delta_0[t].copy()

            dWo += delta_o@self.states[t+1].T
            dbo += delta_o.copy()

            ds = self.Wo.T@delta_o + ds_1

            s = self.states[t+1]
            z = self.hidden_Zs[t]
            delta_s = ds*df1(z)

            dWsx += delta_s @ x_t.T
            dWss += delta_s @ self.states[t].T
            dbs += delta_s.copy()
            
            ds_1 = self.Wss.T @ delta_s

        self.dWo , self.dbo = dWo.copy() , dbo.copy()
        self.dWss , self.dWsx , self.dbs = dWss.copy() , dWsx.copy() , dbs.copy()

        return (self.dWo , self.dbo , self.dWss , self.dWsx , self.dbs) , self._loss_

    def Train_RNN(self , X , Y , epoch=5 , iterations=10000 , lr = lambda t: 0.01 , f1 = "relu" , f2 = "same" , loss = "mse" , s0=None , one_hot=False , stocastic=False , interval=100 , for_pass=True):

        if stocastic:

            for ep in range(epoch):
                for t in range(1 , iterations+1):

                    k = np.random.choice(len(X))
                    T = ep*iterations + t

                    derivs , _loss_ = self.backward_pass(X[k] , Y[k] , for_pass=for_pass , f1=f1 , f2=f2 , loss=loss , s0=s0 ,one_hot=one_hot)

                    self.Wo , self.bo = self.Wo - (lr(T)*(derivs[0])) , self.bo - (lr(T)*(derivs[1]))
                    self.Wss , self.Wsx , self.bs = self.Wss - (lr(T)*(derivs[2])) , self.Wsx - (lr(T)*(derivs[3])) , self.bs - (lr(T)*(derivs[4]))

                    if t%interval == 0:
                        print(f"Loss at {t} : {_loss_}")

            return True

        elif not stocastic:

            for ep in range(1,1+epoch):
                for k in range(len(X)):
                    t = k

                    derivs , _loss_ = self.backward_pass(X[k] , Y[k] , for_pass=for_pass , f1=f1 , f2=f2 , loss=loss , s0=s0 , one_hot=one_hot)

                    self.Wo , self.bo = self.Wo - (lr(T)*(derivs[0])) , self.bo - (lr(T)*(derivs[1]))
                    self.Wss , self.Wsx , self.bs = self.Wss - (lr(T)*(derivs[2])) , self.Wsx - (lr(T)*(derivs[3])) , self.bs - (lr(T)*(derivs[4]))

                    if t%interval == 0:
                        print(f"Loss at {t} : {_loss_}")

            return True

    def save_model(self , file):

        Data = (self.Wo , self.bo , self.Wsx , self.Wss , self.bs)

        with open(file , "wb") as f:
            pickle.dump(Data , f)

        return True

    def load_model(self , file):

        with open(file , "rb") as f:
            Data = pickle.load(f)

        self.Wo , self.bo , self.Wsx , self.Wss , self.bs = Data

        return True