

'Implementation of Koopman operator with Fourier feature and testing it with acasxu data samples'


import os
import sys
import copy
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


class KOOPMAN:

    def __init__(self, sim_states, sim_cmds):

        if len(sim_states) == 0:
            print("Empty training set.")
            return 

        # states, cmds = self.preprocessing(sim_states, sim_cmds)

        n_sim_states, n_sim_cmds = self.preprocessing(sim_states, sim_cmds)
        # n_sim_states, n_sim_cmds = sim_states, sim_cmds
        #states, cmds = sim_states, sim_cmds
  
        rate = 0.8  # rate of train data
        train_size = int(np.round(len(n_sim_states) * rate)) # train dataset size
        #print("Koopman: train size: ", train_size)
        #print("train_size ", train_size)
        test_states = n_sim_states[train_size:] # test
        test_cmds = n_sim_cmds[train_size:] # test
        states = n_sim_states[0: train_size] # train
        cmds = n_sim_cmds[0: train_size] # train
    
        #print("states: ", len(states))
        #print("cmds: ", len(cmds))
 
        'Generate Fourier matrices for count number of observables'
        self.num_observables = 10 #200
        rank = 10 #self.num_observables
        X = [0, 0, 0] # we consider 3 states speed, rpm, gear for making the model
        XP = []
        self.WF, self.BF = self.generate_Fourier(len(X), self.num_observables)


        'making input from states and observables for DMD'     
        for s in states:
            xps = self.generate_xp(s)
            XP.append(xps)

   
        self.A, self.B = self.DMD(XP, cmds, rank) # Koopman operators

        print("max A: ", np.max(self.A))
        print("min A: ", np.min(self.A))

        #print("A: ", self.A[0, :])
        #print("B: ", self.B)

        'Test Koopman'
        # for i in range(len(states)):
        #     print(i)
        #     self.Test(test_states[i], test_cmds[i])
        #     # self.Test(states[i], cmds[i]) #sanity check




    def generate_Fourier(self, n, count):
    
        """Generating a set of Wi and bi for Fourier features
           to be used in obsevables g()"""

        'n: numbeer of features in a state (x,y,t)'

        print("koopman generate_Fourier")
        lw = []
        lb = []
        l = 1 # For test it should be 1
        np.random.seed(0)
 
        for i in range(count):
            WT = stats.norm.rvs(loc=0, scale=l, size=n)
            b = stats.uniform.rvs(loc=0, scale=2*np.pi, size=1)
            lw.append(WT)
            lb.append(b)
        return lw, lb
       


    def get_Fourier_matrices(self):
        return self.WF , self.BF



    def g(self, X, WT, b):

        """creating observables g1(x), g2(x), ..., gn(x). 
           We generate them using Fourier feature
           g(X) = cos(wT*X + b)"""

        # print(f" g() WT.shape {WT}  X.shep {X}  b.shape {b}")

        out = np.cos(np.dot(WT, X) + b)
        return out



    def DMD(self, X, U, rank):

        'Dynamic Mode Decomposition'

        print("koopman: DMD")

        tmp = X[0]
        X1 = tmp[:,0:tmp.shape[1] -1]
        X2 = tmp[:,1:tmp.shape[1]]

        for i in range(1, len(X)):
            tmp = np.array(X[i])
            X1 = np.concatenate((X1,tmp[:, 0:tmp.shape[1] - 1]), axis=1)
            X2 = np.concatenate((X2,tmp[:, 1:tmp.shape[1]]), axis=1)

        # U_ = np.array([U[0]])
        U_ = np.array(U[0])

        for i in range(1, len(U)):
            # U_ = np.concatenate((U_, np.array([U[i]])), axis=1)
            U_ = np.concatenate((U_, np.array(U[i])), axis=1)


        X1 = np.concatenate((X1, U_), axis=0)

        #singular value decomposition
        V,S,W = np.linalg.svd(X1, full_matrices=False)
        
        #reduce rank by removing the smallest singular values
        V = V[:, 0:rank]
        S = S[0:rank] 
        W = W[0:rank, :]
    
        AA = np.linalg.multi_dot((X2, np.transpose(W), np.diag(np.divide(1, S)), np.transpose(V)))
    
    
        #devide into state matrix and input matrix

        B = AA[:, AA.shape[0] : AA.shape[1]]
        A = AA[:, 0: AA.shape[0]]
    
        return A, B



    def get_DMD_matrices(self):
    
        return self.A, self.B
        


    def generate_xp(self, s):

        out = np.zeros((len(self.WF), s.shape[1]))
        for i in range(len(self.WF)): # iterating thorough each wi
            for r in range(s.shape[1]):
                # print(f"generate_xp_2 s[:,r].shep {s[:,r].shape}")
                # print()
                out[i, r] = self.g(s[:, r], self.WF[i], self.BF[i])

        res = np.concatenate((s, out))

        return res
    


    def predict(self, xs, us):

        g_xs = self.generate_xp(xs)
        # print(f"g_xs {g_xs.shape}   self.A {self.A.shape}  self.B {self.B.shape}  us {us.shape}")
        out = np.dot(self.A, g_xs) + np.multiply(self.B, us)

        #print("np.dot(self.A, g_xs) ", np.dot(self.A, g_xs)[0:3], " np.multiply(self.B, us) ", np.multiply(self.B, us)[0:3])
        #print("A.shape ", self.A.shape, "g_xs.shape: ", g_xs.shape )
        #print("B.shape ", self.B.shape, " us ", us)

        return out
   


    def preprocessing(self, sim_states, sim_cmds):
  
        'Normalizing and scalling the dataset'
        print("koopman: preprocessing...")

        # print(f"sim_cmds before {sim_cmds}")

        'concating all arrays in sim_states column-wise to get the norm' 
        tmp_s = np.zeros((sim_states[0].shape[0],1))
        # print(f"koopman preprocessing sim_states[0].shape {sim_states[0].shape}")
        for a in sim_states:
            for j in range(a.shape[1]):
                x = np.transpose(np.array([a[:,j]]))
                tmp_s = np.concatenate((tmp_s, x), axis=1)

        print(f"tmp_s.shape {tmp_s.shape}")
        self.norm_s_x = max(np.absolute(tmp_s[0, :])) #np.linalg.norm(tmp_s[0, :])
        self.norm_s_y = max(np.absolute(tmp_s[1, :])) #np.linalg.norm(tmp_s[1, :])
        self.norm_s_t = max(np.absolute(tmp_s[2, :])) #np.linalg.norm(tmp_s[2, :])
        print(f"self.norm_s_x {self.norm_s_x}  self.norm_s_y {self.norm_s_y}  self.norm_s_t {self.norm_s_t}")

        for a in sim_states:
            a[0,:] = a[0,:] / self.norm_s_x
            a[1,:] = a[1,:] / self.norm_s_y
            a[2,:] = a[2,:] / self.norm_s_t

        # tmp_c = np.concatenate(sim_cmds)
        # self.norm_c = np.linalg.norm(tmp_c)
        # 'Fom MPC we need normalizing the command but for Koompan we shuold NOT do it; to be investigated more for koopman'
        # sim_cmds = sim_cmds / self.norm_c
        # #print("self.norm_c ", self.norm_c)


        'Because in Automatic Transmission we have two types of commands we have to normalize both cmds'
        'Concating all arrays in sim_states column-wise to get the norm'
        tmp_c = np.zeros((sim_cmds[0].shape[0],1))
        # print(f"koopman preprocessing sim_states[0].shape {sim_cmds[0].shape}")
        for c in sim_cmds:
            for j in range(c.shape[1]):
                x = np.transpose(np.array([c[:,j]]))
                tmp_c = np.concatenate((tmp_c, x), axis=1)

        print(f"tmp_c.shape {tmp_c.shape}")
        self.norm_c_0 = max(np.absolute(tmp_c[0, :])) #np.linalg.norm(tmp_c[0, :])
        self.norm_c_1 = max(np.absolute(tmp_c[1, :])) #np.linalg.norm(tmp_c[1, :])
        # print(f"self.norm_c_0 {self.norm_c_0}  self.norm_c_1 {self.norm_c_1}")


        for c in sim_cmds:
            c[0,:] = c[0,:] / self.norm_c_0
            c[1,:] = c[1,:] / self.norm_c_1

        # print(f"sim_cmds after {sim_cmds}")


        return sim_states, sim_cmds


    def normalize(self, data):

        data[0,0] = data[0,0] / self.norm_s_x
        data[0,1] = data[0,1] / self.norm_s_y
        if data.shape[1] > 2:
            data[0,2] = data[0,2] / self.norm_s_t

        return data


    def reverse_normalize(self, data):
 
        data[0,0] = data[0,0] * self.norm_s_x
        data[1,0] = data[1,0] * self.norm_s_y
        if data.shape[1] > 2:
            data[2,0] = data[2,0] * self.norm_s_t

        return data


    def Test(self, init_test_state, test_cmds):

        cmds = copy.copy(test_cmds)
        outx = [] 
        outy = []
        realx = []
        realy = []

        rs = self.reverse_normalize(np.transpose(np.array([[init_test_state[0, 0],
                                init_test_state[1, 0], init_test_state[2, 0]]])))

        outx.append(rs[0,0])
        outy.append(rs[1,0])
        realx.append(rs[0,0])
        realy.append(rs[1,0])

        for i in range(0, len(cmds[0])):
            s = init_test_state[:,i]
            tmp = np.transpose(np.array([s]))
            # print(f"cmds[:,i] {cmds[:, i]}")
            p = self.predict(tmp, copy.copy(np.transpose(np.array(cmds[:,i]))))

            rs = self.reverse_normalize(np.transpose(np.array([[init_test_state[0, i],
                                        init_test_state[1, i], init_test_state[2, i]]])))
            ps = self.reverse_normalize(np.transpose(np.array([[p[0, 0], p[1, 0], p[2, 0]]])))

            realx.append(rs[0, 0])
            realy.append(rs[1, 0])
            outx.append(ps[0, 0])
            outy.append(ps[1, 0])

        # print(f"realx {realx}")
        # print(f"outx {outx}")
        # print(f"realy {realy}")
        # print(f"outy {outy}")

        lines = plt.plot(outx, outy)
        # plt.figure(figsize=(8.5, 6), dpi=100)
        # plt.plot(lines, linestyle=":", linewidth=2, color="red")
        lines = plt.plot(realx, realy)
        # plt.plot(lines, linestyle="-", linewidth=1.8, color="blue")

        plt.xlabel("X", fontsize=16, fontweight='bold')
        plt.ylabel("Y", fontsize=16, fontweight='bold')
        plt.xticks(fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        plt.legend(('Prediciton', 'Simulation'), loc='upper right', fontsize=14)
        # plt.title('State trajectory - multi step MPC')
        plt.show()



"""def main():
if __name__ == "__main__":
    main()"""
