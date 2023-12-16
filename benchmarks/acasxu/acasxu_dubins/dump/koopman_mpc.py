
'Implementation of Koopman operator with Fourier feature and testing it with acasxu data samples'

import numpy as np
from copy import copy
from scipy import stats
import matplotlib.pyplot as plt


class KOOPMAN:

    def __init__(self, sim_states, sim_cmds):

        if len(sim_states) == 0:
            print("Empty training set !")
            return

        # states, cmds = self.preprocessing(sim_states, sim_cmds)

        n_sim_states, n_sim_cmds = self.preprocessing(sim_states, sim_cmds)
        #n_sim_states, n_sim_cmds = states, cmds
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
        self.num_observables = 70 # This is the best I tries several of them
        rank = 10#self.num_observables
        X = [0, 0, 0, 0, 0, 0] #[0, 0, 0]
        XP = []
        self.WF , self.BF = self.generate_Fourier(len(X), self.num_observables)


        'making input from states and observables for DMD'     
        for s in states:
            xps = self.generate_xp(s)
            XP.append(xps)

   
        self.A, self.B = self.DMD(XP, cmds, rank) # Koopman operators

        print("max A: ", np.max(self.A))
        print("min A: ", np.min(self.A))
        print("max B: ", np.max(self.B))
        print("min B: ", np.min(self.B))
        # print("A: ", self.A)
        # print("B: ", self.B)
   

        # for i in range(len(states)):
        # for i in range(len(test_states)):
        #     print(i)
        #     self.Test(test_states[i], test_cmds[i])
        # #     #self.Test(states[i], cmds[i]) #sanity check




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
        
        out = np.cos(np.dot(WT, X) + b)
 
        return out



    def DMD(self, X, U, rank):

        'Dynamic Mode Decomposition'

        print("koopman: DMD")
 
        tmp = X[0]
        X1 = tmp[:,0:tmp.shape[1] -1]
        X2 = tmp[:,1:tmp.shape[1]]
        # print(f"x1.shape {X1.shape}")
   
        for i in range(1, len(X)):
            tmp = np.array(X[i])
            # print(f"tmp.shape {tmp.shape}")
            X1 = np.concatenate((X1,tmp[:,0:tmp.shape[1] - 1]), axis=1)
            X2 = np.concatenate((X2,tmp[:,1:tmp.shape[1]]), axis=1)
            # print(f"x1.shape {X1.shape}")
    
        U_ = np.array([U[0]])
        # print(f"U_.shape {U_.shape}")
        # print(f"U.len {len(U)} , X.len {len(X)}")

        for i in range(1, len(U)):
            U_ = np.concatenate((U_, np.array([U[i]])), axis=1)
            # print(f"U[i].shape {U[i].shape}")

        # print(f"U_.shape {U_.shape} , X1.shape {X1.shape}")

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
                out[i, r] = self.g(s[:,r], self.WF[i], self.BF[i])

        # print(f"generate_xp s.shape {s.shape}   out.shape {out.shape}")

        res = np.concatenate((s, out))

        return res
    

    def predict(self, xs, us):

        g_xs = self.generate_xp(xs)
        out = np.dot(self.A, g_xs) + np.multiply(self.B, us)

        #print("np.dot(self.A, g_xs) ", np.dot(self.A, g_xs)[0:3], " np.multiply(self.B, us) ", np.multiply(self.B, us)[0:3])
        #print("A.shape ", self.A.shape, "g_xs.shape: ", g_xs.shape )
        #print("B.shape ", self.B.shape, " us ", us)

        return out
   


    def preprocessing(self, sim_states, sim_cmds):
  
        'Normalizing and scalling the dataset'
        print("koopman: preprocessing ...")

        'concating all arrays in sim_states column-wise to get the norm' 
        # tmp_s = np.zeros((sim_states[0].shape[0],1))
        # for a in sim_states:
        #     for j in range(a.shape[1]):
        #         x = np.transpose(np.array([a[:,j]]))s
        #         tmp_s = np.concatenate((tmp_s, x), axis=1)
        # print(f"tmp_s.shape {tmp_s.shape}")

        self.norm_s_x = 10000.0 # max(np.absolute(tmp_s[0, :]))  # np.linalg.norm(tmp_s[0, :])
        self.norm_s_y = 10000.0 # max(np.absolute(tmp_s[1, :]))  # np.linalg.norm(tmp_s[1, :])
        self.norm_s_t = 1.0 # max(np.absolute(tmp_s[2, :]))  # np.linalg.norm(tmp_s[2, :])
        # print(f"self.norm_s_x {self.norm_s_x}  self.norm_s_y {self.norm_s_y}  self.norm_s_t {self.norm_s_t}")

        for a in sim_states:
            # print(f"before a {a}")
            a[0,:] = a[0,:] / self.norm_s_x   # x intruder
            a[1,:] = a[1,:] / self.norm_s_y   # y intruder
            #a[2,:] = a[2,:] / self.norm_s_t
            a[2,:] = a[2,:] / self.norm_s_x   # x ownship
            a[3,:] = a[3,:] / self.norm_s_y   # y ownship
            # a[4, :] = self.normalize_ownship_y(a[4, :])
            #a[5,:] = a[5,:] / self.norm_s_t
            # print(f"after a {a}")
            # print()

        'Concating all arrays in sim_states column-wise to get the norm'
        'It covers cases where we have more than one cmd'
        # tmp_c = np.zeros((sim_cmds[0].shape[0],1))
        # tmp_c = np.zeros(1)
        # # print(f"koopman preprocessing sim_states[0].shape {sim_cmds[0].shape}")
        # for c in sim_cmds:
        #     for j in range(c.shape[0]):
        #         # x = np.transpose(np.array([c[:,j]]))
        #         # x = np.transpose(np.array([c[j]]))
        #         x = np.array([c[j]])
        #         tmp_c = np.concatenate((tmp_c, x))#, axis=1)
        #
        # self.norm_c_0 = max(np.absolute(tmp_c)) #np.linalg.norm(tmp_c[0, :])
        self.norm_c_0 = 1.0
        # self.norm_c_1 = max(np.absolute(tmp_c[1, :])) #np.linalg.norm(tmp_c[1, :])
        print(f"self.norm_c_0 {self.norm_c_0}") #self.norm_c_1 {self.norm_c_1}")

        # for c in sim_cmds:
        #     for i in range(c.shape[0]):
        #         c[i] = c[i] / self.norm_c_0


        return sim_states, sim_cmds


    def normalize(self, data):
        'Intruder'
        data[0,0] = data[0,0] / self.norm_s_x
        data[0,1] = data[0,1] / self.norm_s_y 
        # data[0,2] = data[0,2] / self.norm_s_t

        'ownship'
        if data.shape[1] > 3:
            data[0,2] = data[0,2] / self.norm_s_x
            data[0,3] = data[0,3] / self.norm_s_y
            # data[0,5] = data[0,5] / self.norm_s_t

        return data


    def reverse_normalize(self, data):

        data[0,0] = data[0,0] * self.norm_s_x
        data[1,0] = data[1,0] * self.norm_s_y
        # data[2,0] = data[2,0] * self.norm_s_t

        if data.shape[1] > 3:
            data[2, 0] = data[2, 0] * self.norm_s_x
            data[3, 0] = data[3, 0] * self.norm_s_y
            # data[5, 0] = data[5, 0] * self.norm_s_t


        return data


    def Test(self, init_test_state, test_cmds):

        cmds = test_cmds
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

        for i in range(0, len(cmds)):
            s = init_test_state[:,i]
            tmp = np.transpose(np.array([s]))
            p = self.predict(tmp, cmds[i])

            rs = self.reverse_normalize(np.transpose(np.array([[init_test_state[0, i],
                                        init_test_state[1, i], init_test_state[2, i]]])))
            ps = self.reverse_normalize(np.transpose(np.array([[p[0, 0], p[1, 0], p[2, 0]]])))

            realx.append(rs[0, 0])
            realy.append(rs[1, 0])
            outx.append(ps[0, 0])
            outy.append(ps[1, 0])

        plt.plot(outx, outy)
        plt.plot(realx, realy)
        # plt.figure(figsize=(8.5, 6), dpi=100)


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
