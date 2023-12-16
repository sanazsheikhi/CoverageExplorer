
from matplotlib import pyplot as plt
import numpy as np
import koopman_mpc as KM
# import gurobipy
import cvxpy as cp
cp.settings.ERROR = [cp.settings.USER_LIMIT]  



class MPC:

    def __init__(self, sim_states_int, sim_cmds_int, sim_states_own):

        print("MPC begining")
        print("sim_states_int ", len(sim_states_int))
        print("sim_cmds_int ", len(sim_cmds_int))
        print("sim_states_own ", len(sim_states_own))

        index = int(len(sim_states_int) * 0.8)
        train_sim_states_int = sim_states_int[0:index]
        train_sim_cmds_int = sim_cmds_int[0:index]
        test_sim_states_int = sim_states_int[index:]
        test_sim_cmds_int = sim_cmds_int[index:]
        test_sim_states_own = sim_states_own[index:]
        
        self.km = KM.KOOPMAN(train_sim_states_int, train_sim_cmds_int)
        self.A_DMD, self.B_DMD = self.km.get_DMD_matrices()

        print("max A: ", np.amax(self.A_DMD))

        self.T = 10 # Time window, look ahead
        self.n = self.A_DMD.shape[1] # num of states
        self.m = self.B_DMD.shape[1] # num of inputs=cmd

        print("T ", self.T, " n ", self.n, " m ", self.m)

        # self.Test_3(test_sim_states_int, test_sim_cmds_int, test_sim_states_own)



    def Test_2(self, test_states_int, test_cmds_int, test_states_own):
       
        print("len(test_states_int) ", len(test_states_int))

        for i in range(len(test_states_int)):
            
            states_int = test_states_int[i]
            states_own = test_states_own[i]
            cmds = test_cmds_int[i]
            tmpc = []
            tmpx = []
            tmpy = []

            tmpx.append(states_int[0,0])
            tmpy.append(states_int[1,0])
 
            #print(i," X: ", states_int[0,0], " Y: ", states_int[1,0], " T: ", states_int[2,0])
            #rs = self.reverse_normalize(np.transpose(np.array([[states_int[0,0], states_int[1,0], states_int[2,0]]]))) # for test
            #print(i, " x_int ", rs[0,0], " y_int ", rs[1,0], "t_int ", rs[2,0])


            for j in range(states_int.shape[1]):

                x_int = states_int[0, j]
                y_int = states_int[1, j]
                t_int = states_int[2, j]

                x_own = states_own[0, j]
                y_own = states_own[1, j]
                t_own = states_own[2, j]
                

                #int_ = self.normalize(np.array([[x_int, y_int, t_int]])) # it is normalized by koopman in preprocessing
                #own_ = self.normalize(np.array([[x_own, y_own, t_own]])) # it is normalized by koopman
                int_ = np.array([[x_int, y_int, t_int]])
                own_ = np.array([[x_own, y_own, t_own]])
                cmd, xp, yp, tp = self.predict(np.transpose(int_), np.transpose(own_))


                #rs = self.reverse_normalize(np.transpose(np.array([[x_int, y_int, t_int]]))) # for test
                #print(i, " x_int ", rs[0,0], " y_int ", rs[1,0])
                #ps = self.reverse_normalize(np.transpose(np.array([[xp, yp, tp]]))) # for test
                #print("xp ", ps[0,0], " yp ", ps[1,0], "\n")


                #tmpc.append(cmd)
                #tmpx.append(ps[0,0])
                #tmpy.append(ps[1,0])


                'The test_states_int data is normalized so for ploting the prediction we use normalized data'
                tmpc.append(cmd)
                tmpx.append(xp)
                tmpy.append(yp)


            """plt.plot(test_cmds_int[i], color='blue', label="random commands from simulation")
            plt.plot(tmpc, color='red', label="commands from prediction")
            plt.show()"""


            print("tmpx ", tmpx)
            print("tmpy ", tmpy)
            print("\n")
            print("states_int[0,:] ", states_int[0,:])
            print("states_int[1,:] ", states_int[1,:])

            print("---------------------------------------------------------------------------")

            lines = plt.plot(tmpx, tmpy)
            plt.setp(lines, linestyle=":",linewidth=2, color="red")
            lines = plt.plot(states_int[0,:], states_int[1,:])
            plt.setp(lines,linestyle="-", linewidth=2, color="blue")

            plt.xlabel("X")
            plt.ylabel("Y")
            plt.legend(('Prediciton', 'Simulation'), loc='upper right')
            plt.title('Intruder trajectory with mpc')
            plt.show()

    def Test_3(self, test_states_int, test_cmds_int, test_states_own):


        km_tmp = KM.KOOPMAN(test_states_int, test_cmds_int) #Just for the sake of normalization/ dont use preprocessing as we need norm data for reversing

       
        print("Test3 len(test_states_int) ", len(test_states_int))

        for i in range(37,len(test_states_int)): #tmp
            
            states_int = test_states_int[i]
            states_own = test_states_own[i]
            cmds = test_cmds_int[i]
            tmpc = []
            tmpx = []
            tmpy = []
            realc = []
            realx = []
            realy = []

            #tmpx.append(states_int[0,0])
            #tmpy.append(states_int[1,0])
 
            #print(i," X: ", states_int[0,0], " Y: ", states_int[1,0], " T: ", states_int[2,0])
            rs = km_tmp.reverse_normalize(np.transpose(np.array([[states_int[0,0], states_int[1,0], states_int[2,0]]]))) # for test
            print(i, " x_int ", rs[0,0], " y_int ", rs[1,0], "t_int ", rs[2,0])

            tmpx.append(rs[0,0])
            tmpy.append(rs[1,0])
            realx.append(rs[0,0])
            realy.append(rs[1,0])

            for j in range(states_int.shape[1]):

                x_int = states_int[0, j]
                y_int = states_int[1, j]
                t_int = states_int[2, j]

                x_own = states_own[0, j]
                y_own = states_own[1, j]
                t_own = states_own[2, j]
                

                #int_ = self.normalize(np.array([[x_int, y_int, t_int]])) # it is normalized by koopman in preprocessing
                #own_ = self.normalize(np.array([[x_own, y_own, t_own]])) # it is normalized by koopman
                int_ = np.array([[x_int, y_int, t_int]])
                own_ = np.array([[x_own, y_own, t_own]])
                cmd, xp, yp, tp = self.predict(np.transpose(int_), np.transpose(own_))


                rs = km_tmp.reverse_normalize(np.transpose(np.array([[x_int, y_int, t_int]]))) # for test
                #print(i, " x_int ", rs[0,0], " y_int ", rs[1,0])
                ps = km_tmp.reverse_normalize(np.transpose(np.array([[xp, yp, tp]]))) # for test
                #print("xp ", ps[0,0], " yp ", ps[1,0], "\n")


                
                tmpx.append(ps[0,0])
                tmpy.append(ps[1,0])

                realx.append(rs[0,0])
                realy.append(rs[1,0])


                """'The test_states_int data is normalized so for ploting the prediction we use normalized data'
                tmpc.append(cmd)
                tmpx.append(xp)
                tmpy.append(yp)"""


            """plt.plot(test_cmds_int[i], color='blue', label="random commands from simulation")
            plt.plot(tmpc, color='red', label="commands from prediction")
            plt.show()"""


            """print("tmpx ", tmpx)
            print("tmpy ", tmpy)For t
            print("\n")
            print("states_int[0,:] ", states_int[0,:])
            print("states_int[1,:] ", states_int[1,:])"""

            print("---------------------------------------------------------------------------")

            lines = plt.plot(tmpx, tmpy)
            plt.setp(lines, linestyle=":",linewidth=2, color="red")
            #lines = plt.plot(states_int[0,:], states_int[1,:])
            lines = plt.plot(realx, realy)
            plt.setp(lines,linestyle="-", linewidth=2, color="blue")

            plt.xlabel("X", fontsize=16, fontweight='bold')
            plt.ylabel("Y", fontsize=16, fontweight='bold')
            plt.xticks(fontsize=12, fontweight='bold')
            plt.yticks(fontsize=12, fontweight='bold')
            plt.legend(('Prediciton', 'Simulation'), loc='upper right', fontsize=12)
            # plt.title('Prediciton with multi-step MPC')
            plt.show()


#-----------------------------------------------------------------------------------------------


#----------------------------------------------------------------------------------------------


    def normalize(self, s):
 
        """ We call koopman functions because we have trained 
            the DMD operator by koopman having the norms """

        return self.km.normalize(s)


    def reverse_normalize(self, s):

        return self.km.reverse_normalize(s)

  
    def preprocessing(self, sim_states, sim_cmds):

        return self.km.preprocessing(self, sim_states, sim_cmds)


    def predict(self, current_state, target_state):

        constr = []
        answers = []
        cost = 0

        # current_state == observation
        # print(f"prediction self.n {self.n}     {self.km.generate_xp(current_state).shape}")
        x_0 = self.km.generate_xp(current_state).reshape(self.n)


        'Optimization variable should be defined per predicition not as class variables'
        x = cp.Variable((self.n, self.T + 1)) # T is window of prediction
        u = cp.Variable((self.m, self.T))

        # Q = np.diag([1.0, 1.0, 1.0])
        Q = np.diag([1.0, 1.0])

        # w0 = 0.8
        # w1 = 0.4
        # w2 = 0.2
        for t in range(self.T):
            constr += [x[:, t + 1] == self.A_DMD @ x[:, t] + self.B_DMD @ u[:, t]] # system dynamics # commented for test
            # dist_func = x[0:3, [t+1]] - target_state[0:3]
            # dist_func = x[0:2, [t+1]] - target_state[0:2] # intruder position
            dist_func = x[3:5, [t + 1]] - target_state[0:2] #ownship position
            cost += cp.quad_form(dist_func, Q)
            # cost += w0*((x[0,[t+1]] - target_state[0])**2 + (x[1,[t+1]] - target_state[1])**2) + w2*((x[2,[t+1]] - target_state[2])**2)


        # dist_func = x[0:2, [self.T]] - target_state[0:2] # intruder position
        # cost += cp.quad_form(dist_func, Q)

        # dist_func = x[3:5, [self.T]] - target_state[0:2]  # ownship position
        # cost += cp.quad_form(dist_func, Q)

        # cost += (x[0, [self.T]] - target_state[0]) ** 2 + (x[1, [self.T]] - target_state[1]) ** 2

        """ cp.quad_form(X, Q) = transpose(X) * Q * X
            when Q == identity matrix I, then:
            cp.quad_form(X, Q) = transpos(X) * X
            which is equal to X^2
            E.g:
            X = [a, b]
            cp.quad_form(X, I) = transpose(X) * X
                               = a^2 + b^2

            quad_form() is a known function for optimizer 
            and it knows how to optimize it. If we implement
            the distance function manually the optimizer can 
            handle it but it would be slower.
            So we usually try to cast the problems to known
            cost functions.
        """

        """ For estimating the cost we take the last state in 
            the time window which is the closest to the target.
            however, for the final result we take the prediction
            for the first step which is more accurate.
            Then we move to the next state and from there make new 
            predictions for the whole time window
        """

        """ The cost function will minimize the distance between 
            current state and target state. 
            quad_form((X_curr - X_target), Q) 
            where state X is (x,y) position
        """


        # constr += [-3.0 / self.km.norm_c_0 <= u[:, :]]
        # constr += [u[:, :] <= 3.0 / self.km.norm_c_0]

        constr += [-3.0 <= u[:, :]]
        constr += [u[:, :] <= 3.0]


        constr += [x[:, 0] == x_0] # the first column of x is equal to our initial state (start point). 


        'Solving the optimization problem'
        problem = cp.Problem(cp.Minimize(cost), constr)
        # f = problem.solve(solver=cp.ECOS)#,  verbose=True)
        f = problem.solve(solver=cp.OSQP)#, verbose=True)
        #f = problem.solve(solver=cp.GUROBI)
        #print("optimal value with ECOS:", problem.value)

     
        if problem.status == cp.OPTIMAL or problem.status == cp.OPTIMAL_INACCURATE:
            #print("solution found")        
            # X_ = x[:, :].value
            #print("Solution X_: ")
            #print("pred x: ", X_[0, :])
            #print("pred y: ", X_[1, :])
            #print("pred t: ", X_[2, :])


            #ps = self.reverse_normalize(np.transpose(np.array([[X_[0,1], X_[1,1], X_[2,1]]]))) # for test
            #ps = self.reverse_normalize(np.transpose(np.array([[X_[0,0], X_[1,0], X_[2,0]]]))) # for test
            #print("@@@@@ xp ", ps[0,0], " yp ", ps[1,0], " u ")#, u[0,0].value * self.km.norm_c, "\n")
            
            #print("action 0: ", u[0,0].value * self.km.norm_c, " action 1: ", u[0,1].value * self.km.norm_c)


            answers.append(f)
            action = u[0, 0].value #* self.km.norm_c_0
            # action = u[0, 1].value * self.km.norm_c

            # action = int(round(action))
            # print(f"action {action}")
            # print(f"a1 {a}, a2 {action}")

        else:
            print("No solution found, action 0")
            X_ = np.array([[0, 0],[0, 0],[0, 0]])
            #X_ = [[0], [0], [0]]
            action = 0
            answers.append(np.inf)


        # print("cost ", problem.value)

        return action
        # return  action, X_[0, 0], X_[1, 0], X_[2, 0]
        # return  action, X_[0, 1], X_[1, 1], X_[2, 1]




# if __name__ == "__main__":
#     main()
