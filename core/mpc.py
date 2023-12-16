import numpy as np
import cvxpy as cp
import AutomaticKoopman as AKM
from copy import copy, deepcopy
cp.settings.ERROR = [cp.settings.USER_LIMIT]  


class MPC:

    def __init__(self, benchmark, train_states, train_cmds, steps, dt):

        self.benchmark = benchmark

        if self.benchmark == 'kinematic_car':
            self.inputs_lower_bound = [-9.8, -0.4]
            self.inputs_upper_bound = [9.8, 0.4]
        elif self.benchmark == 'ACASXU':
            self.inputs_lower_bound = [-3]
            self.inputs_upper_bound = [3]
        elif self.benchmark == 'automatic_transmission':
            self.inputs_lower_bound = [0.0, 0.0]
            self.inputs_upper_bound = [100.0, 350.0]
        elif self.benchmark == 'point_mass':
            self.inputs_lower_bound = [-15.0, -15.0]
            self.inputs_upper_bound = [15.0, 15.0]

        'AutoKoopman'
        states_, cmds_ = self.preprocessing(train_states, train_cmds)
        self.model = AKM.get_model(states_, cmds_, steps, dt)
        self.A_DMD, self.B_DMD = self.model.A, self.model.B
        self.WF, self.BF =  (self.model.obs.observables[1].w ,
                             self.model.obs.observables[1].u)

        self.T = 5 # Time window, lookahead
        self.n = self.A_DMD.shape[1] # num of states
        self.m = self.B_DMD.shape[1] # num of input cmd


    def normalize_state(self, state_):
        'Normalize state(s)'
        dimension = state_.shape[1]
        data = copy(state_)
        for i in range(dimension):
            data[:, i] = data[:, i] / self.norm_states[i]

        return data


    def normalize_input(self, input_):
        'Normalize inputs'
        dimension = input_.shape[1]
        data = copy(input_)
        for i in range(dimension):
            data[:, i] = data[:, i] / self.norm_inputs[i]

        return data


    def reverse_normalize_state(self, state_):
        'Reverse normalize states'
        dimension = state_.shape[1]
        data = copy(state_)
        for i in range(dimension):
            data[:, i] = data[:, i] * self.norm_states[i]

        return data


    def reverse_normalize_input(self, input_):
        'Reverse normalize inputs'
        dimension = input_.shape[1]
        data = copy(input_)
        for i in range(dimension):
            data[:, i] = data[:, i] * self.norm_inputs[i]

        return data


    def generate_xp(self, s):
        res = self.model.obs.obs_fcn(s)
        return res


    def predict(self, current_state, target_state,
                      dimension_state=2, dimension_input=2):
        constr = []
        answers = []
        cost = 0
        x_0 = self.generate_xp(current_state).reshape(self.n)
        # x_0 = current_state.reshape(self.n) # In case we use manual A and B
        x = cp.Variable((self.n, self.T + 1))
        u = cp.Variable((self.m, self.T))
        Q = np.diag([1.0]*dimension_state)

        z_ = target_state
        for t in range(self.T):
            'system dynamics'
            constr += [x[:, t + 1] == self.A_DMD @ x[:, t] + self.B_DMD @ u[:, t]]

        if self.benchmark == 'kinematic_car' or \
           self.benchmark == 'automatic_transmission' :
            dist_func = x[0:2, [t + 1]] - z_[0:2]
            cost += cp.quad_form(dist_func, Q)
            constr += [self.inputs_lower_bound[0] / self.norm_inputs[0] <= u[0, :]]
            constr += [u[0, :] <= self.inputs_upper_bound[0] / self.norm_inputs[0]]
            constr += [self.inputs_lower_bound[1] / self.norm_inputs[1] <= u[1, :]]
            constr += [u[1, :] <= self.inputs_upper_bound[1] / self.norm_inputs[1]]

        elif self.benchmark == 'ACASXU':
            'both aircraft positions'
            dist_func = (0.5 *(x[0:2, [t + 1]] - target_state[0:2]) +
                         0.5 *( x[3:5, [t + 1]] - target_state[0:2]))
            cost += cp.quad_form(dist_func, Q)
            constr += [self.inputs_lower_bound[0] / self.norm_inputs[0]  <= u[:, :]]
            constr += [u[:, :] <= self.inputs_upper_bound[0] / self.norm_inputs[0]]


        ' the first column of x is equal to our initial state (start point).'
        constr += [x[:, 0] == x_0]

        'Solving the optimization problem'
        problem = cp.Problem(cp.Minimize(cost), constr)
        try:
            f = problem.solve(solver=cp.OSQP)#, verbose=True)
        except ValueError as ve:
            print(f"No valid solution for taget "
                  f"{self.reverse_normalize_state(np.transpose(target_state))}")
            return  None, None, None, None

        action = [0.0] * dimension_input
        if problem.status == cp.OPTIMAL or problem.status == cp.OPTIMAL_INACCURATE:

            X_ = x[:, :].value
            ps = self.reverse_normalize_state(np.transpose(np.array([[X_[0,0], X_[1,0], X_[2,0]]])))
            answers.append(f)
            for i in range(dimension_input):
                action[i] = u[i, 0].value * self.norm_inputs[i]

        else:
            print(f"No solution found,  taget "
                  f"{self.reverse_normalize_state(np.transpose(target_state))}  "
                  f"current_state "
                  f"{self.reverse_normalize_state(np.transpose(current_state))}")
            print(f"problem.status {problem.status}  problem.value {problem.value}")
            return None, None, None, None

        return action, ps[0,0], ps[1,0], problem.value


    def koopman_predict(self, curr_state, input, steps, dt):
        teval = []
        for i in range(steps):
            teval.append((i+1)*dt)
        trajectory = self.model.solve_ivp(
            initial_state=curr_state,
            inputs=np.array(input),
            tspan=(0.0, dt*steps),
            teval=np.array(teval),
            sampling_period=dt
        )

        return trajectory



    def preprocessing(self, sim_states_, sim_cmds_):
        """
        Normalizing and scaling the dataset
        inputs:sime_states: list of trajectories, where each trajectory is
        an array of atates sim_cmds: list of input sequences, where each
        input sequence is an array of input entry
        output: Normalized states and inputs and normalization factors to
        be used for normalize/denormalize
        """
        sim_states = deepcopy(sim_states_)
        sim_cmds = deepcopy(sim_cmds_)
        dimension_states = sim_states[0].shape[1]
        dimension_inputs = sim_cmds[0].shape[1]

        'Finding the maximum value of each dimension among all trajectories'
        norm_states = [0.0] * dimension_states
        for a in sim_states:
            for j in range(dimension_states):
                tmp_max = max(abs(a[:, j]))
                if tmp_max > norm_states[j]:
                    norm_states[j] = tmp_max

        self.norm_states = norm_states

        'Normalizing the trajectories of states'
        for a in sim_states:
            for j in range(dimension_states):
                a[:, j] = a[:, j] / self.norm_states[j]

        'Normalizing the sequences of inputs'
        norm_inputs = [0.0] * dimension_inputs
        for c in sim_cmds:
            for j in range(dimension_inputs):
                tmp_max = max(abs(c[:, j]))
                if tmp_max > norm_inputs[j]:
                    norm_inputs[j] = tmp_max

        self.norm_inputs = norm_inputs
        for c in sim_cmds:
            for j in range(dimension_inputs):
                c[:, j] = c[:, j] / self.norm_inputs[j]

        return sim_states, sim_cmds

