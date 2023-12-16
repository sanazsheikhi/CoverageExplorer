import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


"""
This is the implementation of Kinematic Single Track car:
dp_x = v.cos(phi)
dp_y = v.sin(phi)
dv = a (accelaration; change in V)
dphi = omega (change in heading)
dpsi = v / l * tan(phi)

We first define an ODE, then give the ODE to a solver in 
a loop where each time inital values for the solver would 
be the output of the solver in the previous loop iteration.
Also, in each iteration we give it new input control signal,
and based on the ODE the solver knows how to replace them.

References: 
https://gitlab.lrz.de/tum-cps/commonroad-vehicle-models/-/blob/master/PYTHON/vehiclemodels/vehicle_dynamics_ks.py
https://gitlab.lrz.de/tum-cps/commonroad-vehicle-models/-/blob/master/PYTHON/vehiclemodels/parameters/parameters_vehicle2.yaml
https://www.researchgate.net/publication/317491725_CommonRoad_Composable_Benchmarks_for_Motion_Planning_on_Roads
"""

# # axes distances
# # distance from spring mass center of gravity to front axle [m]  LENA
# a = 1.1561957064
# # distance from spring mass center of gravity to rear axle [m]  LENB
# b = 1.4227170936
# # wheelbase
# l = a + b

ode = lambda t, x, u1, u2: [x[2] * np.cos(x[3]),
                            x[2] * np.sin(x[3]),
                                             u1,
                                             u2] #, x[2] / l * np.tan(x[3])]

def run_training(sim_count, steps, dt):
    # print(f"KinematicCar::run_training")
    x0 = [0.0, 0.0, 15.0, 0.0] # initial state [x,y,velocity,orientation]

    # range for the uncertain inputs [acceleration, steer]
    input_lower_bound = [-9.81, -0.4]
    input_upper_bound = [9.81, 0.4]

    # create training data by simulating the system
    train_states, train_inputs = [], []

    # plot
    X_, Y_ = [0.0], [0.0]

    for i in range(sim_count):

        states = [x0]
        inputs = []

        # loop over all time steps
        for j in range(steps):
            # choose random input
            input = np.random.uniform(low=input_lower_bound, high=input_upper_bound)

            # simulate the system
            sol = solve_ivp(ode, [0, dt], states[-1], args=(input), dense_output=True)

            # save the trajectory
            inputs = inputs + [input]
            states = states + [sol.sol(dt)]

            # plot
            X_.append(sol.sol(dt)[0])
            Y_.append(sol.sol(dt)[1])


        train_states.append(np.array(states))
        train_inputs.append(np.array(inputs))

        # plot
        # plt.title = "Train data"
        # plt.plot(X_, Y_)

    # plt.show()

    return train_states, train_inputs



def run_singleStep(state_, acc_cmd, steer_cmd):
    # print(f"run_mpc state_ {state_} acc {acc_cmd} steer {steer_cmd}")
    tFinal= 0.1
    sol = solve_ivp(ode, [0, tFinal], state_, args=(acc_cmd, steer_cmd), dense_output=True)
    state_out =  sol.sol(tFinal)

    return  state_out



def run_kinematic_car(sim_count=100, steps=100, mpc_obj=None, sampler_obj=None):

    # print(f"Sim_Handler:: run_kinematic_car")

    all_traj_states = [] # to store all sim_count trajectories
    all_traj_inputs = [] # to store all sim_count trajectories
    rsampler = sampler_obj
    mpc = mpc_obj

    i = 0
    internal_counter = 0 # this is to generate new seed
    while i < sim_count:
    # for i in range(sim_count):

        if i % 10 == 0: print(f"Run {i}")

        states_ = [0.0, 0.0, 15.0, 0.0]  # init_state
        tmp_states, tmp_inputs = [states_], [] # to form traj from steps to store

        'Finding a target state'
        target_ = rsampler.rejection_sampling(seed= internal_counter)
        target = np.transpose(mpc.normalize_state(np.array([target_])))
        internal_counter += 1

        # print(f"target_ {target_}")

        min_distance  = math.sqrt((target_[0] - states_[0]) ** 2 + (target_[1] - states_[1]) ** 2)

        for j in range(steps):
            # curr_state = copy(np.transpose(copy(mpc.normalize(copy(np.array([states_]))))))
            curr_state = mpc.normalize_state(np.array([states_]))  # no transpose() for new obs_function

            action, m_x, m_y, cost = mpc.predict(curr_state, target)

            if action is None:
                'No solution found'
                break

            states_ = run_singleStep(states_, action[0], action[1])
            tmp_states.append(states_) # to form traj to store
            tmp_inputs.append(action)  # to form traj to store

            distance = math.sqrt((target_[0] - states_[0]) ** 2 + (target_[1] - states_[1]) ** 2)
            # print(f"distance {distance}  cost {cost}")
            if distance < min_distance:
                min_distance = distance


        # if len(tmp_states) > 2:
        if len(tmp_states) == steps + 1: # we need trajectories of the same length for kmeans clustering
            'Omit no solution found traces'
            all_traj_states.append(np.array(tmp_states))
            all_traj_inputs.append(np.array(tmp_inputs))

            'Add to kdt-tree for rejection sampling'
            rsampler.update_kdt([np.array(tmp_states)])
            i += 1

        plt.plot(np.array(tmp_states)[:,0], np.array(tmp_states)[:,1], label="simulate")
        plt.scatter(0.0, 0.0, c='red', s=14.0)
        # plt.scatter(target_[0], target_[1], c='black', s=14.0)

        # if not train:
    # plt.show()


    return  all_traj_states, all_traj_inputs

