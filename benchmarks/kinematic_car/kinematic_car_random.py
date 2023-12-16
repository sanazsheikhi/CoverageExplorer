import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt




ode = lambda t, x, u1, u2: [x[2] * np.cos(x[3]),
                            x[2] * np.sin(x[3]),
                                             u1,
                                             u2]


def run_kinematic_car(sim_count, steps=100, dt_=0.1):

    'This version of the program runs with random control inputs'

    all_traj_states, all_traj_inputs = [], []

    seed = 0
    while len(all_traj_states) < sim_count:

        tFinal= dt_
        state_ = [0.0, 0.0, 15.0, 0] # init = [px, py, v, phi]#, yaw_angle]
        output_state = [[state_[0], state_[1]]]

        np.random.seed(seed)
        acc_ = np.random.uniform(-9.81, 9.81, steps) # based on real car
        steer_ = np.random.uniform(-0.4, 0.4, steps) # based on real car
        inputs = list(map(lambda a, b: [a, b], acc_, steer_))
        seed += 1

        for i in range(steps):
            sol = solve_ivp(ode, [0, tFinal], state_, args=(acc_[i], steer_[i]), dense_output=True)
            state_ = sol.sol(tFinal)
            output_state.append([state_[0], state_[1]])

        all_traj_states.append(np.array(output_state))
        all_traj_inputs.append(np.array(inputs))

        plt.plot(np.array(output_state)[:, 0], np.array(output_state)[:, 1], label="simulate")
    plt.scatter(0.0, 0.0, c='red', s=14.0)
    plt.show()


    return  all_traj_states, all_traj_inputs


