import random
import time
import matlab.engine
import numpy as np
from copy import  deepcopy
import matplotlib.pyplot as plt


global mpc


def plot_2(trajectories, trajectories_pred=None, target=None):

    'plot trajectories seperately'

    plt.xlabel("Velocity")
    plt.ylabel("RPM")
    plt.xlim(0, 120)
    plt.ylim(0, 4500)
    # plt.xticks([-1000, -500, 0, 500, 1000])
    # plt.yticks([-1000, -500, 0, 500, 1000])
    plt.title("MPC")

    for i in range(len(trajectories)):
        T = trajectories[i]
        plt.plot(T[0], T[1],'b')
        # plt.scatter(T[0][0], T[0][1], s=100, cmap='black')

    if trajectories_pred is not None:
        for i in range(len(trajectories_pred)):
            T_pred = trajectories_pred[i]
            # print(f"{i} T_pred[0] {T_pred[0]}")
            # print(f"{i} T_pred[0] {T_pred[1]}")
            # print()
            plt.plot(T_pred[0], T_pred[1], 'r')

    if target is not None:
        plt.scatter(target[0], target[1], s=100, cmap='orange')  # target
        # print(f"plotting traj # {i} len traj {len(T[0])}")
    plt.show()



def process(states, brake, throttle):

    trace_len = len(brake)
    speed = np.array([]) # state
    RPM = np.array([])   # state
    gear = np.array([])  # state

    # cmd_seq = np.array([])
    cmd_seq_ = []

    'For training we use all three states variables: speed, RPM, and gear but for MPC cost we just use speed & RPM'
    for i in range(trace_len):
        # print(f"process {states[i][0]}  {states[i][1]}  {states[i][2]}")
        speed = np.concatenate((speed, np.array([states[i][0]])))
        RPM = np.concatenate((RPM, np.array([states[i][1]])))
        gear = np.concatenate((gear, np.array([states[i][2]])))
        # cmd_seq = np.concatenate((cmd_seq, np.array([brake[i], throttle[i]])))
        cmd_seq_.append([throttle[i], brake[i]])

    cmd_seq = np.transpose(np.array(cmd_seq_[0:len(cmd_seq_)-1]))
    states_ = np.concatenate((np.array([speed]), np.array([RPM]), np.array([gear])))

    return states_, cmd_seq



def get_training_data(sim_count=100, steps=100, dt=0.01):
    'Running the simulation to collect training data'

    'Control inputs: throttle & brake; Output States: speed & RPM & gear'

    # print(f"automatic_transmission:: get_training_data")

    dt = dt
    trace_len = steps # size of data trace
    sim_count = sim_count
    train_set_states = []
    train_set_cmdseq = []

    'Starting Matlab engine'
    eng = matlab.engine.start_matlab()
    # eng.cd(r'/home/sanaz/Documents/Projects/STaliro/benchmarks/ARCH2014/', nargout=0)
    eng.cd(r'/home/sanaz/Documents/Projects/s-taliro_public/trunk/benchmarks/ARCH2019', nargout=0)

    s = time.time()
    for j in range(sim_count):
        # print(f"training round {j}")
        np.random.seed(j)
        time_steps = []
        brake = [0.0]
        throttle = [0.0]
        cmd_seq = []
        'Generating control input to simulate the system'
        # Input signal range throttle=[0,100], brake=[0,350]
        time_steps = np.linspace(0, dt*trace_len, num=trace_len).tolist()
        brake = brake + np.random.uniform(0, 350, trace_len-1).tolist()
        throttle = throttle + np.random.uniform(0, 100, trace_len-1).tolist()

        for i in range(trace_len):
            cmd_seq.append([throttle[i], brake[i]])

        # print(f"brake {brake}")
        # print(f"throttle {throttle}")

        'Simulating in Matlab & processing the simulation data'
        # print(f"timesteps {time_steps}")
        # print(f"brake {brake}")
        # print(f"throttle {throttle}")
        states = eng.test(throttle, brake, time_steps, trace_len*dt)

        train_set_states.append(np.array(states))
        train_set_cmdseq.append(np.array(cmd_seq))

    eng.quit()
    e = time.time()
    print(f"Training data collection time {e-s}")
    return train_set_states, train_set_cmdseq



def run_automatic_transmission(sim_count=100, steps=100, dt_=0.01, mpc_obj=None, sampler_obj=None):

    'Control inputs: throttle & brake; Output States: speed & RPM'

    dt = dt_ # based on the real execution of programs under staliro
    simTime = 10 # 30 was so long for mpc ; equal to 3000 steps with dt=0.01
    sim_count = sim_count  # 100 # num of simulations
    sim_steps = steps # 600# simTime = sim_steps * dt = simTime (Staliro default value); we go with 10 sec
    'ranges for sampling based on Arch 2014 benchmark paper and Arch 2019 falsification report'
    r_sampler = sampler_obj
    mpc = mpc_obj

    'Starting Matlab engine'
    eng = matlab.engine.start_matlab()
    # eng.cd(r'/home/sanaz/Documents/Projects/STaliro/benchmarks/ARCH2014/', nargout=0)
    eng.cd(r'/home/sanaz/Documents/Projects/s-taliro_public/trunk/benchmarks/ARCH2019', nargout=0)

    # final_states = [] # Storing all states from all steps of all runs
    trajectories = []
    cmd_seq_list = []

    for i in range(sim_count):
        s = time.time()
        # print(f" run {i}")
        seed_ = random.randint(1, 2147483647)
        random.seed(seed_)

        ts_val = 0.0
        brake_ = 0.0
        throttle_ = 0.0
        brake_cmd_seq = [brake_]
        throttle_cmd_seq = [throttle_]
        time_steps = [ts_val]
        traj = [[0.0, 0.0, 1.0]] # state = [speed, rpm, gear] # TBD for initialization
        cmd_seq = []

        target_ = r_sampler.rejection_sampling(seed=seed_)
        target_state_ = np.transpose(mpc.normalize_state(np.array([target_])))

        print(f"run {i} target {target_}")

        'Plotting'
        traj_vel = [0.0]
        traj_rpm = [0.0]
        # traj_vel_pred = [0.0]
        # traj_rpm_pred = [0.0]

        'We need to iterate over timesteps becasue simulator does not keep '
        'state and we need to run with u0, u0u1, u0u1u2, ... to get states.'
        'So only the last state of the trace after each execution is new.'
        for j in range(sim_steps):

            'Generating control input to simulate the system'
            states = eng.test(throttle_cmd_seq, brake_cmd_seq, time_steps, len(time_steps) * dt) # trajectory of states
            states_ = mpc.normalize_state(np.array(deepcopy(states[-1])))

            # specs.check_falsified_specs(states[-1][0], states[-1][1], states[-1][2], ts_val, i) # For evaluation

            action, _, _, _ = mpc.predict(states_, target_state_) # no current state transpose for new obs_function

            if action is not None:
                throttle_, brake_ = action[0], action[1]
            else:
                print(f"prediction failed!")

            brake_cmd_seq.append(float(brake_))
            throttle_cmd_seq.append(float(throttle_))
            cmd_seq.append([throttle_, brake_])
            traj.append(deepcopy(list(states[-1])))

            ts_val += dt
            time_steps.append(ts_val)

            # 'Storing all states from all steps of all runs'
            # final_states.append([float(states[-1][0]), float(states[-1][1]), float(states[-1][2])])

        trajectories.append(np.array(traj))
        cmd_seq_list.append(np.array(cmd_seq))

        e = time.time()
        # print(f"Run {i}  runt-time {e-s}")

    # specs.print_falsification_result() # Do it when all sim_count are done

    # fname = 'states_mpc_' + str(seed_)+'.pkl'
    # with open(fname, 'wb') as fhandle:
    #     pickle.dump(final_states, fhandle)

    eng.quit()

    return trajectories, cmd_seq_list



def main():

    'Collecting training data'
    train_set_states, train_set_cmdseq = get_training_data()

    'Running the system'
    global mpc
    mpc = mpc_cvxpy.MPC(train_set_states, train_set_cmdseq)


    for i in range(5):
        print(f"Round {i}")
        run_automatic_transmission()




if __name__ == '__main__':
    main()