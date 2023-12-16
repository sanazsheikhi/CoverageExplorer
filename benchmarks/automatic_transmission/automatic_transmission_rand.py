import matlab.engine
import numpy as np
import random
import Specification as specification
import matplotlib.pyplot as plt


def plot_2(trajectories, target=None):

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
        plt.plot(T[0], T[1])

    plt.show()



def run_automatic_transmission(sim_count, steps, dt_):

    'Running simulations with random control input'
    'Control inputs: throttle & brake; Output States: speed & RPM'

    dt = dt_
    sim_steps = steps # size of data trace; we want to go with 10 sec
    sim_count = sim_count # num of simulations
    all_traj_states, all_traj_inputs = [], []

    'Starting Matlab engine'
    eng = matlab.engine.start_matlab()
    # eng.cd(r'/home/sanaz/Documents/Projects/STaliro/benchmarks/ARCH2014/', nargout=0)
    eng.cd(r'/home/sanaz/Documents/Projects/s-taliro_public/trunk/benchmarks/ARCH2019', nargout=0)


    for j in range(sim_count):

        print(f"run {j}")
        specs = specification.Specs()
        seed_ = random.randint(1, 2147483647)
        np.random.seed(seed_)
        brake = [0.0]
        throttle = [0.0]
        'Generating control input to simulate the system'
        time_steps = np.linspace(0, dt*sim_steps, num=sim_steps).tolist()
        brake = brake + np.random.uniform(0, 350, sim_steps-1).tolist()
        throttle = throttle + np.random.uniform(0, 100, sim_steps-1).tolist()
        inputs = list(map(lambda a, b: [a, b], brake, throttle))

        'Simulating in Matlab & processing the simulation data'
        states = eng.test(throttle, brake, time_steps, sim_steps*dt)

        all_traj_states.append(states)
        all_traj_inputs.append(inputs)


        # trajectories.append([traj_vel, traj_rpm])
        # plot_2([[traj_vel, traj_rpm]])
    # plot_2(trajectories)

    # fname = 'states_random_' + str(seed_) + '.pkl'
    # with open(fname, 'wb') as fhandle:
    #     pickle.dump(all_traj_states, fhandle)

    eng.quit()

    return all_traj_states, all_traj_inputs




def main():

    for i in range(5):
        print(f"Round {i}")
        run_automatic_transmission()



if __name__ == '__main__':
    main()