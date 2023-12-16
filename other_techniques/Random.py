import os
import sys
import pickle

# explorer_path = '/home/sanaz/Documents/Projects/CoverageGuidedExplorer'
explorer_path = os.path.dirname(os.getcwd())
benchmark_path = explorer_path + '/benchmarks'
utility_path = explorer_path  + '/utility'
sys.path.append(explorer_path)
sys.path.append(benchmark_path)
sys.path.append(utility_path)

import simulation_handler as sh

def main(benchmark=None):
    run_sim_count = 100  # number of simulations for test runs

    if benchmark is None:
        benchmark = sys.argv[1]

    if benchmark == 'kinematic_car':
        print(f"kinematic_car benchmark")
        run_sim_count = 100  # number of simulations for test runs
        steps = 100
        dt= 0.1

    elif benchmark == 'ACASXU':
        print(f"ACASXU benchmark")
        run_sim_count = 100  # number of simulations for test runs
        steps = 120
        dt= 1.0

    elif benchmark == 'automatic_transmission':
        print(f"automatic_transmission benchmark")
        run_sim_count = 100  # number of simulations for test runs
        steps = 100
        dt= 0.01

    elif benchmark == 'point_mass':
        print(f"point_mass benchmark")
        run_sim_count = 100  # number of simulations for test runs
        steps = 100
        dt= 0.01


    sim_handler = sh.Simulation_Handler(benchmark)
    all_traj_states, _  = sim_handler.run_random(benchmark,
                                                 [run_sim_count, steps, dt])

    fname = utility_path + '/results/' + benchmark + '_random.pkl'
    with open(fname, 'wb') as fhandle:
        pickle.dump(all_traj_states, fhandle)



if __name__ == "__main__":
    main()