import sys
import os
import matlab.engine
import importlib



# explorer_path = '/home/sanaz/Documents/Projects/CoverageGuidedExplorer'
explorer_path = os.path.dirname(os.getcwd())
sampling_path = explorer_path + '/sampling'
sys.path.append(sampling_path)
sys.path.append(explorer_path + '/benchmarks/kinematic_car/')
sys.path.append(explorer_path + '/benchmarks/acasxu/acasxu_dubins/')
sys.path.append(explorer_path + '/benchmarks/automatic_transmission/')
sys.path.append(explorer_path + '/benchmarks/point_mass/')



'import case studies'
import sampling as rs  # rs for rejection sampling
import kinematic_car as kc
import kinematic_car_random as kcr
import acasxu_dubins_mpc as acasxu
import acasxu_dubins_random as acasxur
import automatic_transmission as at
import automatic_transmission_rand as atr
import point_mass_mpc as pm
import point_mass_random as pmr


class Simulation_Handler:

    mpc = None
    rsampler = None
    curr_run = 0
    benchmark = ''

    def __init__(self, benchmark=None):
        self.benchmark = benchmark

        system_path = os.path.join(explorer_path, 'benchmarks', benchmark)
        sys.path.append(system_path)
        self.system_module = importlib.import_module(
            f'benchmarks.{benchmark}.{benchmark}_main')


    def update_mpc(self, mpc_):
        self.mpc = mpc_


    def update_sampler(self, trajectories):

        self.rsampler.update_kdt(trajectories)


    def make_sampler(self, lower_bound=None, upper_bound=None, trajectories=None, convexHull=None):
        # print(f"sim_Handler::update_sampler")
        'Make the sampler object either from traiing data to grow the koopman model or get it from the last determined boxbound'
        if trajectories:
            self.rsampler = rs.Sampling(benchmark=self.benchmark, trajectories=trajectories)
        elif convexHull:
            self.rsampler = rs.Sampling(benchmark=self.benchmark, convexHull=convexHull)
        elif lower_bound and upper_bound:
            self.rsampler = rs.Sampling(benchmark=self.benchmark, lower_bound=lower_bound, upper_bound=upper_bound)
        else:
            'make sampler from the box bound of the last koopman training data'
            lower_bound_, upper_bound_ = self.get_boxbound()
            self.rsampler = rs.Sampling(benchmark=self.benchmark, lower_bound= lower_bound_, upper_bound=upper_bound_)



    def get_boxbound(self):

        'make sampler from the box bound of the last koopman training data'
        if self.rsampler:
            return self.rsampler.get_boxbound()

        return None, None


    def get_training_data(self, benchmark, sim_count, steps, dt):
        # print(f"sim_handler::get_training_data sim_count  {sim_count}")
        traindata_states, traindata_inputs = [], []

        if benchmark == 'kinematic_car':
            traindata_states, traindata_inputs = kc.run_training(sim_count, steps, dt)

        elif benchmark == 'ACASXU':
            traindata_states, traindata_inputs = acasxu.get_training_data(sim_count)

        elif benchmark == 'automatic_transmission':
            traindata_states, traindata_inputs = at.get_training_data(sim_count, steps, dt)

        return  traindata_states, traindata_inputs



    def run(self, benchmark, args=None):
        all_traj_states, all_traj_inputs = [], []

        if benchmark == 'kinematic_car':
            all_traj_states, all_traj_inputs = kc.run_kinematic_car(sim_count=args[0], steps=args[1], mpc_obj=self.mpc,
                                                                    sampler_obj=self.rsampler)
        elif benchmark == 'ACASXU':
            all_traj_states, all_traj_inputs = acasxu.run_acasxu(sim_count=args[0], train=False, mpc_obj=self.mpc,
                                                                 sampler_obj=self.rsampler)
        elif benchmark == 'automatic_transmission':
            all_traj_states, all_traj_inputs = at.run_automatic_transmission(sim_count=args[0], steps=args[1], dt_=args[2],
                                                                             mpc_obj=self.mpc, sampler_obj=self.rsampler)
        elif benchmark == 'point_mass':
            all_traj_states, all_traj_inputs = pm.run()
            # self.system_module.run()

        return all_traj_states, all_traj_inputs



    def run_singleStep_kinematic_car(self, state_, acc_cmd, steer_cmd):

        state_out = kc.run_singleStep(state_, acc_cmd, steer_cmd)

        return state_out



    def run_random(self, benchmark, args=None):

        all_traj_states, all_traj_inputs = [], []

        if benchmark == 'kinematic_car':
            all_traj_states, all_traj_inputs = kcr.run_kinematic_car(sim_count=args[0], steps=args[1], dt_=args[2])

        elif benchmark == 'ACASXU':
            all_traj_states, all_traj_inputs = acasxur.run_acasxu(sim_count=args[0], steps=args[1], dt_=args[2])

        elif benchmark == 'automatic_transmission':
            all_traj_states, all_traj_inputs = atr.run_automatic_transmission(sim_count=args[0], steps=args[1], dt_=args[2])

        elif benchmark == 'point_mass':
            pmr.run()



        return all_traj_states, all_traj_inputs



    def run_staliro(self, benchmark, args=None):

        if benchmark == 'kinematic_car':
            pass

        elif benchmark == 'ACASXU':
            pass

        elif benchmark == 'automatic_transmission':
            'Starting Matlab engine'
            eng = matlab.engine.start_matlab()
            # eng.cd(r'/home/sanaz/Documents/Projects/s-taliro_public/trunk/benchmarks/ARCH2019/', nargout=0)
            eng.cd(r'/home/sanaz/Documents/Projects/s-taliro_public/trunk/benchmarks/ARCH2019/', nargout=0)

            'Simulating in Matlab & processing the simulation data'
            # eng.run_automatic_transmission(args[0], args[1], args[2])
            eng.staliro_run_trans_SOAR_s6()

            eng.quit()
        # return all_traj_states, all_traj_inputs