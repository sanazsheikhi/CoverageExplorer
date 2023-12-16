'''
ACASXu neural networks closed loop simulation with dubin's car dynamics

Used for falsification, where the opponent is allowed to maneuver over time
'''
import pickle
from functools import lru_cache
import math
import numpy as np
from scipy import ndimage
from scipy.linalg import expm
# import mpc_cvxpy
import matplotlib.pyplot as plt
from matplotlib import patches, animation
from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox, TransformedBbox
from matplotlib.collections import LineCollection
from matplotlib.path import Path
from matplotlib.lines import Line2D
import onnxruntime as ort
from numba import njit

from dataclasses import dataclass

acasxu_path = '/home/sanaz/Documents/Projects/CoverageGuidedExplorer/benchmarks/acasxu/acasxu_dubins/'




def init_plot():
    'initialize plotting style'

    #matplotlib.use('TkAgg') # set backend

    p = '/home/sanaz/Documents/Projects/CoverageGuidedExplorer/benchmarks/acasxu/acasxu_dubins/bak_matplotlib.mlpstyle'
    plt.style.use(['bmh', p])

def load_network(last_cmd):
    '''load the one neural network as a 2-tuple (range_for_scaling, means_for_scaling)'''

    # onnx_filename = f"ACASXU_run2a_{last_cmd + 1}_1_batch_2000.onnx"
    onnx_filename = f"/home/sanaz/Documents/Projects/CoverageGuidedExplorer/benchmarks" \
                    f"/acasxu/acasxu_dubins/ACASXU_run2a_{last_cmd + 1}_1_batch_2000.onnx"

    #print(f"Loading {mat_filename}...")
    #matfile = loadmat(mat_filename)
    #range_for_scaling = matfile['range_for_scaling'][0]
    #means_for_scaling = matfile['means_for_scaling'][0]
    #mat_filename = f"ACASXU_run2a_1_1_batch_2000.mat"

    means_for_scaling = [19791.091, 0.0, 0.0, 650.0, 600.0, 7.5188840201005975]
    range_for_scaling = [60261.0, 6.28318530718, 6.28318530718, 1100.0, 1200.0]

    session = ort.InferenceSession(onnx_filename)

    # warm up the network
    i = np.array([0, 1, 2, 3, 4], dtype=np.float32)
    i.shape = (1, 1, 1, 5)
    session.run(None, {'input': i})

    return session, range_for_scaling, means_for_scaling

def load_networks():
    '''load the 5 neural networks into nn-enum's data structures and return them as a list'''

    nets = []

    for last_cmd in range(5):
        nets.append(load_network(last_cmd))

    return nets

def get_time_elapse_mat(command1, dt, command2=0):
    '''get the matrix exponential for the given command

    state: x, y, vx, vy, x2, y2, vx2, vy2 
    '''

    y_list = [0.0, 1.5, -1.5, 3.0, -3.0]
    y1 = y_list[command1]
    y2 = y_list[command2]
    
    dtheta1 = (y1 / 180 * np.pi)
    dtheta2 = (y2 / 180 * np.pi)

    a_mat = np.array([
        [0, 0, 1, 0, 0, 0, 0, 0], # x' = vx
        [0, 0, 0, 1, 0, 0, 0, 0], # y' = vy
        [0, 0, 0, -dtheta1, 0, 0, 0, 0], # vx' = -vy * dtheta1
        [0, 0, dtheta1, 0, 0, 0, 0, 0], # vy' = vx * dtheta1
    #
        [0, 0, 0, 0, 0, 0, 1, 0], # x' = vx
        [0, 0, 0, 0, 0, 0, 0, 1], # y' = vy
        [0, 0, 0, 0, 0, 0, 0, -dtheta2], # vx' = -vy * dtheta2
        [0, 0, 0, 0, 0, 0, dtheta2, 0], # vy' = vx * dtheta1
        ], dtype=float)

    return expm(a_mat * dt)


def run_network(network_tuple, x, stdout=False):
    'run the network and return the output'

    session, range_for_scaling, means_for_scaling = network_tuple

    # normalize input
    for i in range(5):
        x[i] = (x[i] - means_for_scaling[i]) / range_for_scaling[i]

    if stdout:
        print(f"input (after scaling): {x}")

    in_array = np.array(x, dtype=np.float32)
    in_array.shape = (1, 1, 1, 5)
    outputs = session.run(None, {'input': in_array})
        
    return outputs[0][0]

@njit(cache=True)
def state7_to_state5(state7, v_own, v_int):
    """compute rho, theta, psi from state7"""

    assert len(state7) == 7

    x1, y1, theta1, x2, y2, theta2, _ = state7

    rho = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    dy = y2 - y1
    dx = x2 - x1

    theta = np.arctan2(dy, dx)
    psi = theta2 - theta1

    theta -= theta1

    while theta < -np.pi:
        theta += 2 * np.pi

    while theta > np.pi:
        theta -= 2 * np.pi

    if psi < -np.pi:
        psi += 2 * np.pi

    while psi > np.pi:
        psi -= 2 * np.pi

    return np.array([rho, theta, psi, v_own, v_int])
 
    # Test of NN fuzzing by SNZ
    #global gseed
    #np.random.seed(gseed)
    #r = np.random.randint(-100, 100) 
    #t = np.random.randint(-100, 100)
    #s = np.random.randint(-500, 500)
    #z = np.random.rand() * 2 * np.pi
    #z = z * np.random.randint(-1, 2)
    #w = np.random.rand() * 2 * np.pi
    #w = w * np.random.randint(-1, 2)
    #return np.array([rho, theta + z, psi + w, v_own + r, v_int + t]) #SNZ test



@njit(cache=True)
def state7_to_state8(state7, v_own, v_int):
    """compute x,y, vx, vy, x2, y2, vx2, vy2 from state7"""

    assert len(state7) == 7

    x1 = state7[0]
    y1 = state7[1]
    vx1 = math.cos(state7[2]) * v_own
    vy1 = math.sin(state7[2]) * v_own

    x2 = state7[3]
    y2 = state7[4]
    vx2 = math.cos(state7[5]) * v_int
    vy2 = math.sin(state7[5]) * v_int

    return np.array([x1, y1, vx1, vy1, x2, y2, vx2, vy2])


@lru_cache(maxsize=None)
def get_airplane_img():
    """load airplane image form file"""
    
    img = plt.imread('/home/sanaz/Documents/Projects/CoverageGuidedExplorer/benchmarks/acasxu/acasxu_dubins/airplane.png')

    return img

def init_time_elapse_mats(dt):
    """get value of time_elapse_mats array"""

    rv = []

    for cmd in range(5):
        rv.append([])
        
        for int_cmd in range(5):
            mat = get_time_elapse_mat(cmd, dt, int_cmd)
            rv[-1].append(mat)

    return rv

@njit(cache=True)
def step_state(state7, v_own, v_int, time_elapse_mat, dt):
    """perform one time step with the given commands"""

    state8_vec = state7_to_state8(state7, v_own, v_int)

    s = time_elapse_mat @ state8_vec

    # extract observation (like theta) from state
    new_time = state7[-1] + dt
    theta1 = math.atan2(s[3], s[2])
    theta2 = math.atan2(s[7], s[6])
    rv = np.array([s[0], s[1], theta1, s[4], s[5], theta2, new_time])

    return rv

class State():
    'state of execution container'

    nets = load_networks()
    plane_size = 3000

    nn_update_rate = 1.0 # 2.0
    dt = 1.0 # 0.05

    v_own = 800 # ft/sec
    v_int = 800

    time_elapse_mats = init_time_elapse_mats(dt)

    

    def __init__(self, init_vec, save_states=False):
        assert len(init_vec) == 7, "init vec should have length 7"
        
        self.vec = np.array(init_vec, dtype=float) # current state
        self.next_nn_update = 0
        self.command = 0 # initial command

        # these are set when simulation() if save_states=True
        self.save_states = save_states
        self.vec_list = [] # state history
        self.commands = [] # commands history
        self.int_commands = [] # intruder command history

        # used only if plotting
        self.artists_dict = {} # set when make_artists is called
        self.img = None # assigned if plotting

        # assigned by simulate()
        self.u_list = []
        self.u_list_index = None
        self.min_dist = np.inf
        
        
       
        
    def artists_list(self):
        'return list of artists'

        return list(self.artists_dict.values())

    def set_plane_visible(self, vis):
        'set ownship plane visibility status'

        self.artists_dict['dot0'].set_visible(not vis)
        self.artists_dict['circle0'].set_visible(False) # circle always False
        self.artists_dict['lc0'].set_visible(True)
        self.artists_dict['plane0'].set_visible(vis)
        
    def update_artists(self, axes):
        '''update artists in self.artists_dict to be consistant with self.vec, returns a list of artists'''

        assert self.artists_dict
        rv = []

        x1, y1, theta1, x2, y2, theta2, _ = self.vec

        for i, x, y, theta in zip([0, 1], [x1, x2], [y1, y2], [theta1, theta2]):
            key = f'plane{i}'

            if key in self.artists_dict:
                plane = self.artists_dict[key]
                rv.append(plane)

                if plane.get_visible():
                    theta_deg = (theta - np.pi / 2) / np.pi * 180 # original image is facing up, not right
                    original_size = list(self.img.shape)
                    img_rotated = ndimage.rotate(self.img, theta_deg, order=1)
                    rotated_size = list(img_rotated.shape)
                    ratios = [r / o for r, o in zip(rotated_size, original_size)]
                    plane.set_data(img_rotated)

                    size = State.plane_size
                    width = size * ratios[0]
                    height = size * ratios[1]
                    box = Bbox.from_bounds(x - width/2, y - height/2, width, height)
                    tbox = TransformedBbox(box, axes.transData)
                    plane.bbox = tbox

            key = f'dot{i}'
            if key in self.artists_dict:
                dot = self.artists_dict[f'dot{i}']
                cir = self.artists_dict[f'circle{i}']
                rv += [dot, cir]

                dot.set_data([x], [y])
                cir.set_center((x, y))

        # line collection
        lc = self.artists_dict['lc0']
        rv.append(lc)

        int_lc = self.artists_dict['int_lc0']
        rv.append(int_lc)

        self.update_lc_artists(lc, int_lc)

        return rv

    def update_lc_artists(self, own_lc, int_lc):
        'update line collection artist based on current state'

        assert self.vec_list

        for lc_index, lc in enumerate([own_lc, int_lc]):
            paths = lc.get_paths()
            colors = []
            lws = []
            paths.clear()
            last_command = -1
            codes = []
            verts = []

            for i, vec in enumerate(self.vec_list):
                if np.linalg.norm(vec - self.vec) < 1e-6:
                    # done
                    break

                if lc_index == 0:
                    cmd = self.commands[i]
                else:
                    cmd = self.int_commands[i]

                x = 0 if lc_index == 0 else 3
                y = 1 if lc_index == 0 else 4

                # command[i] is the line from i to (i+1)
                if cmd != last_command:
                    if codes:
                        paths.append(Path(verts, codes))

                    codes = [Path.MOVETO]
                    verts = [(vec[x], vec[y])]

                    if cmd == 1: # weak left
                        lws.append(2)
                        colors.append('b')
                    elif cmd == 2: # weak right
                        lws.append(2)
                        colors.append('c')
                    elif cmd == 3: # strong left
                        lws.append(2)
                        colors.append('g')
                    elif cmd == 4: # strong right
                        lws.append(2)
                        colors.append('r')
                    else:
                        assert cmd == 0 # coc
                        lws.append(2)
                        colors.append('k')

                codes.append(Path.LINETO)

                verts.append((self.vec_list[i+1][x], self.vec_list[i+1][y]))

            # add last one
            if codes:
                paths.append(Path(verts, codes))

            lc.set_lw(lws)
            lc.set_color(colors)

    def make_artists(self, axes, show_intruder):
        'make self.artists_dict'

        assert self.vec_list
        self.img = get_airplane_img()

        posa_list = [(v[0], v[1], v[2]) for v in self.vec_list]
        posb_list = [(v[3], v[4], v[5]) for v in self.vec_list]
        
        pos_lists = [posa_list, posb_list]

        if show_intruder:
            pos_lists.append(posb_list)

        for i, pos_list in enumerate(pos_lists):
            x, y, theta = pos_list[0]
            
            l = axes.plot(*zip(*pos_list), f'c-', lw=0, zorder=1)[0]
            l.set_visible(False)
            self.artists_dict[f'line{i}'] = l

            if i == 0:
                lc = LineCollection([], lw=2, animated=True, color='k', zorder=1)
                axes.add_collection(lc)
                self.artists_dict[f'lc{i}'] = lc

                int_lc = LineCollection([], lw=2, animated=True, color='k', zorder=1)
                axes.add_collection(int_lc)
                self.artists_dict[f'int_lc{i}'] = int_lc

            # only sim_index = 0 gets intruder aircraft
            if i == 0 or (i == 1 and show_intruder):
                size = State.plane_size
                box = Bbox.from_bounds(x - size/2, y - size/2, size, size)
                tbox = TransformedBbox(box, axes.transData)
                box_image = BboxImage(tbox, zorder=2)

                theta_deg = (theta - np.pi / 2) / np.pi * 180 # original image is facing up, not right
                img_rotated = ndimage.rotate(self.img, theta_deg, order=1)

                box_image.set_data(img_rotated)
                axes.add_artist(box_image)
                self.artists_dict[f'plane{i}'] = box_image

            if i == 0:
                dot = axes.plot([x], [y], 'k.', markersize=6.0, zorder=2)[0]
                self.artists_dict[f'dot{i}'] = dot

                rad = 2500
                c = patches.Ellipse((x, y), rad, rad, color='k', lw=3.0, fill=False)
                axes.add_patch(c)
                self.artists_dict[f'circle{i}'] = c



    def step(self, train, target, mpc_obj):
        'execute one time step and update the model'

        cmd_val = [0.0, 1.5, -1.5, 3.0, -3.0]
        tol = 1e-6

        if self.next_nn_update < tol:
            assert abs(self.next_nn_update) < tol, f"time step doesn't sync with nn update time. " + \
                      f"next update: {self.next_nn_update}"
            # update command
            self.update_command()
            self.next_nn_update = State.nn_update_rate

        self.next_nn_update -= State.dt
        intruder_cmd = self.u_list[self.u_list_index]


        if train == False:
            mpc = mpc_obj

            'For optimizer'
            """ 
                We put the intruder state first  then the ownship states 
                in the train data record (system_states) because these data 
                will be used for clustering and making convex. The ownship coordinates, 
                specially the first iterations are not proper and make convexhull 
                creation fail. Also, using the intruder states for data refinemnt
                (cluster & convex) is Okay as we train the model on both intruder 
                and ownship states
            """

            states_aircraft = mpc.normalize_state(np.array([[self.vec[3], self.vec[4], self.vec[5],
                                                             self.vec[0], self.vec[1], self.vec[2]]]))
            #states_arcraft = mpc.normalize_state(np.array([[self.vec[0], self.vec[1], self.vec[2],
            #                                                self.vec[3], self.vec[4], self.vec[5]]]))
            state_target = mpc.normalize_state(np.array([[target[0], target[1]]]))
            predicted_cmd, _, _, _ = mpc.predict(states_aircraft, np.transpose(state_target),
                                                 dimension_state=2, dimension_input=1)

            'Added by Sanaz to handle the situation where MPC can not solve the problem; use previous command'
            if predicted_cmd == None:
                predicted_cmd = [self.command]
                print(f"MPC failed to solve the problem. Using previous input command {self.command}.")


            intruder_cmd = self.map_cmd(predicted_cmd)[0]  # cmd value to cmd index


        if self.save_states:
            self.commands.append(self.command)
            self.int_commands.append(intruder_cmd)

        time_elapse_mat = State.time_elapse_mats[self.command][intruder_cmd]
        self.vec = step_state(self.vec, State.v_own, State.v_int, time_elapse_mat, State.dt)
         
        return cmd_val[intruder_cmd]
 


    def simulate(self, cmd_list, train, target=None, mpc_obj=None):
        '''
        simulate system
        saves result in self.vec_list
        also saves self.min_dist
        '''

        step_count = 0 # test

        'Data for training Auto-Koopman'
        system_states = []
        cmds_int = []


        self.u_list = cmd_list
        self.u_list_index = None


        assert isinstance(cmd_list, list)
        tmax = len(cmd_list) * State.nn_update_rate
        t = 0.0

        if self.save_states:
            rv = [self.vec.copy()]

        prev_dist_sq = (self.vec[0] - self.vec[3])**2 + (self.vec[1] - self.vec[4])**2

        while t + 1e-6 < tmax:

            'Data for training the Auto Koopman operator and test mpc'
            """ We put the intruder state first  then the ownship states 
            in the train data record (system_states) because these data 
            will be used for clustering and making convex. The ownship coordinates, 
            specially the first iterations are not proper and make convexhull 
            creation fail. Also, using the intruder states for data refinemnt
            (cluster & convex) is Okay as we train the model on both intruder 
            and ownship states"""

            system_states.append([self.vec[3], self.vec[4], self.vec[5],
                                  self.vec[0], self.vec[1], self.vec[2]])
            # system_states.append([self.vec[0], self.vec[1], self.vec[2],
            #                       self.vec[3], self.vec[4], self.vec[5]]) #


            cmd = self.step(train, target, mpc_obj)
            cmds_int.append([cmd])

            step_count += 1

            cur_dist_sq = (self.vec[0] - self.vec[3])**2 + (self.vec[1] - self.vec[4])**2

            if self.save_states:
                rv.append(self.vec.copy())

            t += State.dt

            prev_dist_sq = cur_dist_sq

        self.min_dist = math.sqrt(prev_dist_sq)
          

        if self.save_states:
            self.vec_list = rv

        if not self.save_states:
            assert not self.vec_list
            assert not self.commands
            assert not self.int_commands


        collision = False
        if self.min_dist <= 500:
            collision = True

        
        deviation = False
        if self.vec[0] > 2 or self.vec[0] < -2:
            deviation = True


        return collision, deviation, np.array(system_states), np.array(cmds_int), step_count



    def update_command(self):
        'update command based on current state'''

        rho, theta, psi, v_own, v_int = state7_to_state5(self.vec, State.v_own, State.v_int)

        # 0: rho, distance
        # 1: theta, angle to intruder relative to ownship heading
        # 2: psi, heading of intruder relative to ownship heading
        # 3: v_own, speed of ownship
        # 4: v_int, speed in intruder

        # min inputs: 0, -3.1415, -3.1415, 100, 0
        # max inputs: 60760, 3.1415, 3,1415, 1200, 1200

        if rho > 60760:
            self.command = 0
        else:
            last_command = self.command

            net = State.nets[last_command]

            state = [rho, theta, psi, v_own, v_int]

            res = run_network(net, state)
            self.command = np.argmin(res)

            #names = ['clear-of-conflict', 'weak-left', 'weak-right', 'strong-left', 'strong-right']

        if self.u_list_index is None:
            self.u_list_index = 0
        else:
            self.u_list_index += 1

            # repeat last command if no more commands
            self.u_list_index = min(self.u_list_index, len(self.u_list) - 1)



    def _plot(self):
        
        global points_x
        global points_y
   
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('ACAS-Xu ownship coverage')
        plt.scatter(np.array(points_x), np.array(points_y), s=5, color='red')

        plt.show()      


    def map_cmd(self, cval_sequence):
        'maps a cmd value to a cmd index'

        s = cval_sequence

        for i in range(len(s)):

            if s[i] < -0.2 and s[i] >= -1.8:    s[i] = 2  # wr
            elif s[i] < -1.8 and s[i] >= -3.1:  s[i] = 4  # sr
            elif s[i] > 0.2 and s[i] <= 1.8:    s[i] = 1  # wl
            elif s[i] > 1.8 and s[i] <= 3.1:    s[i] = 3  # sl
            else:                               s[i] = 0  # coc

        return s




def plot(s, save_mp4=False, target=None):
    """plot a specific simulation"""
    init_plot()

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(15, 12))
    axes.axis('equal')

    axes = plt.gca()
    plt.locator_params(axis='x', nbins=5)
    plt.locator_params(axis='y', nbins=5)
    axes.set_xlabel('X Position (ft)', fontsize=35, fontweight='bold')
    axes.set_ylabel('Y Position (ft)', fontsize=35, fontweight='bold')
    xticks, _ = plt.xticks()
    yticks, _ = plt.yticks()
    plt.xticks(fontsize=35, fontweight='bold')
    plt.yticks(fontsize=35, fontweight='bold')

    time_text = axes.text(0.02, 0.98, 'Time: 0', horizontalalignment='left', fontsize=14,
                          verticalalignment='top', transform=axes.transAxes)
    time_text.set_visible(True)

    custom_lines = [Line2D([0], [0], color='g', lw=2),
                    Line2D([0], [0], color='b', lw=2),
                    Line2D([0], [0], color='k', lw=2),
                    Line2D([0], [0], color='c', lw=2),
                    Line2D([0], [0], color='r', lw=2)]

    # axes.legend(custom_lines, ['Strong Left', 'Weak Left', 'Clear of Conflict', 'Weak Right', 'Strong Right'], \
    #             fontsize=20, loc='lower left')
    axes.legend(custom_lines, ['SL', 'WL', 'COC', 'WR', 'SR'], fontsize=20, loc='lower left')

    s.make_artists(axes, show_intruder=True)
    states = [s]

    plt.tight_layout()

    num_steps = len(states[0].vec_list)
    interval = 20  # ms per frame
    freeze_frames = 10 if not save_mp4 else 80

    num_runs = 1  # 3
    num_frames = num_runs * num_steps + 2 * num_runs * freeze_frames

    #plt.savefig('plot.png')



    def animate(f):
        'animate function'

        if not save_mp4:
            f *= 5 # multiplier to make animation faster

        if (f+1) % 10 == 0:
            print(f"Frame: {f+1} / {num_frames}")

        run_index = f // (num_steps + 2 * freeze_frames)

        f = f - run_index * (num_steps + 2*freeze_frames)

        f -= freeze_frames

        f = max(0, f)
        f = min(f, num_steps - 1)

        num_states = len(states)

        if f == 0:
            # initiaze current run_index
            show_plane = num_states <= 10
            for s in states[:num_states]:
                s.set_plane_visible(show_plane)

            for s in states[num_states:]:
                for a in s.artists_list():
                    a.set_visible(False)

        time_text.set_text(f'Time: {f * State.dt:.1f}')

        artists = [time_text]

        for s in states[:num_states]:
            s.vec = s.vec_list[f]
            artists += s.update_artists(axes)

        for s in states[num_states:]:
            artists += s.artists_list()

        return artists

    my_anim = animation.FuncAnimation(fig, animate, frames=num_frames,
                                      interval=interval, blit=True, repeat=True)

    if save_mp4:
        writer = animation.writers['ffmpeg'](fps=50,
                                             metadata=dict(artist='Stanley Bak'),
                                             bitrate=1800)

        my_anim.save('sim.mp4', writer=writer)
    else:
        # plt.savefig('plot_mpc.pdf', format="pdf")
        plt.show()


def make_random_input(seed, num_inputs=100):
    """make a random input for the system"""

    np.random.seed(seed) # deterministic random numbers

    # state vector is: x, y, theta, x2, y2, theta2, time
    init_vec = np.zeros(7)
    init_vec[2] = np.pi / 2 # ownship moving up initially

    radius = 60760 + np.random.random() * 2400  # [0, 63160]
    angle = np.random.random() * 2 * np.pi

    int_x = radius * np.cos(angle)
    int_y = radius * np.sin(angle)

    'Trying to get the intruder to the upper half of the plain where ego operates'
    while (int_y < 0):
        angle = np.random.random() * 2 * np.pi
        int_y = radius * np.sin(angle)

    int_heading = np.random.random() * 2 * np.pi
    
    init_vec[3] = int_x
    init_vec[4] = int_y
    init_vec[5] = int_heading

    # intruder commands for every control period (0 to 4)
    cmd_list = []

    for _ in range(num_inputs):
        cmd_list.append(np.random.randint(5))

    return init_vec, cmd_list



def get_training_data(count=100):
    'main entry point'

    print("ACASXU:: get_training_data ")

    sim_states = []
    sim_cmds = []
    collision_deviation = 0

    i = 0
    while len(sim_states) < count:
        seed = i
        'We choose len control input sequence to be 120, equivalent to two minutes'
        init_vec, cmd_list = make_random_input(seed,120)
        i += 1

        # reject start states where initial command is not clear-of-conflict
        state5 = state7_to_state5(init_vec, State.v_own, State.v_int)
        res = run_network(State.nets[0], state5)
        command = np.argmin(res)

        if command != 0:
            continue

        s = State(init_vec, save_states=True)
        collison, deviation, system_states, cmds,  _ = s.simulate(cmd_list, True)

        # plot(s)

        #reject simulations where the minimum distance was near the start
        if s.vec[-1] < 3.0:
            continue

        if deviation or collison:
            collision_deviation += 1

        # if system_states.shape[1] > 2:
        """ 
            As in this project we use the similarity of trajectories to cluster them,
            trajectories should have same length. On the othe hand number of steps of execution 
            (in simulate()), relates to the length of cmd_list.
        """

        if system_states.shape[0] == len(cmd_list):
            sim_states.append(system_states)
            sim_cmds.append(cmds[0:cmds.shape[0]-1])


    return sim_states, sim_cmds



def run_acasxu(sim_count=100, train=False, mpc_obj=None, sampler_obj=None):

    'main entry point'

    sim_states = []
    sim_cmds = []

    current_timesteps = 0
    seed = 0
    r_sampler = sampler_obj


    # for seed in range(0, num_sims):
    'To get diverse seeds in each set of execution we start with a new set of seed'
    # while current_timesteps < total_timesteps:
    while  len(sim_states) < sim_count:
        'We choose len control input sequence to be 120, equivalent to two minutes'
        init_vec, cmd_list = make_random_input(seed, 120)
        seed += 1

        # reject start states where initial command is not clear-of-conflict
        state5 = state7_to_state5(init_vec, State.v_own, State.v_int)
        res = run_network(State.nets[0], state5)
        command = np.argmin(res)

        if command != 0:
            continue

        target = r_sampler.rejection_sampling(seed=seed)


        s = State(init_vec, save_states=True)
        (collision, deviation, system_states, int_cmd_seq,
         step_count) = s.simulate(cmd_list, train, target, mpc_obj)

        # reject simulations where the minimum distance was near the start
        if s.vec[-1] < 3.0:
            continue


        current_timesteps += step_count

        """ 
        As in this project we use the similarity of trajectories to cluster 
        them, trajectories should have same length. On the othe hand number 
        of steps of execution (in simulate()), relates to the length of cmd_list.
        """
        if system_states.shape[0] == len(cmd_list):
            sim_states.append(system_states)
            sim_cmds.append(int_cmd_seq[0:int_cmd_seq.shape[0] - 1])
            # plot(s, False, target)

    return sim_states, sim_cmds






