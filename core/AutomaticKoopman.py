import numpy as np
import pickle
from autokoopman import auto_koopman
import autokoopman.core.trajectory as traj


def get_model(train_states, train_cmds, steps, dt):
    data = []
    for i in range(len(train_states)):
        states = train_states[i]
        inputs = train_cmds[i]
        times = np.linspace(0, len(states) * dt, len(states))
        data.append(traj.Trajectory(times[:-1], np.asarray(states[:-1]),
                                    np.asarray(inputs)))

    # construct training data object
    ids = np.arange(0, len(data)).tolist()
    training_data = traj.TrajectoriesData(dict(zip(ids, data)))

    # learn model from data
    experiment_results = auto_koopman(
        training_data,
        sampling_period=dt,
        obs_type='rff',  # observable function (random Fourier features)
        opt='grid',  # method used for parameter optimization
        n_obs= 50,    # number of observable functions (= dimension of high-dim.)
        rank=(1,5),   # ranks (default is set to maximum n_obs which messes with A)
        normalize=False,  # normalize the data
        verbose=True
    )
    # get the model from the experiment results
    model = experiment_results['tuned_model']

    return model



def save_model(fname, model):
    sub_model = [model.A, model.B, model.obs]
    with open(fname,'wb') as fhandle:
        pickle.dump(sub_model, fhandle)



def get_model_from_file(fname):
    with open(fname, 'rb') as fhandle:
        model = pickle.load(fhandle)

    'test the model'
    print(f"Reading the model from file {fname}")
    s = np.array([[0.0, 0.0, 15.0, 0.0]])
    print(f"model.A  {model[0]}  \n  model.B {model[1]}  "
          f"\n  obs = {model[2].obs_fcn(s)}")

