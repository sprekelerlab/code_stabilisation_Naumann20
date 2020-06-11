import matplotlib.pyplot as plt
import numpy as np
import time
import rate_network_cython as ratenet


def create_synapses(N_pre, N_post, c, autapse=True):
    """
    Create random connections between two groups or within one group.
    :param N_pre: number of neurons in pre group
    :param N_post: number of neurons in post group
    :param c:   connectivity
    :param autapse: whether to allow autapses (connection of neuron to itself if pre = post population)
    :return: 2xN_con array of connection indices (pre-post pairs)
    """

    indegree = int(np.round(c*N_pre))  # no variance of indegree = fixed indegree

    i = np.array([], dtype=int)
    j = np.array([], dtype=int)

    for n in range(N_post):

        if not autapse: # if autapses are disabled, remove index of present post neuron from pre options
            opts = np.delete(np.arange(N_pre, dtype=int), n)
        else:
            opts = np.arange(N_pre, dtype=int)

        pre = np.random.choice(opts, indegree, replace=False)

        # add connection indices to list
        i = np.hstack((i, pre))
        j = np.hstack((j, np.repeat(n, indegree)))

    return np.array([i, j])


if __name__ in "__main__":

    # ------------------------------------------- #
    #  Set up parameters and network connections  #
    # ------------------------------------------- #

    # Network paramters
    N_e = 1024  # number of exc neurons
    N_i = 256  # number of inh neurons
    c = 100*2**(-10)  # connection probability (almost 0.1 but gives integer indegree)

    # Generate lists pre-post connection pairs
    con_ee = create_synapses(N_e, N_e, c, autapse=False)
    con_ei = create_synapses(N_i, N_e, c)
    con_ie = create_synapses(N_e, N_i, c)
    con_ii = create_synapses(N_i, N_i, c, autapse=False)

    # Parameters are passed as a parameter dictionary to the run function

    # network parameters and connection lists
    params = {'N_e': N_e, 'N_i': N_i, 'c': c,
              'con_ee': con_ee, 'con_ei': con_ei, 'con_ie': con_ie, 'con_ii': con_ii}

    # synaptic weight parameters
    # -w_ee_init gives EE weights if plasticity is off
    # -J is the overall recurrence (scales all weights), if 1 then has no effect
    params.update({'w_ee_init': 2, 'w_ee_init_noise': 0.1, 'w_ei': 1, 'w_ie': 1.5, 'w_ii': 0.5, 'J': 1})

    # simulation parameters (time in seconds)
    # - pre_sim: duration of running network before turning plasticity on
    # - dt_rec: interval at which rates and weights are recorded
    params.update({'duration': 20, 'dt': 0.001, 'pre_sim': 10, 'dt_rec': 0.01})

    # plasticity parameters
    params.update({'plastic': False, 'tau_w': 300, 'tau_c': 5, 'eta': 10, 'kappa': 5})

    # input parameters and initial rates
    params.update({'stim_strength': 0.5, 'noise_strength': 0, 're_init': 5, 'ri_init': 5})

    # presynaptic inhibition parameters
    # - g_func determines the transfer function type: linear (0), exponential (1) or sigmoid (2)
    # - shift parameter is only relevant for the sigmoid transfer function
    params.update({'beta': 0.05, 'tau_p': 0.5, 'g_func': 0, 'shift': 0})

    # --------------------------------- #
    #  Run simulation and time runtime  #
    # --------------------------------- #

    print('Running network...')
    start = time.time()
    re, ri, W, p, t = ratenet.c_run_ratenet(params)
    stop = time.time()
    print(f"Simulation took {stop-start: 1.1f} s")

    # --------------------------------- #
    #  Plot results of the simulation   #
    # --------------------------------- #

    print('Plotting results...')
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(5, 4), dpi=150,
                           gridspec_kw={'hspace': 0.3})
    ax[0].plot(t, re, lw=1, c='salmon', alpha=0.2)
    ax[0].plot(t, ri, lw=1, c='lightblue', alpha=0.2)
    ax[0].plot(t, np.mean(ri, axis=1), lw=2, c='C0', label='inh mean')
    ax[0].plot(t, np.mean(re, axis=1), lw=2, c='C3', label='exc mean')
    ax[0].legend(loc='best')
    ax[0].set(ylabel='inh/exc rates')
    ax[1].plot(t, p, c='C2')
    ax[1].set(ylim=[0, 1])
    ax[1].set(ylabel='release prob.')
    ax[2].plot(t, W[:, :500], lw=1, c='gray', alpha=0.5)
    ax[2].plot(t, W.mean(axis=1), lw=2, c='C1')
    ax[2].set(xlabel='time [min]', ylabel='EE weights', ylim=[0, 0.1])
    plt.show()  # might take a while
