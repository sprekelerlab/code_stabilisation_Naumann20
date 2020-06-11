import brian2 as b2
from brian2 import ms, mV, Hz, second, implementation, check_units
from brian2.devices.cpp_standalone import CPPStandaloneCodeObject
import numpy as np
import time
import matplotlib.pyplot as plt


def run_network(duration, beta=0.0, w_ee_init=0.1, rate_ext=5, N=5000, save_spikes=True, pre_sim=1,
                monitor_weights=True, eta=2., plastic=False, tau_p=0.3, kappa=5, tau_c=1, c=0.1, gamma=5.5,
                spike_rec_ival=20*60, output_dir=None, w_ei=1, dt_wmon=10):
    """
    :param duration:        length of simulation (in seconds)
    :param beta:            strength of presynaptic inhibition
    :param w_ee_init:       initial weights for E-E connections
    :param rate_ext:        population rate of external Poisson input
    :param N:               total number of neurons
    :param save_spikes:     whether to save spike times (takes memory!)
    :param pre_sim:         simulation length before starting plasticity (in seconds)
    :param monitor_weights: whether to track E-E weights 
    :param eta:             learning rate (scales plasticity time constant)
    :param plastic:         whether E-E weights are plastic
    :param tau_p:           timescale of presynaptic inhibition
    :param kappa:           target rate
    :param tau_c:           timescale of homeostatic control
    :param c:               connection probability
    :param gamma:           E/I balance factor for weights
    :param spike_rec_ival:  interval between recording spikes
    :param output_dir:      directory where to compile the code
    :param w_ei:            scales strength of I->E connection
    :param dt_wmon:         timestep of recording weights (in seconds)

    Returns spike monitor (Brian object), rate monitor (as numpy array) and weight monitor (as Brian object)
    """

    # ----------------------------- #
    #   Set up brian environment    #
    # ----------------------------- #

    b2.start_scope()
    b2.set_device('cpp_standalone', build_on_run=False)  # ensures code is run in c++ only (speed!)
    b2.prefs.codegen.cpp.headers += ['"run.h"']  # This is necessary to use brian_end()
    b2.prefs.core.default_float_dtype = np.float32  # only used 32 bit floats (memory & speed)

    # ------------------------------- #
    #   Default network parameters    #
    # ------------------------------- #

    # paremeters for monitoring the rate
    dt_rate = 10*ms  # monitor rate every 100 ms
    tau_rate = 20*ms

    # network size dependent parameters
    N_e = int(N * (4 / 5))
    N_i = int(N * (1 / 5))
    N_ext = 2*int(N/5)

    # neuron parameters
    taue = 10 * ms
    taui = 10 * ms
    taup = tau_p*1000 * ms
    Vt = -50 * mV
    Vr = -70 * mV
    taum_e = 20
    taum_i = 20

    # synaptic weight parameters
    J = 0.5
    # w_ie = we/np.sqrt(c*N_e)*J
    # w_ei = gamma*we/np.sqrt(c*N_i)*J
    # w_ii = gamma*we/np.sqrt(c*N_i)*J
    w_ie = J * mV
    w_ei = -w_ei*J*gamma * mV
    w_ii = -J*gamma * mV
    w_ext = 2 * mV
    w0 = w_ee_init * mV  # initial weight for EE connections
    w_max = 5*w0  # maximum weight

    # GABA spillover and presynaptic inhibition
    IE_ratefrac = taum_i/taum_e
    A_GABA = IE_ratefrac/(c*N_i)/taui*second  # amount of GABA released for every inhibitory spike (indep. of timescale)
    p_init = np.clip(1 - beta*kappa, 0, 1)
    if plastic:
        w0 /= p_init
    etap = 1
    if beta == 0:  # disable changes in p if presynaptic inhibition inactive, beta = 0 means psi off
        etap = 0

    # plasticity parameters
    tauplus = 16.8 * ms
    tauminus = 33.7 * ms
    tauslow = 114 * ms
    Aplus = 6.5 * 10**(-3)  # amplitude of potentiation events
    eta_w = 0 * mV  # plasticity is off initially
    w_min = 0
    tau_c = tau_c*1000 * ms
    kappa = kappa * Hz
    pref = Aplus*tauplus*tauslow/(tauminus*kappa)  # prefactor of adaptive depression amplitude term

    # ------------------------------- #
    #   Model equations in Brian2     #
    # ------------------------------- #

    # dynamics for excitatory and inhibitory neurons
    eqs = '''
    dv/dt  = ((Vr-v) + g_e + g_i) /taum : volt (unless refractory)
    dg_e/dt = -g_e/taue : volt
    dg_i/dt = -g_i/taui : volt
    dGABA/dt = -GABA/taui: 1
    dp/dt = (-p + clip(1-beta*GABA,0,1))*etap/taup : 1
    taum : second
    dzplus/dt = -zplus /tauplus : 1
    dzminus/dt = -zminus /tauminus : 1 
    dnu/dt = -nu/tau_c : Hz 
    '''

    neurons = b2.NeuronGroup(N, eqs, threshold='v>Vt', refractory=5 * ms, method='rk4',
                             reset='''v=Vr
                                      zplus+=1
                                      zminus+=1
                                      nu+=1/tau_c''')

    exc_neurons = neurons[:N_e]
    inh_neurons = neurons[N_e:]

    # set initial value for single neurons
    neurons.v = 'Vr + rand() * (Vt - Vr)'
    neurons.g_e = 'rand()*w_ie'
    neurons.g_i = 'rand()*w_ii'
    exc_neurons.taum = taum_e * ms
    inh_neurons.taum = taum_i * ms
    neurons.p = p_init
    neurons.nu = kappa  # set target rate

    # synapse models
    syn_ee = b2.Synapses(exc_neurons, exc_neurons,  # E-E connections are plastic and subject to presynaptic inhibition
                         '''w_ee : volt
                            dzslow/dt = -zslow/tauslow : 1 (event-driven)''',
                         on_pre='''g_e += p*w_ee
                                   w_ee -= eta_w*pref*(nu_post**2)*zminus_post
                                   w_ee = clip(w_ee,w_min,w_max)''',  # depression
                         on_post='''w_ee += eta_w*Aplus*zplus_pre*zslow
                                    w_ee = clip(w_ee,w_min,w_max)
                                    zslow += 1''', method='euler')  # potentiation
    syn_ie = b2.Synapses(exc_neurons, inh_neurons, on_pre='g_e += w_ie')
    syn_ei = b2.Synapses(inh_neurons, exc_neurons, on_pre='''g_i += w_ei
                                                             GABA += A_GABA''')  # GABA spillover
    syn_ii = b2.Synapses(inh_neurons, inh_neurons, on_pre='g_i += w_ii')

    # connection lists for EE, EI, IE and II
    connections = {}
    pre_idx, post_idx = create_synapses(N_e, N_e, c, autapse=False)
    connections['ee'] = np.array([pre_idx, post_idx])
    pre_idx, post_idx = create_synapses(N_e, N_i, c)
    connections['ie'] = np.array([pre_idx, post_idx])
    pre_idx, post_idx = create_synapses(N_i, N_e, c)
    connections['ei'] = np.array([pre_idx, post_idx])
    pre_idx, post_idx = create_synapses(N_i, N_i, c, autapse=False)
    connections['ii'] = np.array([pre_idx, post_idx])

    # connect populations using the connection lists
    syn_ee.connect(i=connections['ee'][0], j=connections['ee'][1])
    syn_ee.w_ee = w0
    syn_ie.connect(i=connections['ie'][0], j=connections['ie'][1])
    syn_ei.connect(i=connections['ei'][0], j=connections['ei'][1])
    syn_ii.connect(i=connections['ii'][0], j=connections['ii'][1])

    # external input
    neurons_ext = b2.PoissonGroup(N_ext, rate_ext * Hz)
    syn_ext = b2.Synapses(neurons_ext, neurons, on_pre='g_e += w_ext')
    pre_idx, post_idx = create_synapses(N_ext, N, c)
    syn_ext.connect(i=pre_idx, j=post_idx)

    # ---------------------------------------------- #
    #   Code to allows interruptions of simulation   #
    # ---------------------------------------------- #

    # - stops run if network activity is pathologically high
    # - implementation of stopping function needs to be in C++

    @implementation(CPPStandaloneCodeObject, '''
    double stop_if_too_high(double rate, double t, double add) {
        if ((rate > 190) && (t>add+0.5)) {
            brian_end();  // save all data to disk
            std::exit(0);
        }
        return 0.0;
    }
    ''')
    @implementation('numpy', discard_units=True)
    @check_units(rate=Hz, t=second, add=1, result=1)
    def stop_if_too_high(rate, t, add):
        if rate > 190 * Hz and t > (add+0.5)*second:
            b2.stop()

    # Instantaneous rate is tracked by external population connected to first 1000 neurons
    instant_rate = b2.NeuronGroup(1, '''rate : Hz
                                        drate_mon/dt = -rate_mon/tau_rate : Hz''',
                                  threshold='True', reset='rate=0*Hz', method='exact')
    con = b2.Synapses(exc_neurons[:1000], instant_rate, on_pre='''rate += 1.0/N_incoming/dt
                                                                  rate_mon += 1.0/N_incoming/tau_rate''')
    con.connect()
    instant_rate.run_regularly('dummy = stop_if_too_high(rate, t, pre_sim)', when='after_synapses')

    # ------------------- #
    #   Set up monitors   #
    # ------------------- #

    rate_mon = b2.StateMonitor(instant_rate, 'rate_mon', dt=dt_rate, record=True)
    spike_mon = 0  # in case simulation is interrupted, spike_mon needs to exist
    if monitor_weights:
        wmon = b2.StateMonitor(syn_ee, 'w_ee', record=np.arange(500), dt=dt_wmon*1000*ms)  # (monitor first 500 weights)
    else:
        wmon = 0

    # ------------------- #
    #   Start simulation  #
    # ------------------- #

    # run for some time before plasticity is turned on
    b2.run(pre_sim * second)

    # switch plasticity on
    if plastic:
        eta_w = eta * mV

    # if spikes should be monitored - do that in the beginning and end of the simulation only (memory!)
    if save_spikes:
        spike_record_interval = np.minimum(spike_rec_ival, duration)
        b2.run((duration - spike_rec_ival) * second)
        spike_mon = b2.SpikeMonitor(exc_neurons[:1000])
        b2.run(5*second)
        spike_mon.active = False
        b2.run(spike_rec_ival * second)
        spike_mon.active = True
        b2.run(5*second)
    else:
        spike_mon = 0
        b2.run(duration * second)

    # this compiles the code and runs it
    b2.device.build(directory=output_dir, compile=True, run=True, debug=False, clean=True)

    # postprocessing of raw rate data to numpy array
    rate_e = np.vstack((np.array(rate_mon.t), np.array(rate_mon.rate_mon))).T
    rate_i = None  # inh rate not monitored

    return spike_mon, rate_e, wmon


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

    now = time.time()
    print('Running network...')
    spike_mon, rate_e, wmon = run_network(5, pre_sim=0, w_ee_init=1, beta=0.1, plastic=False, tau_c=5,
                                          monitor_weights=True, save_spikes=True, spike_rec_ival=0, dt_wmon=1)
    print(f"Simulation took {time.time()-now: 1.1f} s")

    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(5, 4), dpi=200)
    ax[0].plot(rate_e[:, 0], rate_e[:, 1], 'C3')
    ax[0].set(ylabel='exc pop rate (1/s)')
    if spike_mon:
        ax[1].scatter(spike_mon.t, spike_mon.i, c='k', s=0.5)
        ax[1].set(ylabel='Neuron idx', xlim=[12.5, 15])
    ax[2].plot(wmon.t, wmon.w_ee.T*1000, lw=1)
    ax[2].set(xlabel='time (s)', ylabel='E-E weights (mV)', ylim=[0, 3])
    plt.tight_layout()
    plt.show()



