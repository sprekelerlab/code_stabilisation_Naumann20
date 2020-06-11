STUFF = "Hi"  # this is an old hack to help Cython compile

import numpy as np
cimport numpy as np
cimport cython
import copy
DTYPE = np.double
FTYPE = np.float
ITYPE = np.int
ctypedef np.double_t DTYPE_t
ctypedef np.float_t FTYPE_t
ctypedef np.int_t ITYPE_t

from libc cimport math
from cpython cimport bool
from libc.math cimport log, sqrt, exp
from libc.stdlib cimport rand, RAND_MAX


# implementation of a c-based function that provides Gaussian random numbers
cdef double random_gaussian():
    cdef double x1, x2, w

    w = 2.0
    while (w >= 1.0):
        x1 = 2.0 * rand()/(RAND_MAX*1.) - 1.0
        x2 = 2.0 * rand()/(RAND_MAX*1.) - 1.0
        w = x1 * x1 + x2 * x2

    w = ((-2.0 * log(w)) / w) ** 0.5
    return x1 * w


@cython.boundscheck(False)
@cython.cdivision(True)
def c_run_ratenet(params):

    #---------------------------------------------------#
    #  load parameters and initialise arrays/variables  #
    #---------------------------------------------------#

    # load connections from parameter dictionary
    cdef np.ndarray[ITYPE_t, ndim=2] con_ee = params['con_ee']
    cdef np.ndarray[ITYPE_t, ndim=2] con_ei = params['con_ei']
    cdef np.ndarray[ITYPE_t, ndim=2] con_ie = params['con_ie']
    cdef np.ndarray[ITYPE_t, ndim=2] con_ii = params['con_ii']

    # determine number of connections between all groups
    cdef int Ncon_ee = len(con_ee[0])
    cdef int Ncon_ei = len(con_ei[0])
    cdef int Ncon_ie = len(con_ie[0])
    cdef int Ncon_ii = len(con_ii[0])

    ## Network parameters
    cdef int N_e = params['N_e']
    cdef int N_i = params['N_i']
    cdef float c = params['c']

    ## Simulation parameters
    cdef float duration = params['duration'] # seconds
    cdef float dt = params['dt'] # 1 ms
    cdef float pre_sim = params['pre_sim'] # 1 ms
    cdef float dt_rec = params['dt_rec']
    # these are parameters for time integration and recording of variables
    cdef int nsteps = int(round(duration/dt))
    cdef int skip_rec = int(round(dt_rec/dt))
    cdef int n_rec = int(round(duration/dt_rec))

    # plasticity parameters
    cdef float kappa = params['kappa']
    cdef float eta = params['eta']
    cdef bool plastic = bool(params['plastic'])

    # presynaptic inhibition
    cdef float beta = params['beta']*c # normalise beta with connectivity to account for global effect
    cdef float shift = params['shift']
    cdef bool pre_inh
    cdef int g_func = params['g_func']
    cdef float p_init
    if beta==0:  # if beta=0, presynaptic inhibition is disabled
        pre_inh = False
        p_init = 1
    else:
        pre_inh = True
        wi = params['w_ie']*params['w_ei']/(1+params['w_ii'])
        if g_func == 0: # linear transfer function
            p_init = 1-params['beta']*wi*kappa
        elif g_func == 1: # sigmoid
            p_init = 1/(1+exp(params['beta']*(wi*kappa-shift)))
        else: # exponential
            p_init = exp(-params['beta']*wi*kappa)

    # load weight parameters and initalise excitatory weight array (flat)
    # for E-E weights: if plasticity is on, weights are initalised at the fixed point
    cdef float J = params['J']
    cdef float w_ee_init = params['w_ee_init']*J/(c*N_e)
    cdef float w_ee_FP = (1 + params['w_ei']*params['w_ie']/(1+params['w_ii']) - params['stim_strength']/kappa)/p_init*J/(c*N_e)
    cdef float ween = params['w_ee_init_noise']
    cdef np.ndarray[FTYPE_t] W_ee
    if plastic: # initialise system at FP
        W_ee = np.maximum((np.ones(Ncon_ee, dtype=FTYPE)+ween*np.random.normal(0,1,size=Ncon_ee)),0)*w_ee_FP
    else: # initial system with predefine initial value
        W_ee = np.maximum((np.ones(Ncon_ee, dtype=FTYPE)+ween*np.random.normal(0,1,size=Ncon_ee)),0)*w_ee_init
    cdef float w_ei = params['w_ei']*J/(c*N_i)
    cdef float w_ie = params['w_ie']*J/(c*N_e)
    cdef float w_ii = params['w_ii']*J/(c*N_i)

    ## Timescales
    cdef float tau_c = params['tau_c'] #sec
    cdef float tau_w = params['tau_w'] #sec
    cdef float tau_e = 0.02 #20 ms
    cdef float tau_i = 0.01 #10 ms
    cdef float tau_p = params['tau_p']

    # plasticity parameters for simulation
    cdef bool plasticity_on = False # plasticity is always off initially
    cdef float w_max = 10*w_ee_FP # limit max weight to 10 x initial value / fixed point value
    cdef float pref = eta*w_ee_init/kappa**3 # prefactor of learning rule

    cdef np.ndarray[FTYPE_t] re = np.ones(N_e, dtype=FTYPE)*params['re_init']
    cdef np.ndarray[FTYPE_t] ri = np.ones(N_i, dtype=FTYPE)*params['ri_init']
    cdef float p = p_init
    cdef np.ndarray[FTYPE_t] rd = np.ones(N_e, dtype=FTYPE)*params['re_init']
    cdef float r_max = 200 # maximum firing rate

    # fixed input (in an array to allow for time-varying input)
    cdef float stim_strength = params['stim_strength']
    cdef np.ndarray[FTYPE_t] I_ext = np.ones(N_e, dtype=FTYPE)
    cdef float A_noise = params['noise_strength']

    # helper arrays for input summation
    cdef np.ndarray[FTYPE_t] inp_ee = np.zeros(N_e, dtype=FTYPE)
    cdef np.ndarray[FTYPE_t] inp_ei = np.zeros(N_e, dtype=FTYPE)
    cdef np.ndarray[FTYPE_t] inp_ie = np.zeros(N_e, dtype=FTYPE)
    cdef np.ndarray[FTYPE_t] inp_ii = np.zeros(N_e, dtype=FTYPE)

    # arrays for tracking 
    cdef np.ndarray[FTYPE_t, ndim=2] re_track = np.zeros((n_rec, N_e), dtype=FTYPE) # exc rates
    cdef np.ndarray[FTYPE_t, ndim=2] ri_track = np.zeros((n_rec, N_i), dtype=FTYPE) # inh rates
    # choose 500 random connections to record from during the simulation (saving memory)
    cdef np.ndarray[ITYPE_t] W_track_id = np.random.choice(np.arange(Ncon_ee), np.min((500, Ncon_ee)), replace=False) 
    cdef np.ndarray[FTYPE_t, ndim=2] W_track = np.zeros((n_rec, len(W_track_id)), dtype=FTYPE) # random set of weights
    cdef np.ndarray[FTYPE_t] p_track = np.zeros(n_rec, dtype=FTYPE) # release probability
    cdef np.ndarray[FTYPE_t] time = np.zeros(n_rec, dtype=FTYPE) # time array
    cdef np.ndarray[FTYPE_t, ndim=2] rd_track = np.zeros((n_rec, N_e), dtype=FTYPE) # delayed estimate of exc rate

    # fill in initial values
    W_track[0,:] = W_ee[W_track_id]
    p_track[0] = p
    rd_track[0, :] = rd
   

    cdef int n, ti, i, j
    cdef int ti_skip = 0
    cdef float noise, sum_ri

    #--------------------#
    #  time integration  #
    #--------------------#

    for ti in range(1, nsteps):

        # to save compute time and memory we only look at pre-post pairs with an existing connection

        # gather inputs
        for ci in range(Ncon_ee): # E to E
            j = con_ee[0,ci]
            i = con_ee[1,ci]
            inp_ee[i] += p*W_ee[ci]*re[j] # presynaptic inhibition (global)
            # plasticity
            if plasticity_on:
                W_ee[ci] = min(max(W_ee[ci] + pref*re[i]*re[j]*(re[i]-rd[i]**2/kappa)/tau_w*dt, 0), w_max)

        for ci in range(Ncon_ei): # I to E
            i = con_ei[0,ci]
            j = con_ei[1,ci]
            inp_ei[j] += ri[i]

        for ci in range(Ncon_ie): # E to I
            i = con_ie[0,ci]
            j = con_ie[1,ci]
            inp_ie[j] += re[i]

        for ci in range(Ncon_ii): # I to I
            i = con_ii[0,ci]
            j = con_ii[1,ci]
            inp_ii[j] += ri[i]

        # loop over all inhibitory neurons
        sum_ri = 0
        for n in range(N_i):
            sum_ri += w_ei*ri[n] # gather inh activity for global pre inh.
            ri[n] = min(max(ri[n] + (-ri[n] + w_ie*inp_ie[n] - w_ii*inp_ii[n])/(tau_i)*dt, 0), r_max)

        if pre_inh:
            if g_func == 0: # linear transfer function
                p = min(max(p + (-p + 1-beta*sum_ri)/tau_p*dt, 0), 1)
            elif g_func == 1: # sigmoid
                p = min(max(p + (-p + 1/(1+exp(beta*(sum_ri-shift/c))))/tau_p*dt, 0), 1) #dividing by c necessary bc of norm of beta and w_ei
            else: # exponential
                p = min(max(p + (-p + exp(-beta*sum_ri)),0),1)
        sum_ri = 0 # reset summing of inh input

		# loop over excitatory neurons and integrate
        for n in range(N_e):

            noise = random_gaussian()*A_noise # compute random Gaussian noise at every time point 

            re[n] = min(max(re[n] + (-re[n] + inp_ee[n] - w_ei*inp_ei[n]+ stim_strength*max(I_ext[n],0)+noise)/tau_e*dt, 0), r_max)
            rd[n] = min(max(rd[n] + (-rd[n] + re[n])/tau_c*dt, 0), r_max)

            #reset input vectors to 0
            inp_ee[n] = 0
            inp_ei[n] = 0
            inp_ie[n] = 0
            inp_ii[n] = 0

        # record weights and time only at certain intervals to save memory
        if (ti%skip_rec) == 0:
            ti_skip = int(round(ti/skip_rec))
            W_track[ti_skip,:] = W_ee[W_track_id]
            p_track[ti_skip] = p
            time[ti_skip] = ti*dt

        # rates are also only sampled at certain intervals but interpolated to avoid subsampling
        re_track[ti_skip,:] += re/skip_rec
        ri_track[ti_skip,:] += ri/skip_rec

        # switch plasticity on after an initial period (if system is plastic)
        if plastic and ti*dt>=pre_sim:
            plasticity_on = True

    return re_track, ri_track, W_track, p_track, time
