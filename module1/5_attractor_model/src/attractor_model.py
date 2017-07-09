import numpy as np
import matplotlib.pyplot as plt

# Let's first define parameters for our model, using dictionaries:

def default_pars():
    '''
    Return: dictionary of default parameters for Wong & Wang (2006) reduced model.
    '''
    
    z = {}
    ### Stimulus Parameters ###
    z['Jext'] = 5.2e-4 # Stimulus input strength [nA]
    
    # Working memory (WM) stimulus parameters
    z['mu1'] = 20. # Strength of stimulus 1 [dimensionless]
    z['mu2'] = 0. # Strength of stimulus 2 [dimensionless]
    
    # Decision making (DM) stimulus parameters
    z['coh'] = 100. # Stimulus coherence [%]
    z['mu0'] = 20. # Stimulus firing rate [spikes/sec]
    
    ### Network Parameters ###
    z['JE'] = 0.2609 # self-coupling strength [nA]
    z['JI'] = -0.0497 # cross-coupling strength [nA]
    z['I0'] = 0.3255 # background current [nA]
    z['tauS'] = 0.1 # Synaptic time constant [sec]
    z['gamma'] = 0.641 # Saturation factor for gating variable
    z['tau0'] = 0.002 # Noise time constant [sec]
    z['sigma'] = 0.02 # Noise magnitude [nA]
    return z

def default_expt_pars():
    '''
    Return: dictionary of parameters related to experimental simulation.
    '''
    
    z = {}
    z['Ntrials'] = 5 # Total number of trials
    z['Ttotal'] = 5. # Total duration of simulation [sec]
    z['Tstim'] = 1. # Time of stimulus 1 onset [sec]
    z['Tdur'] = 2. # Duration of stimulus 1 [sec]
    z['dt'] = 0.0005 # Simulation time step [sec]
    z['dt_smooth'] = 0.02 # Temporal window size for smoothing [sec]
    z['S1_init'] = 0.1 # initial condition for dimension-less gating variable S1
    z['S2_init'] = 0.1 # initial condition for dimension-less gating variable S2
    return z

# Let's define the transfer function for our cells, the firing rate as a function of input current, known as the F-I curve:

def F(I, a=270., b=108., d=0.154):
    '''
    Transfer function: Firing rate as a function of input.
    
    Parameters:
    I : Input current
    a, b, d : parameters of F(I) curve
    
    Return: F(I) for vector I
    '''
    
    return (a*I - b)/(1. - np.exp(-d*(a*I - b)))

# Now let's make a function that simulates the model in time, for multiple trials:

def run_sim(pars,expt_pars,expt, verbose=False):
    '''
    Run simulation, for multiple trials.
    
    Parameters:
    pars : circuit model parameters
    expt_pars : other parameters 
    expt: Experimental paradigm: 'WM' (for working memory ) or 'DM' (for decision making)
    
    Return: dictionary with activity traces
    '''
    
    # Make lists to store firing rate (r) and gating variable (s)
    S1_traj = [];  
    S2_traj = [];
    r1_traj = [];  
    r2_traj = [];
    r1smooth_traj = [];  
    r2smooth_traj = [];
    
    for i in xrange(expt_pars['Ntrials']): #Loop through trials

        if verbose and (i % 10 == 0):
            print "trial # ", i+1, 'of', expt_pars['Ntrials']

        #Set random seed
        np.random.seed(i)

        #Initialize
        r1smooth = []
        r2smooth = []
        
        NT = int(expt_pars['Ttotal']/expt_pars['dt'])
        
        Ieta1 = np.zeros(NT+1)
        Ieta2 = np.zeros(NT+1)
        S1 = np.zeros(NT+1)
        S2 = np.zeros(NT+1)
        r1 = np.zeros(NT)
        r2 = np.zeros(NT)
        
        # Initialize S1, S2
        S1[0] = expt_pars['S1_init']
        S2[0] = expt_pars['S2_init']
        
        for t in xrange(0,NT): #Loop through time for a trial

            #---- Stimulus------------------------------------------------------

            if expt == 'WM':
                Istim1 = \
                    ((expt_pars['Tstim']/expt_pars['dt'] < t) & \
                     (t<(expt_pars['Tstim']+expt_pars['Tdur'])/expt_pars['dt'])
                    ) * (pars['Jext']*pars['mu1']) # To population 1
                Istim2 = \
                    ((expt_pars['Tstim']/expt_pars['dt'] < t) & \
                     (t<(expt_pars['Tstim']+expt_pars['Tdur'])/expt_pars['dt'])
                    ) * (pars['Jext']*pars['mu2']) # To population 1
                
            elif expt == 'DM':
                Istim1 = ((expt_pars['Tstim']/expt_pars['dt']<t) & \
                          (t<(expt_pars['Tstim']+expt_pars['Tdur'])/expt_pars['dt'])
                         ) * (pars['Jext']*pars['mu0']*(1+pars['coh']/100.)) # To population 1
                Istim2 = ((expt_pars['Tstim']/expt_pars['dt']<t) & \
                          (t<(expt_pars['Tstim']+expt_pars['Tdur'])/expt_pars['dt'])
                         ) * (pars['Jext']*pars['mu0']*(1-pars['coh']/100.)) # To population 2
            
            # Total synaptic input
            
            Isyn1 = pars['JE']*S1[t] + pars['JI']*S2[t] + Istim1 + Ieta1[t]
            Isyn2 = pars['JI']*S1[t] + pars['JE']*S2[t] + Istim2 + Ieta2[t]
            #Isyn1, Isyn2 = currents_WM(S1[t], S2[t], pars)
            
            # Transfer function to get firing rate
            
            r1[t]  = F(Isyn1)
            r2[t]  = F(Isyn2)
        
            #---- Dynamical equations -------------------------------------------

            # Mean NMDA-mediated synaptic dynamics updating
            S1[t+1] = S1[t] + expt_pars['dt']*(-S1[t]/pars['tauS'] + (1-S1[t])*pars['gamma']*r1[t]);
            S2[t+1] = S2[t] + expt_pars['dt']*(-S2[t]/pars['tauS'] + (1-S2[t])*pars['gamma']*r2[t]);

            # Ornstein-Uhlenbeck generation of noise in pop1 and 2
            Ieta1[t+1] = Ieta1[t] + \
                            (expt_pars['dt']/pars['tau0']) * (pars['I0']-Ieta1[t]) + \
                            np.sqrt(expt_pars['dt']/pars['tau0'])*pars['sigma']*np.random.randn()
            Ieta2[t+1] = Ieta2[t] + \
                            (expt_pars['dt']/pars['tau0'])*(pars['I0']-Ieta2[t]) + \
                            np.sqrt(expt_pars['dt']/pars['tau0'])*pars['sigma']*np.random.randn()
        
        smooth_wind = int(expt_pars['dt_smooth']/expt_pars['dt'])
        
        r1smooth = np.array([np.mean(r1[j:j+smooth_wind]) for j in xrange(NT)])
        r2smooth = np.array([np.mean(r2[j:j+smooth_wind]) for j in xrange(NT)])
        
        S1_traj.append(S1)
        S2_traj.append(S2)
        r1_traj.append(r1)
        r2_traj.append(r2)
        r1smooth_traj.append(r1smooth)
        r2smooth_traj.append(r2smooth)
        
    tvec = expt_pars['dt']*np.arange(NT)
    
    z = {'S1':S1_traj, 'S2':S2_traj, # NMDA gating variables
         'r1':r1_traj, 'r2':r2_traj, # Firing rates
         'r1smooth':r1smooth_traj, 'r2smooth':r2smooth_traj, # smoothed firing rates
         't':tvec}

    return z

# Now let's build some functions that let us analyze the model using tools of dynamical systems theory: phase plane, nullclines fixed points, and flow fields. 

def currents_WM(S1,S2,pars):
    '''
    Input currents for working memory task.
    '''
    
    I1 = (pars['JE']*S1 + pars['JI']*S2) + pars['I0'] + pars['mu1']*pars['Jext']
    I2 = (pars['JI']*S1 + pars['JE']*S2) + pars['I0'] + pars['mu2']*pars['Jext']
    return I1, I2

def currents_DM(S1,S2,pars):
    '''
    Input currents for decision making task.
    '''
    
    I1 = (pars['JE']*S1 + pars['JI']*S2) + pars['I0'] + pars['mu0']*pars['Jext']*(1 + pars['coh']/100.)
    I2 = (pars['JI']*S1 + pars['JE']*S2) + pars['I0'] + pars['mu0']*pars['Jext']*(1 - pars['coh']/100.)
    return I1, I2

def Sderivs(S1,S2,I1,I2,pars):
    '''
    Time derivatives for S variables (dS/dt).
    '''
    
    dS1dt = -S1/pars['tauS'] + pars['gamma']*(1.0-S1)*F(I1)
    dS2dt = -S2/pars['tauS'] + pars['gamma']*(1.0-S2)*F(I2)
    return dS1dt, dS2dt
    
def plot_nullcline(ax,x,y,z,color='k',label=''):
    '''
    Nullclines.
    '''
    nc = ax.contour(x,y,z,levels=[0],colors=color) # S1 nullcline
    nc.collections[0].set_label(label)
    return nc

def plot_flow_field(ax,x,y,dxdt,dydt,n_skip=1,scale=None,facecolor='gray'):
    '''
    Vector flow fields.
    '''
    v = ax.quiver(x[::n_skip,::n_skip], y[::n_skip,::n_skip], 
              dxdt[::n_skip,::n_skip], dydt[::n_skip,::n_skip], 
              angles='xy', scale_units='xy', scale=scale,facecolor=facecolor)
    return v 

def plot_phase_plane(pars, expt, ax=None):
    '''
    Phase plane plot with nullclines and flow fields.
    '''
    
    if ax is None:
        ax = plt.gca()
    
    # Make 2D grid of (S1,S2) values
    S_vec = np.linspace(0.001,0.999,200) # things break down at S=0 or S=1
    S1,S2 = np.meshgrid(S_vec,S_vec)

    if expt == 'WM':
        I1, I2 = currents_WM(S1,S2,pars)
    elif expt == 'DM':
        I1, I2 = currents_DM(S1,S2,pars)
    else:
        print "Must define expt as 'WM' or 'DM'"
        return 0
    
    dS1dt, dS2dt = Sderivs(S1, S2, I1, I2, pars)
    
    plot_nullcline(ax, S2, S1, dS1dt, color='orange', label='S1 nullcline') # S1 nullcline
    plot_nullcline(ax, S2, S1, dS2dt, color='green', label='S2 nullcline') # S2 nullcline
    plt.legend(loc=1)

    plot_flow_field(ax,S2,S1,dS2dt,dS1dt,n_skip=12,scale=40)

    ax.set_xlabel('$S_2$')
    ax.set_ylabel('$S_1$')
    ax.set_xlim(0,0.8)
    ax.set_ylim(0,0.8)
    ax.set_aspect('equal')
