#
# Bayesian Analysis with Python written by Osvaldo Marthin
# Chapter 2
##################################################################

import pymc3 as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')

# In[0]:
def main():
    # np.random.seed(123)
    # n_experiments = 4
    theta_real = 0.35
    # data = stats.bernoulli.rvs(p=theta_real, size=n_experiments)
    data = [1, 0, 0, 0]
    print(data)
    
    with pm.Model() as our_first_model:
      θ = pm.Beta('θ', alpha=1, beta=1)
      y = pm.Bernoulli('y', p=θ, observed=data)
    
      start = pm.find_MAP()
      step = pm.Metropolis()
      trace = pm.sample(1000, step=step, start=start)
    
    burnin = 100
    chain = trace[burnin:]
    print('chain: ', chain)
    
    pm.traceplot(chain, lines={'theta':theta_real});
    plt.savefig('img204.png', dpi=300, figsize=(5.5, 5.5))
    
    plt.figure()
    
# In[0]:
    with our_first_model:
      step = pm.Metropolis()
      multi_trace = pm.sample(1000, step=step, cores=4)
    
    burnin = 100
    multi_chain = multi_trace[burnin:]
    pm.traceplot(multi_chain, lines={'theta':theta_real});
    plt.savefig('img206.png', dpi=300, figsize=(5.5, 5.5))

    plt.figure()

# In[0]:
    pm.gelman_rubin(multi_chain) 
    {'theta': 1.0074579751170656, 'theta_logodds': 1.009770031607315}
    
    pm.forestplot(multi_chain) #, varnames={'theta'});
    plt.savefig('img207.png', dpi=300, figsize=(5.5, 5.5))
    
    plt.figure()
    
# In[0]:
    #pm.df_summary(multi_chain)
    pm.summary(multi_chain)
    
    pm.autocorrplot(chain)
    plt.savefig('img208.png', dpi=300, figsize=(5.5, 5.5))
    
    plt.figure()
    
# In[0]:
    pm.effective_n(multi_chain)['θ']
    pm.plot_posterior(chain)
    # pm.plot_posterior(chain, kde_plot=True)
    plt.savefig('img209.png', dpi=300, figsize=(5.5, 5.5))
    
    plt.figure()
    
# In[0]:
    pm.plot_posterior(chain, rope=[0.45,.55])
    # pm.plot_posterior(chain, kde_plot=True, rope=[0.45,.55])
    plt.savefig('img210.png', dpi=300, figsize=(5.5, 5.5))
    
    plt.figure()
    
# In[0]:
    pm.plot_posterior(chain, ref_val=0.5)
    # pm.plot_posterior(chain, kde_plot=True, ref_val=0.5)
    plt.savefig('img211.png', dpi=300, figsize=(5.5, 5.5))
    
    plt.figure()

# In[0]:
if __name__ == '__main__':
    main()