import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import tigramite
import tigramite.data_processing as dp
from tigramite.independence_tests import parcorr
from tigramite.pcmci import PCMCI

data = np.random.randn(1000, 3)  
  
dataframe = dp.DataFrame(data) 

print(f"Number of variables: {dataframe.N}")
print(f"Time series length: {dataframe.T}")
print(f"Variable names: {dataframe.var_names}")

cond_ind_test = parcorr.ParCorr()
pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test)  
  
results = pcmci.run_pcmci(tau_max=2, pc_alpha=None)  
  
pcmci.print_significant_links(  
    p_matrix=results['p_matrix'],  
    val_matrix=results['val_matrix'],  
    alpha_level=0.05)