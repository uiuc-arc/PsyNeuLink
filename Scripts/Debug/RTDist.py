#import cupy
#rtdist = cupy.RawModule(path='D:/Dropbox/princeton/PsyNeuLink_Stuff/RTDist_1.0.0-rc2/src/CUDA/RTDist.ptx')

import numpy as np
import numpy.matlib
import matlab.engine
import time

eng = matlab.engine.start_matlab()
eng.addpath('D:\\Dropbox\\princeton\\PsyNeuLink_Stuff\\RTDist_1.0.0-rc2\\MATLAB')

#%%
settings = {
    'nGPUS': 1,
    'nWalkers': 5000,
}

nDim = 4
lcaV = np.array([[0,   0.1, 0.2, 0.3]])
lcaV = numpy.matlib.repmat(lcaV, 500, 1)
nStimuli = lcaV.shape[0]
threshold = np.full((nDim,1), 0.08)
leak = 0.2
competition = 0.3
gamma = np.full((nDim,nDim), competition)
np.fill_diagonal(gamma, leak)

LCAPars = {
    'nDim': 4,
    'nStimuli': nStimuli,
    'a': matlab.double(threshold.tolist()),
    'v': matlab.double(np.transpose(lcaV).tolist()),
    'Ter': 0.3,
    'Gamma': matlab.double(gamma.tolist())
}

t0 = time.time()
D,ok = eng.LCADist(LCAPars,settings,1, nargout=2)
D = np.asarray(D)
t1 = time.time()
print(f'LCA Execution Time: {t1-t0}')

#%%
import matplotlib.pyplot as plt
plt.ion()

#%%
x = np.linspace(0, 3000*.001, 3000)
plt.close('all')
# f, axes= plt.subplots(nStimuli, 1, sharey=True)
# if nStimuli == 1:
#     axes_list = [axes]
# else:
#     axes_list = axes.flat
#
# k = 0
# for ax in axes_list:
#     ax.plot(x, D[:,k:(k+nDim)])
#     k = k + 2

D_by_stim = np.hsplit(D, nStimuli)
D_all = sum(D_by_stim)
f2, ax2 = plt.subplots()
p = ax2.plot(x, D_all)