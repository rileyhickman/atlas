import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"text.usetex": True,
                     "axes.spines.right" : False,
                     "axes.spines.top" : False,
                     "font.size": 15,
                     "savefig.dpi": 400,
                     "savefig.bbox": "tight",
                     'text.latex.preamble': r"\usepackage{amsfonts}"
                    }
                   )

from geomstats.geometry.euclidean import Euclidean
# from geomstats.geometry.functions import SRVF

import fdasrsf as fs
from head.distances import AmplitudePhaseDistance

N_SAMPLES = 100
TARGET = [0,0.5]

lambda_ = np.linspace(-5,5,num=N_SAMPLES)
Rn = Euclidean(N_SAMPLES)
# srsf = SRVF(lambda_)
def gaussian(mu,sig):
    scale = 1/(np.sqrt(2*np.pi)*sig)
    return scale*np.exp(-np.power(lambda_ - mu, 2.) / (2 * np.power(sig, 2.)))

yt = gaussian(*TARGET)

dRn = lambda xi,yi : float(Rn.metric.dist(yi, yt))

# dSRSF = lambda xi,yi : srsf.metric.dist(yi, yt)   

test_x = np.linspace(-5, 5, 101)

fig, axs = plt.subplots(1,2, figsize=(4*2, 4))
fig.subplots_adjust(wspace=0.3)
ground_truth_Rn = [dRn(i, gaussian(i,TARGET[1])) for i in test_x]
ax = axs[0]
ax.plot(test_x, ground_truth_Rn, label='MSE',color='k')
ax.set_ylabel(r'$d_{\mathcal{M}}(x, x_{t})$')

# ground_truth_srvf = [dSRSF(i, gaussian(i,TARGET[1])) for i in test_x]
# ax.plot(test_x, ground_truth_srvf, label='SRSF',color='k', ls='--')
# ax.legend(ncol=2,loc='upper center', bbox_to_anchor=[0.5,1.2])


ax = axs[1]
efda = np.asarray([AmplitudePhaseDistance(gaussian(i,TARGET[1]), yt, lambda_) for i in test_x])
ax.plot(test_x, efda[:,0], label=r"$d_{a}$")
ax.plot(test_x, efda[:,1], label=r"$d_{p}$")
ax.plot(test_x, efda.sum(axis=1), label=r"$d_{a}+d_{p}$")
ax.legend(ncol=3,loc='upper center', bbox_to_anchor=[0.5,1.2])
fig.supxlabel(r'$\mathcal{X}$', y=-0.05)
plt.savefig('../figures/mse_vs_ap.pdf')
plt.close()