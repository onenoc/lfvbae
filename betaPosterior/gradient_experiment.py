from betaPosterior import gradient_lower_bound
from vbil import grad_KL
from vbil import sample_theta as generate_samples
from scipy.stats import beta, gamma, gaussian_kde
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
params = [  16.42568426,  151.18846825]
#params = [50.,80.]
num_samples = 10
num_particles = 10
gradAVABC10 = []
gradBBVI10 = []
for i in range(1000):
    params = np.zeros(2)
    params[0] = np.random.uniform(10,20)
    params[1] = np.random.uniform(125,175)
    gradBBVI10.append(-grad_KL(params,num_samples,num_particles)[0])
    gradAVABC10.append(gradient_lower_bound(params,num_samples,num_particles)[0])
AVABC_gradients_10 = np.asarray(gradAVABC10)
BBVI_gradients_10 = np.asarray(gradBBVI10)

num_samples = 1
num_particles = 1
gradAVABC1 = []
gradBBVI1 = []
for i in range(100):
    params = np.zeros(2)
    params[0] = np.random.uniform(10,20)
    params[1] = np.random.uniform(125,175)
    gradBBVI1.append(-grad_KL(params,num_samples,num_particles)[0])
    gradAVABC1.append(gradient_lower_bound(params,num_samples,num_particles)[0])
AVABC_gradients_1 = np.asarray(gradAVABC1)
BBVI_gradients_1 = np.asarray(gradBBVI1)

density_AVABC_10 = gaussian_kde(AVABC_gradients_10)
density_BBVI_10 = gaussian_kde(BBVI_gradients_10)
density_AVABC_1 = gaussian_kde(AVABC_gradients_1)
density_BBVI_1 = gaussian_kde(BBVI_gradients_1)

print density_AVABC_1(1000)

xs = np.linspace(-60000,200,200)
plt.plot(xs,density_AVABC_10(xs),label='AVABC 10 samples')
plt.plot(xs,density_AVABC_1(xs),label='AVABC 1 sample')
plt.plot(xs,density_BBVI_10(xs),label='BBVI 10 samples')
plt.plot(xs,density_BBVI_1(xs),label='BBVI 1 sample')
plt.legend(loc=1)
plt.title(r'$\nabla_{\mu}\mathcal{L}$, AVABC and BBVI, convergence parameters,$n=100,k=70$')
plt.show()
#
plt.hist(BBVI_gradients_1,label='BBVI')
plt.hist(AVABC_gradients_1,label='AVABC')
plt.show()
