import numpy as np
from numpy.linalg import *
import torch
##########################
# Loss func: f(k,c)**2
##########################
def f(k, c):
    out = c[0] + c[1]*k[0]**2 + c[2]*k[1]**2 + c[3]*(k[2] - 0.5)**2
    return out
def grad_f(k):
    return np.array([1, k[0]**2, k[1]**2, (k[2] - 0.5)**2])

if __name__ == '__main__':
    np.random.seed(10)
    c = np.random.rand(4)
    # c = np.array([-0.00166253, 0.32985641, 0.63364823, 0.12121279])
    K = np.loadtxt('kout2.dat')
    dt = 0.5
    for i in range(20000):
        loss = 0
        for j in range(len(K)):
            dc = -2*f(K[j], c)*grad_f(K[j])
            ############### line search #################
            t = 1
            while norm(f(K[j], c + t*dc)) >= norm(f(K[j], c)):
                t = t * dt
                if t < 1e-4:
                    break
            ############### line search #################
            c += t*dc
            loss += norm(f(K[j], c))
        if i % 100 == 0:
            print('Step: {}'.format(i + 1))
            print('Parameter: {}; Loss: {}'.format(c, loss))

######################################## FINAL ##################################################
# Parameter: [-0.00166253  0.32985641  0.63364823  0.12121279]; Loss: 6.632136090763265e-07
######################################## FINAL ##################################################
