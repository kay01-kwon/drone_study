import numpy as np
def do_step(f, s, u, tspan):
    '''
    Custom RK4
    :param f: function of dynamics
    :param s: state
    :param u: input
    :param tspan: t0 and t1
    :return: s_next
    '''
    dt = tspan[1] - tspan[0]
    tm = tspan[0] + dt/2.0
    tf = tspan[1]
    K1 = f(tspan[0], s, u)
    K2 = f(tm, s + K1*dt/2, u)
    K3 = f(tm, s + K2*dt/2, u)
    K4 = f(tf, s + K3*dt, u)
    s_next = s + 1/6.0*(K1 + 2.0*K2 + 2.0*K3 + K4)*dt
    return s_next