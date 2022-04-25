from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation
from collections import deque

g = 9.81  # acceleration due to gravity, in m/s^2
L = 0.2  # length of pendulum in m
# M1= 1.0  # mass of pendulum in kg
t_stop = 15  # how many seconds to simulate
history_len = 500  # how many trajectory points to display
A = 0.1 # amplitude of oscilation, in m
a = A / L
w_crit = np.sqrt(g / L) # value of w to keep pendulum upright
w = 45 # frequeny of oscillation

def derivs(state, t): # given positions and velocities return accelerations 
    
    dydx = np.zeros_like(state)
    dydx[0] = state[1]
    # dydx[1] = -state[0] * (a * (w ** 2) * cos(w * t) - (w_crit ** 2))
    dydx[1] = -np.sin(state[0]) * ((A * (w ** 2) * np.cos(w * t)) - g) / L
    #dydx[2] = -A * w_crit * w_crit * cos(w_crit * t)
    return dydx

# create a time array from 0..t_stop sampled at 0.02 second steps
dt = 0.02
t = np.arange(0, t_stop, dt)

# th is the initial angles(degrees)
# w1 is the initial angular velocity (degrees per second)
th = 0.1
w1 = 0.0

# initial state
state = np.radians([th, w1])

# integrate your ODE using scipy.integrate.
y = integrate.odeint(derivs, state, t)

x = L*sin(y[:, 0])
y = L*cos(y[:, 0]) 
# + y0 
#+ y[:, 2]

fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(autoscale_on=False, xlim=(-0.15, 0.15), ylim=(-0.1, 0.3))
ax.set_aspect('equal')
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
trace, = ax.plot([], [], '.-', lw=1, ms=2)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
history_x, history_y = deque(maxlen=history_len), deque(maxlen=history_len)


def animate(i):
    thisx = [0, x[i]]
    thisy = [0, y[i]]

    if i == 0:
        history_x.clear()
        history_y.clear()

    history_x.appendleft(thisx[1])
    history_y.appendleft(thisy[1])

    line.set_data(thisx, thisy)
    trace.set_data(history_x, history_y)
    time_text.set_text(time_template % (i*dt))
    return line, trace, time_text


ani = animation.FuncAnimation(
    fig, animate, len(y), interval=dt*1000, blit=True)

ani.save('inverted_pendulum.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()
