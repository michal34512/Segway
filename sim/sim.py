import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
import scipy.linalg

class PID:
    def __init__(self, Kp, Ki, Kd, windup = 100000000, offset = 0):
        # Initialize PID
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.windup = windup
        self.offset = offset
        
        self.integral = 0
        self.prev_t = 0
        self.prev_e = 0
    def step(self, curr_t, curr_e):
        delta_t = curr_t-self.prev_t
        delta_e = curr_e-self.prev_e

        proportion = self.Kp * curr_e
        self.integral = self.integral + self.Ki*curr_e*delta_t
        if(self.integral > self.windup):
            self.integral = self.windup + self.offset
        elif(self.integral < -self.windup):
            self.integral = -self.windup - self.offset
        if(abs(delta_t) < 0.001):
            derivative = 0
        else:
            derivative = self.Kd*delta_e/delta_t

        u_signal = proportion + self.integral + derivative + self.offset

        self.prev_e = curr_e
        self.prev_t = curr_t
        return u_signal

pid = PID(200, 400, 3, 1000)
# Definicja równań ruchu
def equations(state, t, params):
    theta, dtheta, phi, dphi, x, dx, px, py = state
    IB, IV, R, b, LT, mB, mW, g = params
    
    # CONTROLLER PID
    val = pid.step(t, theta)
    uL = val+4
    uR = val

    # WHEELS
    dpsiR = (dx - (b * dphi)/2)/R
    dpsiL = (dx + (b * dphi)/2)/R

    TL = (uL-dpsiL)*10000
    TR = (uR-dpsiR)*10000
    

    # SEGWAY
    ddtheta = ((2*IW*LT*g*mB*np.sin(theta)+IW*LT**2*dphi**2*mB*np.sin(2*theta)-12*LT**6*R**2*dtheta**6*mB**2*np.cos(theta)*np.sin(theta)-6*LT**3*R**3*TL*dtheta**2*mB*np.cos(theta)-6*LT**3*R**3*TR*dtheta**2*mB*np.cos(theta)+6*LT*R**2*dx**2*g*mB**2*np.sin(theta)-8*LT**2*R**3*TL*dtheta*dx*mB-8*LT**2*R**3*TR*dtheta*dx*mB+LT**4*R**2*dphi**2*dtheta**2*mB**2*np.sin(2*theta)+3*LT**2*R**2*dphi**2*dx**2*mB**2*np.sin(2*theta)+12*LT**2*R**2*dtheta**2*dx**4*mB**2*np.sin(2*theta)-28*LT**4*R**2*dtheta**4*dx**2*mB**2*np.sin(2*theta)+2*LT*R**2*g*mB*mW*np.sin(theta)+4*IW*LT**2*dtheta**2*dx**2*mB*np.sin(2*theta)-8*LT**5*R**2*dtheta**5*dx*mB**2*np.sin(theta)+2*LT**3*R**2*dtheta**2*g*mB**2*np.sin(theta)+LT**2*R**2*dphi**2*mB*mW*np.sin(2*theta)-6*LT*R**3*TL*dx**2*mB*np.cos(theta)-6*LT*R**3*TR*dx**2*mB*np.cos(theta)+8*IW*LT**3*dtheta**3*dx*mB*np.sin(theta)-24*LT**3*R**2*dtheta**3*dx**3*mB**2*np.sin(theta)-24*LT**3*R**2*dtheta**3*dx**3*mB**2*np.cos(2*theta)*np.sin(theta)+4*LT**2*R**2*dtheta**2*dx**2*mB*mW*np.sin(2*theta)-4*LT**2*R**3*TL*dtheta*dx*mB*np.cos(2*theta)-4*LT**2*R**3*TR*dtheta*dx*mB*np.cos(2*theta)+2*LT**4*R**2*dphi**2*dtheta**2*mB**2*np.sin(2*theta)*np.cos(theta)**2-16*LT**4*R**2*dtheta**4*dx**2*mB**2*np.cos(2*theta)*np.sin(2*theta)+8*LT**4*R**2*dtheta**4*dx**2*mB**2*np.sin(2*theta)*np.cos(theta)**2+8*LT**3*R**2*dtheta**3*dx*mB*mW*np.sin(theta)-8*LT**5*R**2*dtheta**5*dx*mB**2*np.cos(2*theta)*np.sin(theta)-24*LT**5*R**2*dtheta**5*dx*mB**2*np.sin(2*theta)*np.cos(theta)-36*LT**2*R**2*dtheta**2*dx**4*mB**2*np.cos(theta)*np.sin(theta)+16*LT**5*R**2*dtheta**5*dx*mB**2*np.cos(theta)**2*np.sin(theta)+4*LT**3*R**2*dtheta**2*g*mB**2*np.cos(theta)**2*np.sin(theta)+6*LT**3*R**2*dphi**2*dtheta*dx*mB**2*np.sin(2*theta)*np.cos(theta)+12*LT**2*R**2*dtheta*dx*g*mB**2*np.cos(theta)*np.sin(theta))/(2*(IB*IW+IB*R**2*mW+6*LT**6*R**2*dtheta**4*mB**2+12*LT**2*R**2*dx**4*mB**2+6*IW*LT**4*dtheta**2*mB+4*IW*LT**2*dx**2*mB+3*IB*R**2*dx**2*mB+6*LT**4*R**2*dtheta**2*dx**2*mB**2+IB*LT**2*R**2*dtheta**2*mB+6*LT**4*R**2*dtheta**2*mB*mW+4*LT**2*R**2*dx**2*mB*mW-6*LT**6*R**2*dtheta**4*mB**2*np.cos(theta)**2+6*LT**2*R**2*dx**4*mB**2*np.cos(2*theta)-18*LT**2*R**2*dx**4*mB**2*np.cos(theta)**2+2*IW*LT**2*dx**2*mB*np.cos(2*theta)-6*LT**4*R**2*dtheta**2*dx**2*mB**2*np.cos(2*theta)+12*LT**4*R**2*dtheta**2*dx**2*mB**2*np.cos(theta)**2+12*IW*LT**3*dtheta*dx*mB*np.cos(theta)+2*IB*LT**2*R**2*dtheta**2*mB*np.cos(theta)**2+24*LT**3*R**2*dtheta*dx**3*mB**2*np.cos(theta)+12*LT**5*R**2*dtheta**3*dx*mB**2*np.cos(theta)+2*LT**2*R**2*dx**2*mB*mW*np.cos(2*theta)-24*LT**3*R**2*dtheta*dx**3*mB**2*np.cos(theta)**3+12*LT**3*R**2*dtheta*dx*mB*mW*np.cos(theta)-12*LT**4*R**2*dtheta**2*dx**2*mB**2*np.cos(2*theta)*np.cos(theta)**2+6*IB*LT*R**2*dtheta*dx*mB*np.cos(theta)-12*LT**5*R**2*dtheta**3*dx*mB**2*np.cos(2*theta)*np.cos(theta))))

    ddphi = (-(2*R**2*(R*TR-R*TL+2*LT**2*b*dphi*dtheta*mB*np.cos(theta)*np.sin(theta)))/(b*(2*IV*R**2+IW*b**2+R**2*b**2*mW+2*LT**2*R**2*mB*np.sin(theta)**2)))

    ddx = ((R**2*(IB*R*TL+IB*R*TR+12*LT**7*dtheta**6*mB**2*np.sin(theta)+24*LT**6*dtheta**5*dx*mB**2*np.sin(2*theta)+24*LT**3*dtheta**2*dx**4*mB**2*np.sin(theta)+28*LT**5*dtheta**4*dx**2*mB**2*np.sin(theta)+2*IB*LT**3*dtheta**4*mB*np.sin(theta)+8*LT**4*dtheta**3*dx**3*mB**2*np.sin(2*theta)+6*LT**4*R*TL*dtheta**2*mB+4*LT**2*R*TL*dx**2*mB+6*LT**4*R*TR*dtheta**2*mB+4*LT**2*R*TR*dx**2*mB+2*LT**2*R*TL*dx**2*mB*np.cos(2*theta)+2*LT**2*R*TR*dx**2*mB*np.cos(2*theta)+48*LT**4*dtheta**3*dx**3*mB**2*np.cos(theta)*np.sin(theta)+4*IB*LT**2*dtheta**3*dx*mB*np.sin(2*theta)-3*LT**5*dphi**2*dtheta**2*mB**2*np.sin(2*theta)*np.cos(theta)-3*LT**3*dphi**2*dx**2*mB**2*np.sin(2*theta)*np.cos(theta)+12*LT**3*dtheta**2*dx**4*mB**2*np.cos(2*theta)*np.sin(theta)-12*LT**3*dtheta**2*dx**4*mB**2*np.sin(2*theta)*np.cos(theta)+4*LT**5*dtheta**4*dx**2*mB**2*np.cos(2*theta)*np.sin(theta)+36*LT**5*dtheta**4*dx**2*mB**2*np.sin(2*theta)*np.cos(theta)-32*LT**5*dtheta**4*dx**2*mB**2*np.cos(theta)**2*np.sin(theta)-2*LT**4*dphi**2*dtheta*dx*mB**2*np.sin(2*theta)-6*LT**4*dtheta**2*g*mB**2*np.cos(theta)*np.sin(theta)-6*LT**2*dx**2*g*mB**2*np.cos(theta)*np.sin(theta)+6*IB*LT*dtheta**2*dx**2*mB*np.sin(theta)+8*LT**4*dtheta**3*dx**3*mB**2*np.cos(2*theta)*np.sin(2*theta)-16*LT**4*dtheta**3*dx**3*mB**2*np.sin(2*theta)*np.cos(theta)**2-4*LT**3*dtheta*dx*g*mB**2*np.sin(theta)-8*LT**3*dtheta*dx*g*mB**2*np.cos(theta)**2*np.sin(theta)+12*LT**3*R*TL*dtheta*dx*mB*np.cos(theta)+12*LT**3*R*TR*dtheta*dx*mB*np.cos(theta)-4*LT**4*dphi**2*dtheta*dx*mB**2*np.sin(2*theta)*np.cos(theta)**2))/(2*(IB*IW+IB*R**2*mW+6*LT**6*R**2*dtheta**4*mB**2+12*LT**2*R**2*dx**4*mB**2+6*IW*LT**4*dtheta**2*mB+4*IW*LT**2*dx**2*mB+3*IB*R**2*dx**2*mB+6*LT**4*R**2*dtheta**2*dx**2*mB**2+IB*LT**2*R**2*dtheta**2*mB+6*LT**4*R**2*dtheta**2*mB*mW+4*LT**2*R**2*dx**2*mB*mW-6*LT**6*R**2*dtheta**4*mB**2*np.cos(theta)**2+6*LT**2*R**2*dx**4*mB**2*np.cos(2*theta)-18*LT**2*R**2*dx**4*mB**2*np.cos(theta)**2+2*IW*LT**2*dx**2*mB*np.cos(2*theta)-6*LT**4*R**2*dtheta**2*dx**2*mB**2*np.cos(2*theta)+12*LT**4*R**2*dtheta**2*dx**2*mB**2*np.cos(theta)**2+12*IW*LT**3*dtheta*dx*mB*np.cos(theta)+2*IB*LT**2*R**2*dtheta**2*mB*np.cos(theta)**2+24*LT**3*R**2*dtheta*dx**3*mB**2*np.cos(theta)+12*LT**5*R**2*dtheta**3*dx*mB**2*np.cos(theta)+2*LT**2*R**2*dx**2*mB*mW*np.cos(2*theta)-24*LT**3*R**2*dtheta*dx**3*mB**2*np.cos(theta)**3+12*LT**3*R**2*dtheta*dx*mB*mW*np.cos(theta)-12*LT**4*R**2*dtheta**2*dx**2*mB**2*np.cos(2*theta)*np.cos(theta)**2+6*IB*LT*R**2*dtheta*dx*mB*np.cos(theta)-12*LT**5*R**2*dtheta**3*dx*mB**2*np.cos(2*theta)*np.cos(theta))))

    # POSITION
    dpx = np.cos(phi) * dx
    dpy = np.sin(phi) * dx

    return [dtheta, ddtheta, dphi, ddphi, dx, ddx, dpx, dpy]

# Parametry fizyczne
IB, IW,  IV, R, b, LT, mB, mW, g = 0.1, 0.02, 0.05, 0.05, 0.3, 0.2, 1, 0.5, 9.81
params = (IB, IV, R, b, LT, mB, mW, g)

# Warunki początkowe
sim_time = 5
t = np.linspace(0, sim_time, 1000)
initial_state = [0.01, 0, 3.14/4, 0, 0, 0, 0, 0]
solution = odeint(equations, initial_state, t, args=(params,))

# Tworzenie układu wykresów
fig = plt.figure(figsize=(12, 6))
gs = GridSpec(2, 3, width_ratios=[2, 1, 1])

ax_anim = plt.subplot(gs[0, :])
ax_traj = plt.subplot(gs[1, 0])
ax_theta = plt.subplot(gs[1, 1])
ax_phi = plt.subplot(gs[1, 2])

# --- Wykres trajektorii ---
traj_line, = ax_traj.plot([], [], label='Path')
ax_traj.set_xlim(min(solution[:, 6]) - R, max(solution[:, 6]) + R)
ax_traj.set_ylim(min(solution[:, 7]) - R, max(solution[:, 7]) + R)
ax_traj.set_title("Tracking Trajectory")
ax_traj.set_aspect('equal')
ax_traj.legend()

# --- Wykresy kątów ---
theta_line, = ax_theta.plot([], [], 'r', label="θ (Wheel tilt angle)")
phi_line, = ax_phi.plot([], [], 'g', label="φ (Robot orientation angle)")

for ax in [ax_theta, ax_phi]:
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(min(solution[:, 0])-0.1, max(solution[:, 0])+0.1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (rad)")
    ax.legend()

# --- Animacja ---
ax_anim.set_xlim(min(solution[:, 4]) - R, max(solution[:, 4]) + R)
ax_anim.set_ylim(-0.2, 0.3)
ax_anim.set_title("Self Balancing Robot Animation 3-DOF")
ax_anim.set_aspect('equal')

# Tworzenie koła jako obiektu patch
wheel = patches.Circle((0, R), R, fc='b', ec='k')
ax_anim.add_patch(wheel)

# Ciało robota
body, = ax_anim.plot([], [], 'r-', lw=5)

# Dane do wykresów
traj_x = []
traj_y = []
theta_t = []
theta_vals = []
phi_t = []
phi_vals = []

def init():
    wheel.set_center((0, R))
    body.set_data([], [])
    traj_line.set_data([], [])
    theta_line.set_data([], [])
    phi_line.set_data([], [])
    return wheel, body, traj_line, theta_line, phi_line

def update(frame):
    if frame == 0:  # Jeśli zaczynamy od nowa, czyścimy dane
        traj_x.clear()
        traj_y.clear()
        theta_t.clear()
        theta_vals.clear()
        phi_t.clear()
        phi_vals.clear()

    x = solution[:, 4][frame]  # Pozycja wózka
    theta = solution[:, 0][frame]  # Kąt nachylenia
    phi = solution[:, 2][frame]  # Kąt przechyłu

    # Aktualizacja pozycji koła
    wheel.set_center((x, R))

    # Pozycje końców korpusu
    x1, y1 = x, R
    x2, y2 = x1 + LT * np.sin(theta), y1 + LT * np.cos(theta)
    body.set_data([x1, x2], [y1, y2])

    # Aktualizacja wykresu trajektorii
    traj_x.append(solution[:, 6][frame])
    traj_y.append(solution[:, 7][frame])
    traj_line.set_data(traj_x, traj_y)

    # Aktualizacja wykresu theta
    theta_t.append(t[frame])
    theta_vals.append(theta)
    theta_line.set_data(theta_t, theta_vals)

    # Aktualizacja wykresu phi
    phi_t.append(t[frame])
    phi_vals.append(phi)
    phi_line.set_data(phi_t, phi_vals)

    return wheel, body, traj_line, theta_line, phi_line

ani = animation.FuncAnimation(fig, update, frames=len(t), init_func=init, blit=True, interval=(sim_time*1000)/len(t)-10)

plt.tight_layout()
plt.show()
