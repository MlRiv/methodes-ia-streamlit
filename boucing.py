import argparse
import streamlit as st
import torch
import torch.nn as nn
from torchdiffeq import odeint
import matplotlib.pyplot as plt
from torchdiffeq import odeint, odeint_adjoint
from torchdiffeq import odeint_event

torch.set_default_dtype(torch.float64)

"""The BouncingBallExample class is a PyTorch nn.Module that implements a bouncing ball simulation. 
The class takes in several parameters such as radius, gravity, and adjoint, and initializes them as parameters for the neural network model. 
The forward function takes in a t and state argument, where state is a tuple of pos, vel, and log_radius, representing the position, velocity, and log of the radius of the ball, 
respectively. The function returns the derivative of pos, vel, and log_radius. The event_fn function determines the collision event between the ball and the ground and returns a positive
value if the ball is in mid-air and negative if the ball is on the ground. The get_initial_state function returns the initial state of the simulation. 
The state_update function updates the state based on an event (collision). The get_collision_times function returns the time at which a collision occurs. 
The simulate function runs the bouncing ball simulation and returns the times, trajectory, velocity, and event times of the simulation. 
The gradcheck function checks the gradient of the simulation. Finally, the app_bouncing_ball function initializes the simulation with default parameters and runs the simulation on Streamlit."""
class BouncingBallExample(nn.Module):
    def __init__(self, radius=0.2, gravity=9.8, adjoint=False):
        super().__init__()
        self.gravity = nn.Parameter(torch.as_tensor([gravity]))
        self.log_radius = nn.Parameter(torch.log(torch.as_tensor([radius])))
        self.t0 = nn.Parameter(torch.tensor([0.0]))
        self.init_pos = nn.Parameter(torch.tensor([10.0]))
        self.init_vel = nn.Parameter(torch.tensor([0.0]))
        self.absorption = nn.Parameter(torch.tensor([0.2]))
        self.odeint = odeint_adjoint if adjoint else odeint

    def forward(self, t, state):
        pos, vel, log_radius = state
        dpos = vel
        dvel = -self.gravity
        return dpos, dvel, torch.zeros_like(log_radius)

    def event_fn(self, t, state):
        # positive if ball in mid-air, negative if ball within ground.
        pos, _, log_radius = state
        return pos - torch.exp(log_radius)

    def get_initial_state(self):
        state = (self.init_pos, self.init_vel, self.log_radius)
        return self.t0, state

    def state_update(self, state):
        """Updates state based on an event (collision)."""
        pos, vel, log_radius = state
        pos = (
            pos + 1e-7
        )  # need to add a small eps so as not to trigger the event function immediately.
        vel = -vel * (1 - self.absorption)
        return (pos, vel, log_radius)

    def get_collision_times(self, nbounces=1):

        event_times = []

        t0, state = self.get_initial_state()

        for i in range(nbounces):
            event_t, solution = odeint_event(
                self,
                state,
                t0,
                event_fn=self.event_fn,
                reverse_time=False,
                atol=1e-8,
                rtol=1e-8,
                odeint_interface=self.odeint,
            )
            event_times.append(event_t)

            state = self.state_update(tuple(s[-1] for s in solution))
            t0 = event_t

        return event_times

    def simulate(self, nbounces=1):
        event_times = self.get_collision_times(nbounces)

        # get dense path
        t0, state = self.get_initial_state()
        trajectory = [state[0][None]]
        velocity = [state[1][None]]
        times = [t0.reshape(-1)]
        for event_t in event_times:
            tt = torch.linspace(
                float(t0), float(event_t), int((float(event_t) - float(t0)) * 50)
            )[1:-1]
            tt = torch.cat([t0.reshape(-1), tt, event_t.reshape(-1)])
            solution = odeint(self, state, tt, atol=1e-8, rtol=1e-8)

            trajectory.append(solution[0][1:])
            velocity.append(solution[1][1:])
            times.append(tt[1:])

            state = self.state_update(tuple(s[-1] for s in solution))
            t0 = event_t

        return (
            torch.cat(times),
            torch.cat(trajectory, dim=0).reshape(-1),
            torch.cat(velocity, dim=0).reshape(-1),
            event_times,
        )
    
def gradcheck(nbounces):

    system = BouncingBallExample()

    variables = {
        "init_pos": system.init_pos,
        "init_vel": system.init_vel,
        "t0": system.t0,
        "gravity": system.gravity,
        "log_radius": system.log_radius,
    }

    event_t = system.get_collision_times(nbounces)[-1]
    event_t.backward()

    analytical_grads = {}
    for name, p in system.named_parameters():
        for var in variables.keys():
            if var in name:
                analytical_grads[var] = p.grad

    eps = 1e-3

    fd_grads = {}

    for var, param in variables.items():
        orig = param.data
        param.data = orig - eps
        f_meps = system.get_collision_times(nbounces)[-1]
        param.data = orig + eps
        f_peps = system.get_collision_times(nbounces)[-1]
        param.data = orig
        fd = (f_peps - f_meps) / (2 * eps)
        fd_grads[var] = fd

    success = True
    for var in variables.keys():
        analytical = analytical_grads[var]
        fd = fd_grads[var]
        if torch.norm(analytical - fd) > 1e-4:
            success = False
            print(
                f"Got analytical grad {analytical.item()} for {var} param but finite difference is {fd.item()}"
            )

    if not success:
        raise Exception("Gradient check failed.")

    print("Gradient check passed.")

system=BouncingBallExample()


def app_bouncing_ball():
    # st.set_page_config(page_title="Bouncing Ball Simulation")
    st.write("Here is a simple example of a fit of a theoretical bouncing ball. This demonstration is from R.T.Q CHEN (see [4]).")

    st.sidebar.title("Simulation Parameters")
    nbounces = st.sidebar.slider("Select the number of bounces", 1, 50, 10)
    gradcheck(nbounces)
    times, trajectory, velocity, event_times = system.simulate(nbounces=nbounces)
    times = times.detach().cpu().numpy()
    trajectory = trajectory.detach().cpu().numpy()
    velocity = velocity.detach().cpu().numpy()
    event_times = torch.stack(event_times).detach().cpu().numpy()
    #fig, ax = plt.subplots()
    plt.figure(figsize=(7, 3.5))

    # Event locations.
    for event_t in event_times:
        plt.plot(
            event_t,
            0.0,
            color="C0",
            marker="o",
            markersize=7,
            fillstyle="none",
            linestyle="",
        )

    (vel,) = plt.plot(
        times, velocity, color="C1", alpha=0.7, linestyle="--", linewidth=2.0
    )
    (pos,) = plt.plot(times, trajectory, color="C0", linewidth=2.0)

    plt.hlines(0, 0, 100)
    plt.xlim([times[0], times[-1]])
    plt.ylim([velocity.min() - 0.02, velocity.max() + 0.02])
    plt.ylabel("Markov State", fontsize=16)
    plt.xlabel("Time", fontsize=13)
    plt.legend([pos, vel], ["Position", "Velocity"], fontsize=16)

    plt.gca().xaxis.set_tick_params(
        direction="in", which="both"
    )  # The bottom will maintain the default of 'out'
    plt.gca().yaxis.set_tick_params(
        direction="in", which="both"
    )  # The bottom will maintain the default of 'out'

    # Hide the right and top spines
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)

    # Only show ticks on the left and bottom spines
    plt.gca().yaxis.set_ticks_position("left")
    plt.gca().xaxis.set_ticks_position("bottom")

    plt.tight_layout()
 #   ax.plot(times, trajectory)
 #   ax.set(xlabel="time", ylabel="position")
    st.pyplot()
st.set_option('deprecation.showPyplotGlobalUse', False)
    # gradcheck(nbounces)


if __name__ == "__main__":
    app_bouncing_ball()
