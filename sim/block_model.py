import time
import numpy as np
import mujoco
import mujoco.viewer
import pickle
import copy


class PIDController:
    def __init__(self, kp, ki, kd, dt, windup_limit=100.0):
        self.kp = np.array(kp)
        self.ki = np.array(ki)
        self.kd = np.array(kd)
        self.dt = dt
        self.integral = np.zeros_like(self.kp)
        self.prev_error = np.zeros_like(self.kp)
        self.windup_limit = windup_limit
    
    def update(self, error):
        p_term = self.kp * error

        self.integral += error * self.dt
        self.integral = np.clip(self.integral, -self.windup_limit, self.windup_limit)
        i_term = self.ki * self.integral
        
        derivative = (error - self.prev_error) / self.dt
        d_term = self.kd * derivative
        
        control = p_term + i_term + d_term
        
        self.prev_error = error.copy()
        
        return control, (p_term, i_term, d_term)

model = mujoco.MjModel.from_xml_path("block_model.xml")
data = mujoco.MjData(model)

start_pose = np.array([0.0, 0.5, 0.0])  # x, z, y
target_pose = np.array([0.0, 1.0, 1.4])

centered_pos = copy.deepcopy(target_pose)

data.qpos[:] = start_pose.copy()
mujoco.mj_forward(model, data)

kp = [50.0, 300.0, 50.0] 
ki = [5.0, 50.0, 5.0]    
kd = [10.0, 20.0, 10.0]  

dt = model.opt.timestep
pid = PIDController(kp, ki, kd, dt, windup_limit=10.0) 

sim_time = 20.0 
num_steps = int(sim_time / dt)

body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block")
mass = model.body_mass[body_id]
print(f"block mass: {mass} kg")
gravity_magnitude = np.sqrt(np.sum(np.square(model.opt.gravity)))
gravity_compensation = mass * gravity_magnitude
print(f"gravity compensation force: {gravity_compensation} N")

simulation_data = {
    'time': [],
    'position': [], 
    'velocity': [],
    'acceleration': [],
    'control': [],
    'target': target_pose
}

# with mujoco.viewer.launch_passive(model, data) as viewer:
#     viewer.cam.lookat[:] = np.array([0, 0.5, 0.5])
#     viewer.cam.distance = 3.0
#     viewer.cam.azimuth = 90
#     viewer.cam.elevation = -20 
    
for step in range(num_steps):
    
    t = data.time

    freq = [ 0.3, 0.12, 0.25 ]

    target_pose[0] = centered_pos[0] + 0.5 * np.sin(2 * np.pi * freq[0] * t)
    target_pose[1] = centered_pos[1] + 0.5 * np.cos(2 * np.pi * freq[1] * t)
    target_pose[2] = centered_pos[2] + 2*np.pi * np.sin(2 * np.pi * freq[2] * t)

    current_time = step * dt
    current_pose = data.qpos.copy()
    current_vel = data.qvel.copy()
    current_acc = data.qacc.copy()
    error = target_pose - current_pose
    control_signal, components = pid.update(error)
    
    # Apply gravity compensation on the vertical channel (slider_z, index 1)
    # Only apply when near the target to avoid overshooting
    if abs(error[1]) < 0.3:  # Only within 30cm of target
        control_signal[1] += gravity_compensation
    
    # Add damping to reduce oscillations (velocity feedback)
    damping = [5.0, 10.0, 5.0]
    damping_forces = -np.array(damping) * current_vel
    control_signal += damping_forces

    control_signal = np.clip(control_signal, -100, 100)
    # data.ctrl[:] = control_signal

    data.qfrc_applied[:] = control_signal
    
    mujoco.mj_step(model, data)
    
    simulation_data['time'].append(current_time)
    simulation_data['position'].append(current_pose)
    simulation_data['velocity'].append(current_vel)
    simulation_data['acceleration'].append(current_acc)
    simulation_data['control'].append(control_signal.copy())

    # viewer.sync()
        
    if step % 100 == 0:
        p_term, i_term, d_term = components
        print(f"Step: {step}, Time: {current_time:.2f}s")
        # print(f"Position: {current_pose}, Error: {error}")
        print(f"Position: {current_pose}")
        print(f"Control: {control_signal}")
        # print(f"Control: {control_signal}, P:{p_term[1]:.1f}, I:{i_term[1]:.1f}, D:{d_term[1]:.1f}")
        print("-" * 50)
    
    # Slow down simulation for better visualization
    # time.sleep(0.005) 

print("Simulation complete.")
# Keep the viewer open for a moment to see the final state
time.sleep(2.0)

for key in ['time', 'position', 'velocity', 'acceleration', 'control']:
    simulation_data[key] = np.array(simulation_data[key])
output_file = 'block_simulation_data.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(simulation_data, f)

print(f"Simulation data saved to {output_file}")