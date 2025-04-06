## Swimmer 


Continuous Control Algo - Emulate the dynamics of a swimmer moving in a fluid. It leverages MuJoCo’s physics engine, including a high-Reynolds drag model, to realistically simulate the effects of fluid dynamics on a slender body

- Swimmer6: swimmer composed of 6 links
- well-shaped, smooth reward function is used
- high rewards when the swimmer’s nose is near or inside the target and decays with increasing distance.

### Observations:

- joints: A 5-dimensional array representing the internal joint angles (excluding the root), which capture the configuration of the swimmer’s articulated segments.

- body_velocities: An 18-dimensional array that encodes the local velocities (both linear and angular components) of the swimmer’s body segments.

- to_target: A 2-dimensional array giving the vector from the swimmer’s “nose” (or head) to the target, usually expressed in the swimmer’s local coordinate frame.


### Actions: 

- Action Space: The environment’s actions are continuous, with each action represented by a vector of 5 elements.
    - Each element of the action vector corresponds to a torque applied at one of the swimmer’s joints. The values are typically bounded,  in interval [-1,1] reflecting continuous nature of control signals

### Reward Calculation: 

- how close the swimmer’s nose is to the target
    - reward of 1 is provided when the nose is within the target region, with a smooth decay
    - continuously adjust its body configuration and generate effective joint torques to move toward and ultimately capture the target