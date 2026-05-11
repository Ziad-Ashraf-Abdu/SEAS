# 🫁 Smart Bronchoscope

![ROS 2](https://img.shields.io/badge/ROS_2-Humble%2FJazzy-blue?logo=ros)
![Python 3](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)
![Gazebo](https://img.shields.io/badge/Gazebo-Ignition-orange?logo=gazebo)

A fully integrated ROS 2 simulation and control stack for a robotic bronchoscope. This project bridges the gap between pure robotics mathematics (*Modern Robotics*) and clinical application, featuring a custom kinematic engine, autonomous airway navigation, active collision avoidance, and a real-time AI computer vision diagnostic pipeline.

**Academic Context:** * **Course:** SBE 361 - Introduction to Robotics
* **Institution:** Systems and Biomedical Engineering (SBME), Cairo University
* **Instructor:** Dr. Muhammad Islam

---

## ✨ Key Features

### 🧮 1. Custom Kinematics & Dynamics Engine (`broncho_kinematics.py`)
Built from scratch without relying on external libraries (like MoveIt or Pinocchio), directly implementing the mathematics of rigid-body motions:
* **Forward Kinematics:** Product of Exponentials (PoE) in both Space and Body frames.
* **Velocity Kinematics:** Space/Body Jacobians and dynamic manipulability ellipsoid analysis ($\mu_1, \mu_2, \mu_3$) to detect and avoid singularities.
* **Inverse Kinematics:** Numerical Newton-Raphson iterative solver using the pseudoinverse Body Jacobian ($J_b^\dagger$).
* **Dynamics & Trajectories:** Recursive Newton-Euler inverse dynamics, mass matrices, and 3rd/5th-order polynomial time-scaling.

### 🧠 2. Clinical AI Vision Pipeline (`vision_pipeline.py`)
A real-time image processing node that reads from the Gazebo CMOS camera to classify and localize airway anomalies using multi-spectral HSV masking, morphological cleaning, and shape-feature extraction (solidity, circularity).
* **Detects:** Hemorrhages (Red/Solid), Carcinomas (Red/Irregular), Lipomas (Yellow), Mucus Plugs (Green), and Foreign Bodies (Blue).

### 🛡️ 3. Autonomous Control Stack (`broncho_controller.py`)
* **Collision Guard (Manual Mode):** A safety layer that uses task-space PI feedforward control. It calculates a repulsive twist when the scope nears a wall and maps it back to joint velocities using $J_b^\dagger$, dampening dangerous user inputs while preserving safe parallel motion.
* **AutoPilot (Autonomous Mode):** A computed-torque navigation system that inserts the scope at a constant velocity while continuously evaluating image moments to steer the distal tip away from the bronchial walls.

### 🖥️ 4. Medical Dashboard (`smart_dashboard.py`)
A surgical dark-themed Tkinter GUI running concurrently with the ROS 2 node.
* **Features:** Live telemetry, manipulability readouts, dynamic collision warnings, customizable illumination, an AI bounding-box overlay, and keyboard teleoperation.

---

## 📂 System Architecture

```text
smart_endoscope/
├── config/
│   └── controllers.yaml         # ROS 2 ros2_control hardware interface definitions
├── launch/
│   ├── sim.launch.py            # Launches Gazebo, Robot State Publisher, and Bridge
│   └── display.launch.py        # Launches RViz2 and Joint State Publisher GUI
├── scripts/
│   ├── smart_dashboard.py       # Main ROS 2 Node & Tkinter GUI
│   ├── broncho_controller.py    # Collision Guard & AutoPilot algorithms
│   ├── broncho_kinematics.py    # The core mathematics engine
│   └── vision_pipeline.py       # OpenCV diagnostic logic
├── urdf/
│   └── smart_broncho.urdf       # Kinematic chain and sensor definitions
└── worlds/
    └── airway_world.sdf         # 3D bronchial tube environment with pathologies
```

---

## 🚀 Installation & Setup

**Prerequisites:** Ubuntu 24.04, ROS 2, Gazebo (Ignition), and OpenCV.

1. **Clone the repository** into your ROS 2 workspace:
   ```bash
   cd ~/ros2_ws/src
   git clone https://github.com/YOUR_USERNAME/smart_endoscope.git
   ```

2. **Install Python dependencies:**
   ```bash
   pip install numpy opencv-python pillow
   ```

3. **Build the package:**
   ```bash
   cd ~/ros2_ws
   colcon build --packages-select smart_endoscope
   source install/setup.zsh
   ```

---

## 🎮 Usage

You need two terminals to run the full simulation and control stack.

**Terminal 1: Launch the Gazebo Simulation**
Spawns the `airway_world`, the robotic bronchoscope, and the ROS-Gazebo camera bridges.
```bash
source install/setup.zsh
ros2 launch smart_endoscope sim.launch.py
```

**Terminal 2: Launch the Clinical Dashboard**
Starts the UI, vision pipeline, and kinematics engine.
```bash
source install/setup.zsh
ros2 run smart_endoscope smart_dashboard.py
```

### ⌨️ Controls
| Key | Action |
| :--- | :--- |
| **`W` / `S`** | Insert / Retract scope (Prismatic joint) |
| **`A` / `D`** | Curl Left / Right (Graduated revolute joint bending) |
| **`T`** | Execute Inverse Kinematics to a predefined target |
| **GUI Tick** | Enable/Disable AI Vision Overlay |
| **GUI Tick** | Engage **AUTO** Navigation Mode |

---

## 🔬 Mathematical Implementation Notes

This project directly translates the following theory into Python code:
* **Equation 4.14 (PoE):** $T(\theta) = e^{[S_1]\theta_1} \cdots e^{[S_n]\theta_n}M$
* **Equation 5.11 (Space Jacobian):** $J_{s_i}(\theta) = [Ad_{e^{[S_1]\theta_1} \cdots e^{[S_{i-1}]\theta_{i-1}}}]S_i$
* **Manipulability Index:** $\mu_3 = \sqrt{\det(J_v J_v^T)}$
* **Numerical IK (Newton-Raphson):** $\theta_{i+1} = \theta_i + J_b^\dagger(\theta_i)V_b$

*All equations and algorithms are sourced from "Modern Robotics: Mechanics, Planning, and Control" by Kevin M. Lynch and Frank C. Park.*
