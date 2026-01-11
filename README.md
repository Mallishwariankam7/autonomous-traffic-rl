# Autonomous Traffic Signal Control using Reinforcement Learning

This project implements an **AI-based traffic signal control system** using **Deep Reinforcement Learning (DQN)** to optimize traffic flow at a 4-way intersection. The system dynamically adapts to varying traffic densities and emergency vehicles (like ambulances), improving average waiting times and throughput compared to traditional fixed-timer signals.

---

## Features

- **Reinforcement Learning Agent**:  
  - **Algorithm**: Dueling Deep Q-Network (DQN)  
  - **State**: Number of vehicles and waiting times in each lane  
  - **Action**: Switch traffic light phases (Green/Red)  
  - **Reward**: Minimize cumulative waiting time and handle emergency vehicles  

- **Adaptive Traffic Control**:  
  - Automatically prioritizes emergency vehicles  
  - Dynamically adapts to **Low**, **Medium**, and **High** traffic densities  

- **Simulation Environment**:  
  - Custom Python-based simulator for a 4-way intersection  
  - Lane vehicle capacities and probabilistic car spawning  
  - Metrics collected: Average wait time, max lane wait, emergency clearance, throughput  

- **Evaluation & Visualization**:  
  - Compare **Fixed Timer** vs **DQN Agent**  
  - Graphs:
    - Average waiting time vs traffic density
    - Throughput vs traffic density  

---

## Project Structure
```
autonomous-traffic-rl/
├── agents/ # DQN agent implementation
├── baselines/ # Fixed-timer baseline controllers
├── evaluation/ # Scripts for evaluating RL agent and baselines
├── env/ # Custom traffic environment
├── training/ # Training scripts for DQN
├── visualization/ # Plotting scripts
├── checkpoints/ # Saved trained models (.pth)
├── logs/ # Training and evaluation metrics
├── main.py 
├── README.md
├── requirements.txt
└── .gitignore
```