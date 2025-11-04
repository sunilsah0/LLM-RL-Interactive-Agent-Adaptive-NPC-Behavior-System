LLM + RL Interactive Agent: Adaptive NPC Behavior System

🌟 Project Overview

This project demonstrates a novel Hybrid AI Agent architecture combining the strategic reasoning capabilities of a Large Language Model (LLM) with the real-time, adaptive decision-making of a Reinforcement Learning (RL) agent.

The system is modeled as a simple Guard Agent (G) navigating a 6x6 grid with obstacles (#) to intercept a randomly moving Intruder (I). The LLM acts as a high-level commander, dynamically adjusting the RL agent's goals and learning parameters (hyperparameters) in response to strategic prompts, forcing the agent to adapt its behavior instantly.

🛠️ Key Architectural Concepts

The single Python file, llm_rl_agent_demo.py, integrates three main components:

LLM Policy Simulator (High-Level Strategy):

Role: Translates natural language instructions (e.g., "Be aggressive", "Patrol") into quantitative adjustments for the RL algorithm (e.g., changing the exploration rate or reward magnitude).

Goal: Dictates what the agent should try to achieve (Hunt vs. Patrol).

Q-Learning Agent (Low-Level Tactics):

Role: Learns the optimal sequence of actions (Up, Down, Left, Right) to maximize reward in the current environment state.

Adaptivity: Its learning rate and exploration policy are dynamically controlled by the LLM's configuration.

GridWorld Environment (The Game):

Features: Includes fixed obstacles (#) and a constantly moving target (I), forcing the agent to learn a reactive, dynamic policy rather than a fixed path.

🚀 How to Run the Demo (GitHub Codespaces)

The project is designed to be runnable immediately in any standard Python environment, making it perfect for demonstrating on a Codespace or local machine.

Prerequisites

You only need Python 3.6+ installed. The project uses standard libraries (random, time, collections, math).

Running in Codespaces

Open Codespace: If you are viewing this on GitHub, click the "Code" button and select "Open with Codespaces."

Launch Terminal: Once the Codespace initializes, open a new Terminal window.

Execute the File: Run the simulation directly:

python llm_rl_agent_demo.py


Expected Output

The simulation will run for 600 episodes, demonstrating four distinct phases controlled by the simulated LLM policy changes:

Episode Range

LLM Instruction

Agent Behavior Change

0 - 149

default

Standard learning; finding initial paths around walls.

150 - 349

aggressive

Exploration Rate (Epsilon) drops (less randomness); Capture Reward Multiplier increases (prioritizes catching the target faster).

350 - 499

patrol mode

Exploration Rate (Epsilon) rises significantly; Reward structure shifts to encourage maximum movement and path diversity, simulating area coverage.

500 - 600

aggressive hunt

Returns to aggressive parameters, re-focusing the learned policy on interception.

The console output will log these policy changes and show the final grid state periodically, illustrating the agent's real-time adaptation.
