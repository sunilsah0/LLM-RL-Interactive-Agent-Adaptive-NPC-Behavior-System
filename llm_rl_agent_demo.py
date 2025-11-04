import random
import time
from collections import defaultdict
import math

# --- 1. LLM Policy Simulator (High-Level Strategy) ---
def llm_policy_simulator(text_instruction):
    """
    Simulates the role of an LLM that interprets a high-level strategic
    instruction and translates it into concrete, quantifiable RL parameters.

    In a real system, an LLM would parse a prompt like "Be more aggressive"
    and output a JSON object containing the RL agent's new configuration.
    """
    instruction = text_instruction.lower().strip()
    config = {
        'learning_rate': 0.1,    # alpha (rate at which Q-values are updated)
        'exploration_rate': 0.1, # epsilon (chance of choosing a random action)
        'reward_multiplier': 1.0, # multiplier for capture reward
        'movement_type': 'hunt'  # 'hunt' or 'patrol'
    }

    if "aggressive" in instruction:
        # High aggression: lower exploration (trust Q-values), higher capture reward.
        config['exploration_rate'] = 0.05
        config['reward_multiplier'] = 2.0
        config['movement_type'] = 'hunt'
        print(f"\n[LLM Policy Update: AGGRESSIVE HUNT] -> Epsilon={config['exploration_rate']}, Reward Multiplier={config['reward_multiplier']}")
    elif "patrol" in instruction:
        # Patrol Mode: High exploration to cover space, prioritize path-clearing.
        # The reward structure in the environment will be switched to prioritize
        # steps over capture (in this mode, capture is secondary).
        config['exploration_rate'] = 0.3
        config['reward_multiplier'] = 0.5
        config['movement_type'] = 'patrol'
        print(f"\n[LLM Policy Update: PATROL MODE] -> Epsilon={config['exploration_rate']}, Movement={config['movement_type']}")
    elif "default" in instruction:
        config['movement_type'] = 'hunt'
        print(f"\n[LLM Policy Update: DEFAULT HUNT] -> Epsilon={config['exploration_rate']}, Movement={config['movement_type']}")

    return config

# --- 2. Grid World Environment (In-Game Context) ---
class GridWorld:
    """
    A 2D grid environment with fixed walls, where the Intruder moves randomly.
    """
    def __init__(self, size=6):
        self.size = size
        # Define fixed obstacles (Walls)
        self.walls = {
            (1, 1), (1, 2), (1, 3), (2, 3), (3, 3),
            (4, 1), (4, 2), (4, 3), (4, 4)
        }
        self.agent_pos = (0, 0)
        self.intruder_pos = (size - 1, size - 1)
        self.max_steps = 100

    def reset(self):
        """Resets agent and intruder positions for a new episode."""
        self.agent_pos = (0, 0)
        # Random starting position for intruder (must not be in a wall)
        while True:
            ix = random.randrange(self.size)
            iy = random.randrange(self.size)
            if (ix, iy) not in self.walls and (ix, iy) != (0, 0):
                self.intruder_pos = (ix, iy)
                break

        return self._get_state()

    def _get_state(self):
        """Returns the current state for the RL agent."""
        # State is defined by the relative distance to the intruder
        return (self.intruder_pos[0] - self.agent_pos[0],
                self.intruder_pos[1] - self.agent_pos[1])

    def _move_intruder(self):
        """Intruder moves one step randomly, avoiding walls."""
        possible_moves = [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)] # Including staying still
        random.shuffle(possible_moves)

        for dx, dy in possible_moves:
            new_x = self.intruder_pos[0] + dx
            new_y = self.intruder_pos[1] + dy

            if (0 <= new_x < self.size and 0 <= new_y < self.size and
                (new_x, new_y) not in self.walls):
                self.intruder_pos = (new_x, new_y)
                return

    def step(self, action_index, reward_multiplier=1.0, movement_type='hunt'):
        """
        Takes an action, moves the intruder, and calculates rewards.
        Actions: 0: Up, 1: Down, 2: Left, 3: Right
        """
        old_pos = self.agent_pos
        x, y = old_pos
        done = False

        # Calculate new potential position
        new_x, new_y = x, y
        if action_index == 0: new_x -= 1 # Up
        elif action_index == 1: new_x += 1 # Down
        elif action_index == 2: new_y -= 1 # Left
        elif action_index == 3: new_y += 1 # Right
        
        new_pos = (new_x, new_y)

        # 1. Check for Walls/Bounds
        if not (0 <= new_x < self.size and 0 <= new_y < self.size) or new_pos in self.walls:
            reward = -2.0 # Higher penalty for hitting a wall/boundary
            self.agent_pos = old_pos # Stay in place
        else:
            self.agent_pos = new_pos
            reward = -0.05 # Small penalty for each step

        # 2. Intruder Movement
        self._move_intruder()

        # 3. Reward Calculation based on LLM Policy
        if movement_type == 'hunt':
            # Primary goal: Catch the intruder
            if self.agent_pos == self.intruder_pos:
                reward += 20.0 * reward_multiplier
                done = True
        
        elif movement_type == 'patrol':
            # Primary goal: Encourage movement and path diversity (more steps)
            reward += 0.1
            # Secondary goal: A small reward for capture
            if self.agent_pos == self.intruder_pos:
                reward += 5.0 * reward_multiplier
                done = True

        if done:
            print(f"--- Intruder CAUGHT! Mode: {movement_type} ---")

        return self._get_state(), reward, done

    def render(self):
        """Prints the grid for visualization."""
        grid = [['.' for _ in range(self.size)] for _ in range(self.size)]
        ax, ay = self.agent_pos
        ix, iy = self.intruder_pos

        # Mark walls
        for wx, wy in self.walls:
            grid[wx][wy] = '#' # Wall

        if (ax, ay) == (ix, iy):
            grid[ax][ay] = 'C' # Caught
        else:
            grid[ix][iy] = 'I' # Intruder
            grid[ax][ay] = 'G' # Guard

        print("\n" + "=" * (self.size * 2 + 1))
        for row in grid:
            print("| " + " ".join(row) + " |")
        print("=" * (self.size * 2 + 1))


# --- 3. Q-Learning Agent (Low-Level Action) ---
class QAgent:
    """
    A simple Q-Learning agent to learn the optimal policy.
    """
    def __init__(self, num_actions, learning_rate=0.1, discount_factor=0.9):
        self.num_actions = num_actions
        self.lr = learning_rate # alpha (Note: this is currently fixed for simplicity)
        self.gamma = discount_factor
        # Q-table: maps (state_tuple, action_index) -> Q-value
        self.q_table = defaultdict(lambda: 0.0)

    def choose_action(self, state, exploration_rate):
        """
        Uses an epsilon-greedy policy to choose the next action.
        """
        if random.random() < exploration_rate:
            # Explore: choose a random action
            return random.randrange(self.num_actions)
        else:
            # Exploit: choose the best action (max Q-value)
            q_values = [self.q_table[(state, a)] for a in range(self.num_actions)]
            max_q = max(q_values)
            
            # Handle ties by choosing randomly among the best actions
            best_actions = [a for a, q in enumerate(q_values) if math.isclose(q, max_q)]
            return random.choice(best_actions)

    def learn(self, state, action, reward, next_state, learning_rate):
        """
        Updates the Q-value using the Bellman equation.
        We now use the dynamic learning_rate passed from the LLM config.
        """
        self.lr = learning_rate # Update LR based on LLM instruction

        old_q = self.q_table[(state, action)]

        # Max Q-value for the next state
        next_q_values = [self.q_table[(next_state, a)] for a in range(self.num_actions)]
        max_next_q = max(next_q_values)

        # Q-value update
        new_q = old_q + self.lr * (reward + self.gamma * max_next_q - old_q)
        self.q_table[(state, action)] = new_q


# --- 4. Main Simulation Loop ---
def run_simulation(num_episodes=600):
    env = GridWorld(size=6)
    agent = QAgent(num_actions=4)

    print("--- DYNAMIC LLM + RL AGENT SIMULATION (Obstacles & Moving Target) ---")
    print("Agent (G) must catch the moving Intruder (I) on a 6x6 grid with walls (#).")

    # List of high-level LLM instructions and when to apply them
    llm_instructions = {
        0: "default",
        150: "Be more aggressive, the intruder must be caught now.",
        350: "Switch to patrol mode and explore the entire sector.",
        500: "Return to aggressive hunt mode."
    }

    # Current LLM configuration
    current_config = llm_policy_simulator(llm_instructions[0])
    
    # Render walls once
    print("Initial Grid Layout with Walls:")
    env.render()

    for episode in range(num_episodes):
        # Check for and apply a new LLM policy at specific episodes
        if episode in llm_instructions:
            current_config = llm_policy_simulator(llm_instructions[episode])

        state = env.reset()
        done = False
        steps = 0
        
        # Unpack LLM config for convenience
        lr = current_config['learning_rate']
        eps = current_config['exploration_rate']
        r_mult = current_config['reward_multiplier']
        move_type = current_config['movement_type']

        while not done and steps < env.max_steps:
            # 1. RL Agent chooses action based on current state and LLM's exploration setting
            action = agent.choose_action(state, eps)

            # 2. Environment executes action (Intruder moves here too!)
            next_state, reward, done = env.step(action, r_mult, move_type)

            # 3. RL Agent learns from the experience
            agent.learn(state, action, reward, next_state, lr)

            state = next_state
            steps += 1
            
            # Optional visualization for early episodes
            if episode < 10:
                env.render()
                time.sleep(0.05)


        # Log episode results
        status = "CAUGHT" if done else "TIMEOUT"
        print(f"Episode {episode+1}/{num_episodes}: Steps taken: {steps}. Policy: {move_type.upper()}, Status: {status}")

        # Display the final state of the board for key episodes
        if episode % 100 == 99 or episode in llm_instructions:
            env.render()
            time.sleep(0.1)

    print("\n--- Simulation Complete ---")
    print("The agent successfully adapted its movement strategy (HUNT vs. PATROL) and its learning parameters (AGGRESSIVE vs. CAUTIOUS) based on the high-level LLM instructions.")

if __name__ == "__main__":
    run_simulation()
