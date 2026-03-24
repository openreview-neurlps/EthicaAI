"""
Coin Game environment for Nash Trap negative control.
A simple 2-player grid game (Lerer & Peysakhovich, 2017).
No tipping points, no irreversible collapse → expected: no Nash Trap.

Used as a negative control to validate that the Nash Trap is
specific to TPSD structure, not a general MARL artifact.
"""
import numpy as np


class CoinGameEnv:
    """
    Simplified Coin Game:
    - 2 players on a 3x3 grid
    - Each step: one coin appears at a random location
    - Players can move up/down/left/right/stay (5 actions)
    - If player i picks up player i's coin: +1 for player i
    - If player i picks up player j's coin: +1 for player i, -2 for player j
    - No tipping points, no resource collapse → purely strategic
    
    Key property: Cooperation means "don't pick up opponent's coins"
    but there's NO irreversible collapse or threshold dynamics.
    """
    
    def __init__(self, grid_size=3, max_steps=50, num_agents=2):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.num_agents = num_agents
        self.action_space_n = 5  # up, down, left, right, stay
        self.obs_dim = 2 * num_agents + 3  # positions + coin info
        self.reset()
    
    def reset(self):
        self.step_count = 0
        # Random starting positions
        self.positions = np.array([
            [np.random.randint(self.grid_size), np.random.randint(self.grid_size)]
            for _ in range(self.num_agents)
        ])
        # Spawn a coin
        self._spawn_coin()
        return self._get_obs()
    
    def _spawn_coin(self):
        """Spawn a coin at random location, assigned to a random player."""
        self.coin_pos = np.array([
            np.random.randint(self.grid_size),
            np.random.randint(self.grid_size)
        ])
        self.coin_owner = np.random.randint(self.num_agents)
    
    def _get_obs(self):
        """Observation: all player positions + coin position + coin owner."""
        obs = []
        for i in range(self.num_agents):
            player_obs = np.zeros(self.obs_dim)
            idx = 0
            for j in range(self.num_agents):
                player_obs[idx] = self.positions[j][0] / self.grid_size
                player_obs[idx+1] = self.positions[j][1] / self.grid_size
                idx += 2
            player_obs[idx] = self.coin_pos[0] / self.grid_size
            player_obs[idx+1] = self.coin_pos[1] / self.grid_size
            player_obs[idx+2] = 1.0 if self.coin_owner == i else 0.0
            obs.append(player_obs)
        return obs
    
    def step(self, actions):
        """
        actions: list of ints, one per agent
        Returns: obs, rewards, done, info
        """
        self.step_count += 1
        
        # Move agents
        moves = {0: [-1, 0], 1: [1, 0], 2: [0, -1], 3: [0, 1], 4: [0, 0]}
        for i, a in enumerate(actions):
            move = moves.get(a, [0, 0])
            self.positions[i] = np.clip(
                self.positions[i] + np.array(move),
                0, self.grid_size - 1
            )
        
        # Check coin collection
        rewards = np.zeros(self.num_agents)
        for i in range(self.num_agents):
            if np.array_equal(self.positions[i], self.coin_pos):
                if i == self.coin_owner:
                    rewards[i] += 1.0  # Own coin: +1
                else:
                    rewards[i] += 1.0  # Opponent's coin: +1 for collector
                    rewards[self.coin_owner] -= 2.0  # -2 for owner
                self._spawn_coin()
                break
        
        done = self.step_count >= self.max_steps
        return self._get_obs(), rewards, done, {}


class CoinGameWithTippingPoint:
    """
    Modified Coin Game with TPSD properties:
    - Shared resource pool that depletes when agents steal coins
    - Below threshold → game ends (irreversible collapse)
    
    This allows us to test: same base game, WITH vs WITHOUT tipping points.
    """
    
    def __init__(self, grid_size=3, max_steps=50, num_agents=2,
                 resource_init=1.0, r_crit=0.2, depletion_rate=0.15):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.num_agents = num_agents
        self.action_space_n = 5
        self.obs_dim = 2 * num_agents + 3 + 1  # +1 for resource
        self.resource_init = resource_init
        self.r_crit = r_crit
        self.depletion_rate = depletion_rate
        self.reset()
    
    def reset(self):
        self.step_count = 0
        self.resource = self.resource_init
        self.positions = np.array([
            [np.random.randint(self.grid_size), np.random.randint(self.grid_size)]
            for _ in range(self.num_agents)
        ])
        self._spawn_coin()
        return self._get_obs()
    
    def _spawn_coin(self):
        self.coin_pos = np.array([
            np.random.randint(self.grid_size),
            np.random.randint(self.grid_size)
        ])
        self.coin_owner = np.random.randint(self.num_agents)
    
    def _get_obs(self):
        obs = []
        for i in range(self.num_agents):
            player_obs = np.zeros(self.obs_dim)
            idx = 0
            for j in range(self.num_agents):
                player_obs[idx] = self.positions[j][0] / self.grid_size
                player_obs[idx+1] = self.positions[j][1] / self.grid_size
                idx += 2
            player_obs[idx] = self.coin_pos[0] / self.grid_size
            player_obs[idx+1] = self.coin_pos[1] / self.grid_size
            player_obs[idx+2] = 1.0 if self.coin_owner == i else 0.0
            player_obs[idx+3] = self.resource  # Resource level
            obs.append(player_obs)
        return obs
    
    def step(self, actions):
        self.step_count += 1
        
        # Move agents
        moves = {0: [-1, 0], 1: [1, 0], 2: [0, -1], 3: [0, 1], 4: [0, 0]}
        for i, a in enumerate(actions):
            move = moves.get(a, [0, 0])
            self.positions[i] = np.clip(
                self.positions[i] + np.array(move),
                0, self.grid_size - 1
            )
        
        # Check coin collection
        rewards = np.zeros(self.num_agents)
        steal_happened = False
        for i in range(self.num_agents):
            if np.array_equal(self.positions[i], self.coin_pos):
                if i == self.coin_owner:
                    rewards[i] += 1.0
                else:
                    rewards[i] += 1.0
                    rewards[self.coin_owner] -= 2.0
                    steal_happened = True
                self._spawn_coin()
                break
        
        info = {}
        
        # Resource dynamics
        if steal_happened:
            self.resource -= self.depletion_rate
        else:
            self.resource = min(1.0, self.resource + 0.02)
        
        # Tipping point check
        done = self.step_count >= self.max_steps
        if self.resource <= 0:
            done = True
            rewards = -np.ones(self.num_agents) * 5.0
            info['collapsed'] = True
        elif self.resource < self.r_crit:
            self.resource = max(0, self.resource - 0.01)
        
        info['resource'] = self.resource
        obs = self._get_obs()
        return obs, rewards, done, info
