import gymnasium as gym
import numpy as np
from gymnasium import spaces
import sys
import gc
import random
import os
from kaggle_environments import make, utils
import pygame
import math
import weakref

class ConnectFourEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 20}
    class Config:
        columns = 7
        rows = 6
        inarow = 4
    def __init__(self, width=7, height=6, connect=4, render_mode=None):
        super().__init__()
        self.win_count = 0
        self.agent_piece = 1 # Default, will be set in reset
        self.games_count = 0
        self.width = width
        self.shaping_factor=0.01
        self.gamma=0.99
        self.height = height
        self.connect = connect
        self.episode_count = 0
        self.board = np.zeros((self.height, self.width), dtype=np.float32)
        self.current_player = 1
        self.last_info = None
        self.label = 1
        self._renderer = None
        self.already_conquer = []
        self.observation_space = spaces.Dict({
            "board": spaces.Box(low=0, high=2, shape=(self.height * self.width,), dtype=np.float32),
            "mark": spaces.Box(low=1, high=2, shape=(1,), dtype=np.float32),
            "action_mask": spaces.Box(low=0, high=1, shape=(self.width,), dtype=np.float32)
        })
        self.folder_path = 'checkopponents'
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

        self.opponent_list = [self.load_agent(f) for f in os.listdir(os.path.join(self.folder_path)) if f.endswith('.py')]
        # self.opponent_list = random.choices(self.opponent_list ,k=3) if len(self.opponent_list) > 5 else self.opponent_list
        print(f'Selected opponents: {self.opponent_list}')
        self.opponent_list.append('self')
        self.opponent_names = [getattr(opp, '_source_file', 'unknown') if callable(opp) else 'self_play' for opp in self.opponent_list]
        print(", ".join(self.opponent_names))
        self.opponent_stats = {name: {'games': 0, 'agent_wins': 0, 'opponent_wins': 0, 'draws': 0} for name in self.opponent_names}
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(self.width)
        self.config = ConnectFourEnv.Config()
    def _get_obs(self, b=None):
        action_mask = np.zeros(self.width, dtype=np.float32)
        for col in range(self.width):
            if self._is_valid_action(col):
                action_mask[col] = 1.0
        return {
            "board": self.board.flatten(),
            "mark": np.array([self.label], dtype=np.float32),
            "action_mask": action_mask
        }

    def _compute_heuristic(self, player):
        """Simple heuristic: Count weighted open lines (threats) for player."""
        opp = 3 - player
        score = 0
        rows, cols = self.height, self.width
        
        # Weights: open 2-in-row (+1), open 3-in-row (+10)
        for n, weight in [(2, 1), (3, 10)]:
            # Horizontal
            for r in range(rows):
                for c in range(cols - n + 1):
                    window = self.board[r, c:c+n]
                    if np.all(window == player) and (c > 0 and self.board[r, c-1] == 0 or c == 0) and (c + n < cols and self.board[r, c+n] == 0 or c + n == cols):
                        score += weight
            
            # Vertical (gravity makes open below rare, but check)
            for c in range(cols):
                for r in range(rows - n + 1):
                    window = self.board[r:r+n, c]
                    if np.all(window == player) and (r > 0 and self.board[r-1, c] == 0 or r == 0) and (r + n < rows and self.board[r+n, c] == 0 or r + n == rows):
                        score += weight
            
            # Diagonal /
            for r in range(rows - n + 1):
                for c in range(cols - n + 1):
                    window = [self.board[r+i, c+i] for i in range(n)]
                    if all(v == player for v in window) and ((r > 0 and c > 0 and self.board[r-1, c-1] == 0) or (r == 0 or c == 0)) and ((r + n < rows and c + n < cols and self.board[r+n, c+n] == 0) or (r + n == rows or c + n == cols)):
                        score += weight
            
            # Diagonal \
            for r in range(rows - n + 1):
                for c in range(n - 1, cols):
                    window = [self.board[r+i, c-i] for i in range(n)]
                    if all(v == player for v in window) and ((r > 0 and c < cols-1 and self.board[r-1, c+1] == 0) or (r == 0 or c == cols-1)) and ((r + n < rows and c - n >= 0 and self.board[r+n, c-n] == 0) or (r + n == rows or c - n < 0)):
                        score += weight
        
        return score

    def _get_reward(self, board, winner, agent_piece):
        filled_ratio = np.count_nonzero(board) / (self.height * self.width)
        base_win = 1.0
        base_loss = -1.0
        length_bonus = 0.5  # Tune: higher encourages efficiency more
        
        if winner == agent_piece:
            return base_win + (1 - filled_ratio) * length_bonus  # +1 to +1.5 (quick win higher)
        else:
            return base_loss + filled_ratio * length_bonus  # -1 to -0.5 (slow loss less bad)
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board.fill(0)
        self.episode_count += 1
        # 確保是 1 或 2
        self.current_player = (self.games_count % 2) + 1
        self.agent_piece = 3 - self.current_player # Agent's piece: if current=1 (opp first), agent=2; if current=2 (agent first), agent=1
        self.label = 1 # Add this line
        self.step_count = 0
        gc.collect()


        self.prev_heuristic = self._compute_heuristic(self.agent_piece) - self._compute_heuristic(3 - self.agent_piece)
        weights = []
        for name in self.opponent_names:
            if name == 'self_play':
                weight = 1.0
            else:
                stats = self.opponent_stats[name]
                games = stats['games']
                if games == 0:
                    weight = 1.0
                else:
                    opp_win_rate = stats['opponent_wins'] / games
                    weight = opp_win_rate * 0.3
            
            weights.append(weight)
    
    
        total_weight = sum(weights)
        if total_weight == 0:
            probabilities = [1.0 / len(weights)] * len(weights)
        else:
            probabilities = [w / total_weight for w in weights]
        idx = np.random.choice(len(self.opponent_list), p=probabilities)
        
        self.opponent = self.opponent_list[idx]
        self._opponent_name_cached = self.opponent_names[idx]
        # print(self._opponent_name_cached in self.opponent_names)
        if self.opponent == 'self':
            self._opponent_name_cached = 'self_play'
        elif callable(self.opponent):
            self._opponent_name_cached = getattr(self.opponent, "_source_file",
                                                getattr(self.opponent, "_source_file", "callable_opponent"))
        else:
            self._opponent_name_cached = str(self.opponent)
       

        import torch
        torch.cuda.empty_cache()

        self.games_count += 1
        return self._get_obs(), {}
    def _get_info(self):
        # 不在這裡判斷勝負 (避免視角混亂)
        info = {
            'game_result': 'ongoing',
            'winner': None,
            'episode_length': getattr(self, 'step_count', 0),
            'total_games': self.games_count,
            'agent_wins': self.win_count,
            'win_rate': self.win_count / max(self.games_count, 1),
            'current_player': self.current_player,
            'board_filled_ratio': np.count_nonzero(self.board) / (self.height * self.width),
            'opponent_type': getattr(self, '_opponent_name_cached', 'unknown'),
            'evaluation': 0.0
        }
        return info
    def load_agent(self, file_path):
        submission = utils.read_file(os.path.join(self.folder_path, file_path))
        agent = utils.get_last_callable(submission)
        setattr(agent, "_source_file", file_path)
        return agent

    def _get_opponent_action(self):
        temp = self._get_obs()
        temp['board'] = temp['board'].astype(np.int8).tolist()
        temp['mark'] = temp['mark'].astype(np.int8).tolist()[0]
        try:
            action = self.opponent(temp, self.config)
        except Exception as e:
            print(f"[get_opponent_action] Opponent error: {e}, falling back to random action")
            valid_actions = np.where(temp['action_mask'] == 1)[0]
            action = np.random.choice(valid_actions) if len(valid_actions) > 0 else 0
        return action
           
    def _update_info(self, info):
        if info['game_result'] != 'ongoing':
            opponent_name = self._opponent_name_cached

        if opponent_name not in self.opponent_stats:
            opponent_name = 'self_play'
        self.opponent_stats[opponent_name]['games'] += 1
        if info['game_result'] == 'win' and info['winner'] == self.agent_piece:
            self.opponent_stats[opponent_name]['agent_wins'] += 1
        elif info['game_result'] == 'loss' or (info['game_result'] == 'win' and info['winner'] != self.agent_piece):
            self.opponent_stats[opponent_name]['opponent_wins'] += 1
        elif info['game_result'] == 'draw':
            self.opponent_stats[opponent_name]['draws'] += 1

    def step(self, action):
        self.step_count = getattr(self, 'step_count', 0) + 1
        
        # Compute shaping reward component (potential-based)
        def get_shaping_reward():
            new_heuristic = self._compute_heuristic(self.agent_piece) - self._compute_heuristic(3 - self.agent_piece)
            shaping = self.gamma * new_heuristic - self.prev_heuristic
            self.prev_heuristic = new_heuristic  # Update for next step
            return self.shaping_factor * shaping  # Small scale to guide without dominating
        
        # Opponent's turn (current_player == 1) or no action provided
        if self.current_player == 1 or action is None:
            info = self._get_info()
            if self.opponent == 'self':
                # Self-play: Agent plays as opponent (but with its own policy)
                if not self._is_valid_action(action):
                    info.update({'evaluation': 0.0})
                    info.update({'game_result': 'win', 'winner': self.agent_piece})
                    self._update_info(info)
                    return self._get_obs(), 0.0, True, False, info
                row = self._next_open_row(action)
                self.board[row, action] = self.label
                # Win/draw check
                if self._is_winner(self.label):
                    reward = self._get_reward(self.board, self.label, self.agent_piece)
                    info.update({'evaluation': reward})
                    info.update({'game_result': 'win' if self.label == self.agent_piece else 'loss', 'winner': self.label})
                    self._update_info(info)
                    return self._get_obs(), reward, True, False, info
                if self._is_draw():
                    info.update({'game_result': 'draw'})
                    info.update({'evaluation': -0.05})  # Slight negative for draw
                    self._update_info(info)
                    return self._get_obs(), -0.05, True, False, info
                # Switch to agent's main role
                self.current_player = 3 - self.current_player
                self.label = 3 - self.label
                shaping_r = get_shaping_reward()
                info.update({'evaluation': shaping_r})
                return self._get_obs(), shaping_r, False, False, info
            else:
                # External opponent
                opp_action = self._get_opponent_action()
                if not self._is_valid_action(opp_action):
                    info.update({'evaluation': 0.0})
                    info.update({'game_result': 'win', 'winner': self.agent_piece})
                    self._update_info(info)
                    return self._get_obs(), 0.0, True, False, info
                row = self._next_open_row(opp_action)
                self.board[row, opp_action] = self.label
                if self._is_winner(self.label):
                    reward = self._get_reward(self.board, self.label, self.agent_piece)
                    info.update({'evaluation': reward})
                    info.update({'game_result': 'loss', 'winner': self.label})
                    self._update_info(info)
                    return self._get_obs(), reward, True, False, info
                if self._is_draw():
                    info.update({'evaluation': -0.05})
                    info.update({'game_result': 'draw'})
                    self._update_info(info)
                    return self._get_obs(), -0.05, True, False, info
                # Switch to agent's turn
                self.current_player = 2
                self.label = self.agent_piece
                shaping_r = get_shaping_reward()
                info.update({'evaluation': shaping_r})
                return self._get_obs(), shaping_r, False, False, info

        # Agent's turn (current_player == 2)
        info = self._get_info()
        if not self._is_valid_action(action):
            info.update({'game_result': 'loss', 'winner': 3 - self.agent_piece})
            info.update({'evaluation': -2.0})  # Moderate penalty for invalid
            self._update_info(info)
            return self._get_obs(), -2.0, True, False, info
        row = self._next_open_row(action)
        self.board[row, action] = self.label
        if self._is_winner(self.label):
            self.win_count += 1
            reward = self._get_reward(self.board, self.label, self.agent_piece)
            info.update({'evaluation': reward})
            info.update({'game_result': 'win', 'winner': self.label})
            self._update_info(info)
            return self._get_obs(), reward, True, False, info
        if self._is_draw():
            info.update({'evaluation': -0.05})
            info.update({'game_result': 'draw'})
            self._update_info(info)
            return self._get_obs(), -0.05, True, False, info
        # Switch to opponent's turn
        self.current_player = 3 - self.current_player
        self.label = 3 - self.label
        shaping_r = get_shaping_reward()
        info.update({'evaluation': shaping_r})
        return self._get_obs(), shaping_r, False, False, info


    def update_opponents(self):
        # 強制清理舊的 opponent_list 和相關引用
        old_opponents = self.opponent_list.copy()  # 複製一份以便清理
        self.opponent_list.clear()  # 清空 set
        self.opponent_names.clear()
        self.opponent_stats.clear()
        
        # 刪除舊 agent 的強引用（如果有）
        for opp in old_opponents:
            if hasattr(opp, '_source_file'):
                del opp  # 嘗試刪除引用
        del old_opponents
        gc.collect()  # 強制垃圾回收

        # 重新初始化
        self.opponent_list.append('self')
        self.opponent_names.append('self_play')
        self.opponent_stats['self_play'] = {'games': 0, 'agent_wins': 0, 'opponent_wins': 0, 'draws': 0}

        # 使用字典追蹤已載入的 agent，避免重複
        loaded_agents = {}  # key: file_path, value: weakref to agent
        files = [f for f in os.listdir(self.folder_path) if f.endswith('.py')]
    
        
        # 限制載入數量（e.g., 最多 5 個外部 agent）
        max_opponents = 5
        selected_files = random.sample(files, max_opponents)  # 簡單取前 5 個，或用 random.sample 如果需要隨機

        for f in selected_files:
            file_path = os.path.join(self.folder_path, f)
            if f in loaded_agents:
                # 如果已載入，直接使用弱引用
                agent_ref = loaded_agents[f]
                agent = agent_ref() if agent_ref() is not None else None
            else:
                agent = self.load_agent(f)
                if agent:
                    loaded_agents[f] = weakref.ref(agent)  # 弱引用，避免強引用
            
            if agent:
                self.opponent_list.append(agent)
                name = getattr(agent, '_source_file', f'unknown_{f}')
                self.opponent_names.append(name)
                self.opponent_stats[name] = {'games': 0, 'agent_wins': 0, 'opponent_wins': 0, 'draws': 0}

        # print(f'Updated opponent stats: {self.opponent_names} opponents loaded')
        gc.collect()  # 再次清理
    def _is_valid_action(self, action):
        if action is None or not isinstance(action, (int, np.integer)) or action < 0 or action >= self.width:
            return False
        return self.board[0, action] == 0
    def _next_open_row(self, col):
        for r in range(self.height - 1, -1, -1):
            if self.board[r, col] == 0:
                return r
        return -1
    def _is_winner(self, player):
        b = self.board
        H, W, C = self.height, self.width, self.connect
        for r in range(H):
            for c in range(W - C + 1):
                if np.all(b[r, c:c+C] == player):
                    return True
        for c in range(W):
            for r in range(H - C + 1):
                if np.all(b[r:r+C, c] == player):
                    return True
        for r in range(H - C + 1):
            for c in range(W - C + 1):
                if all(b[r+i, c+i] == player for i in range(C)):
                    return True
        for r in range(H - C + 1):
            for c in range(C - 1, W):
                if all(b[r+i, c-i] == player for i in range(C)):
                    return True
        return False
    def _is_draw(self):
        return np.all(self.board != 0)
   
        # Human mode with pygame
    def render(self, mode='human'):
        # ASCII 模式
        if mode == 'ansi':
            print("\n" + "=" * (self.width * 4 + 1))
            for r in range(self.height):
                row_str = "|"
                for c in range(self.width):
                    v = self.board[r, c]
                    row_str += " R |" if v == 1 else " Y |" if v == 2 else " |"
                print(row_str)
            print("=" * (self.width * 4 + 1))
            print(" " + " ".join([f" {i} " for i in range(self.width)]))
            return
        # Human/pygame 模式
        if self._renderer is None:
            self._renderer = ConnectFourRenderer(self.width, self.height, self._opponent_name_cached)
            # Pass agent_piece along with other params
        self._renderer.render(self.board, self.current_player, self.win_count, self.games_count, self.agent_piece)
       
    def close(self):
        if self._renderer and self._renderer.pygame_initialized:
            try:
                pygame.quit()
            except Exception as e:
                print(f"[ConnectFourEnv.close] Error during pygame.quit: {e}")
            self._renderer.pygame_initialized = False
            self._renderer = None
        super().close()       
class ConnectFourRenderer:
    def __init__(self, width=7, height=6, _opponent_name_cached='Piyan'):
        self.width = width
        self.height = height
        self._opponent_name_cached = _opponent_name_cached
        self.pygame_initialized = False
        self.animations = []  # Store animation states
        self.particles = []  # Store particle effects
        self.screen = None
        self.font_large = None
        self.font_medium = None
        self.font_small = None
        self.initialize_pygame()  # Initialize Pygame once during construction
    def initialize_pygame(self):
        if not self.pygame_initialized:
            pygame.init()
            self.pygame_initialized = True
            # Colors with modern palette
            self.COLORS = {
                'background_start': (10, 10, 30),
                'background_end': (50, 50, 100),
                'board': (20, 20, 60, 200),
                'board_shadow': (0, 0, 0, 100),
                'player1': (255, 80, 80),
                'player2': (80, 255, 255),
                'empty': (255, 255, 255, 50),
                'text': (200, 200, 255),
                'highlight': (100, 255, 100, 150),
                'border': (50, 50, 80),
                'glow': (255, 255, 255, 80)
            }
            # Dimensions
            self.CELL_SIZE = 80
            self.MARGIN = 12
            self.TOP_MARGIN = 160
            self.BOTTOM_MARGIN = 80
            self.screen_width = self.width * self.CELL_SIZE + (self.width + 1) * self.MARGIN
            self.screen_height = (self.height * self.CELL_SIZE +
                                 (self.height + 1) * self.MARGIN +
                                 self.TOP_MARGIN + self.BOTTOM_MARGIN)
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("🔴🟡 Connect Four - " + self._opponent_name_cached)
            self.clock = pygame.time.Clock()
            # Initialize fonts once
            try:
                self.font_large = pygame.font.SysFont('consolas', 40, bold=True)
                self.font_medium = pygame.font.SysFont('consolas', 30)
                self.font_small = pygame.font.SysFont('consolas', 24)
            except:
                self.font_large = pygame.font.Font(None, 40)
                self.font_medium = pygame.font.Font(None, 30)
                self.font_small = pygame.font.Font(None, 24)
    def add_piece_animation(self, row, col, player):
        """Add a falling animation for a piece"""
        start_y = self.TOP_MARGIN - self.CELL_SIZE
        end_y = self.TOP_MARGIN + row * (self.CELL_SIZE + self.MARGIN) + self.MARGIN // 2 + self.CELL_SIZE // 2
        self.animations.append({
            'row': row,
            'col': col,
            'player': player,
            'y': start_y,
            'end_y': end_y,
            'speed': 20, # Pixels per frame
            't': 0 # For easing
        })
    def update_animations(self):
        """Update all active animations"""
        for anim in self.animations[:]:
            anim['t'] += 0.05 # Animation progress
            if anim['t'] >= 1:
                anim['y'] = anim['end_y']
                self.animations.remove(anim)
            else:
                # Ease-out quadratic
                eased_t = 1 - (1 - anim['t']) ** 2
                anim['y'] = anim['y'] + (anim['end_y'] - anim['y']) * eased_t
    def add_particle(self, x, y, color):
        """Add particle effect at position"""
        for _ in range(5):
            self.particles.append({
                'x': x,
                'y': y,
                'vx': random.uniform(-2, 2),
                'vy': random.uniform(-2, 2),
                'life': 20,
                'color': color
            })
    def update_particles(self):
        """Update particle effects"""
        for particle in self.particles[:]:
            particle['x'] += particle['vx']
            particle['y'] += particle['vy']
            particle['life'] -= 1
            if particle['life'] <= 0:
                self.particles.remove(particle)


    def render(self, board, current_player, win_count, games_count, agent_piece=1):
        if not self.pygame_initialized:
            self.initialize_pygame()  # Fallback in case initialization didn't occur
        self.board = board
        self.current_player = current_player
        self.win_count = win_count
        self.games_count = games_count

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        # Update animations and particles
        self.update_animations()
        self.update_particles()
        # Draw gradient background
        for y in range(self.screen_height):
            t = y / self.screen_height
            r = int(self.COLORS['background_start'][0] + t * (self.COLORS['background_end'][0] - self.COLORS['background_start'][0]))
            g = int(self.COLORS['background_start'][1] + t * (self.COLORS['background_end'][1] - self.COLORS['background_start'][1]))
            b = int(self.COLORS['background_start'][2] + t * (self.COLORS['background_end'][2] - self.COLORS['background_start'][2]))
            pygame.draw.line(self.screen, (r, g, b), (0, y), (self.screen_width, y))
        # Draw title with glow
        title_text = self.font_large.render("Connect Four", True, self.COLORS['text'])
        title_rect = title_text.get_rect(center=(self.screen_width // 2, 30))
        for offset in range(1, 4):
            glow_text = self.font_large.render("Connect Four", True, self.COLORS['glow'])
            self.screen.blit(glow_text, title_rect.move(offset, offset))
            self.screen.blit(glow_text, title_rect.move(-offset, -offset))
        self.screen.blit(title_text, title_rect)
        # Draw player info with icons
        if self._opponent_name_cached == 'self_play':
                player1_text = "🔴 Player 1: AI (Self-Play)"
                player2_text = "🟡 Player 2: AI (Self-Play)"
        else:
            if agent_piece == 1:
                player1_text = "🔴 Player 1: AI Agent"
                player2_text = f"🟡 Player 2: {self._opponent_name_cached}"
            else:
                player1_text = f"🔴 Player 1: {self._opponent_name_cached}"
                player2_text = "🟡 Player 2: AI Agent"
           
        p1_surface = self.font_medium.render(player1_text, True, self.COLORS['player1'])
        p2_surface = self.font_medium.render(player2_text, True, self.COLORS['player2'])
        self.screen.blit(p1_surface, (20, 60))
        self.screen.blit(p2_surface, (20, 90)) # Draw current turn with glow
        current_player_name = self._opponent_name_cached if self.current_player == 1 else "AI Agent"
        turn_text = f"🎯 Turn: {current_player_name}"
        turn_color = self.COLORS['player1'] if self.current_player == 1 else self.COLORS['player2']
        turn_surface = self.font_medium.render(turn_text, True, turn_color)
        turn_rect = turn_surface.get_rect(topleft=(20, 120))
        for offset in range(1, 3):
            glow_surface = self.font_medium.render(turn_text, True, self.COLORS['glow'])
            self.screen.blit(glow_surface, turn_rect.move(offset, offset))
        self.screen.blit(turn_surface, turn_rect)
        # Draw stats
        win_rate = self.win_count / self.games_count if self.games_count > 0 else 0
        stats_text = f"📊 Games: {self.games_count} | AI Wins: {self.win_count} | Win Rate: {win_rate:.3f}"
        stats_surface = self.font_small.render(stats_text, True, self.COLORS['text'])
        stats_rect = stats_surface.get_rect(center=(self.screen_width // 2, self.TOP_MARGIN - 20))
        self.screen.blit(stats_surface, stats_rect)
        # Draw board with shadow
        board_start_x = self.MARGIN
        board_start_y = self.TOP_MARGIN
        board_width = self.width * self.CELL_SIZE + (self.width - 1) * self.MARGIN
        board_height = self.height * self.CELL_SIZE + (self.height - 1) * self.MARGIN
        # Board shadow
        shadow_rect = pygame.Rect(
            board_start_x + 6, board_start_y + 6,
            board_width + self.MARGIN, board_height + self.MARGIN
        )
        pygame.draw.rect(self.screen, self.COLORS['board_shadow'], shadow_rect, border_radius=15)
        # Main board
        board_rect = pygame.Rect(board_start_x, board_start_y,
                               board_width + self.MARGIN, board_height + self.MARGIN)
        pygame.draw.rect(self.screen, self.COLORS['board'], board_rect, border_radius=15)
        # Draw pieces
        for r in range(self.height):
            for c in range(self.width):
                x = board_start_x + c * (self.CELL_SIZE + self.MARGIN) + self.MARGIN // 2
                y = board_start_y + r * (self.CELL_SIZE + self.MARGIN) + self.MARGIN // 2
                center_x = x + self.CELL_SIZE // 2
                center_y = y + self.CELL_SIZE // 2
                radius = self.CELL_SIZE // 2 - 8
                cell_value = self.board[r, c]
                color = (self.COLORS['player1'] if cell_value == 1 else
                        self.COLORS['player2'] if cell_value == 2 else
                        self.COLORS['empty'])
                # Draw piece shadow
                if cell_value != 0:
                    shadow_center = (center_x + 3, center_y + 3)
                    pygame.draw.circle(self.screen, self.COLORS['board_shadow'], shadow_center, radius - 2)
                # Draw piece with glow
                if cell_value != 0:
                    for offset in range(1, 4):
                        pygame.draw.circle(self.screen, self.COLORS['glow'], (center_x, center_y), radius + offset, 1)
                pygame.draw.circle(self.screen, color, (center_x, center_y), radius)
                # Draw empty cell highlight
                if cell_value == 0:
                    highlight_radius = radius // 2
                    highlight_center = (center_x - radius // 3, center_y - radius // 3)
                    pygame.draw.circle(self.screen, self.COLORS['highlight'], highlight_center, highlight_radius)
        # Draw animated pieces
        for anim in self.animations:
            center_x = (board_start_x + anim['col'] * (self.CELL_SIZE + self.MARGIN) +
                       self.MARGIN // 2 + self.CELL_SIZE // 2)
            center_y = anim['y']
            radius = self.CELL_SIZE // 2 - 8
            color = self.COLORS['player1'] if anim['player'] == 1 else self.COLORS['player2']
            # Glow effect
            for offset in range(1, 4):
                pygame.draw.circle(self.screen, self.COLORS['glow'], (center_x, center_y), radius + offset, 1)
            pygame.draw.circle(self.screen, color, (center_x, center_y), radius)
            self.add_particle(center_x, center_y, color)
        # Draw particles
        for particle in self.particles:
            pygame.draw.circle(self.screen, particle['color'],
                             (int(particle['x']), int(particle['y'])), 3)
        # Draw column numbers
        for c in range(self.width):
            col_x = board_start_x + c * (self.CELL_SIZE + self.MARGIN) + self.CELL_SIZE // 2
            col_text = self.font_small.render(str(c), True, self.COLORS['text'])
            col_rect = col_text.get_rect(center=(col_x, self.screen_height - 40))
            self.screen.blit(col_text, col_rect)
        pygame.display.flip()
        self.clock.tick(60)