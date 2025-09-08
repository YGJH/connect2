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
    def __init__(self, width=7, height=6, connect=4, render_mode=None, folder_path='checkopponents'):
        super().__init__()
        self.win_count = 0
        self.agent_piece = 1 # Default, will be set in reset
        self.games_count = 0

        self.width = width
        self.folder_path = folder_path
        self.max_opponents = 5


        self.loaded_model = None
        self.shaping_factor=0.01
        self.gamma=0.99
        self.height = height
        self.always_lose_2 = 0
        self.always_lose_1 = 0

        self.is_second = (self.agent_piece == 2)
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
        })
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
        self.opponent_list = [f for f in os.listdir(self.folder_path) if f.endswith('.py')]
        

        if render_mode=='human':
            self.opponent_list = [random.choice(self.opponent_list)]
        else:
            self.opponent_list = random.sample(self.opponent_list , self.max_opponents)



        if len(self.opponent_list) >= 1:
            self.opponent_list = [self.load_agent(f) for f in self.opponent_list]

    
        self.opponent_names = [getattr(opp, '_source_file', 'unknown') if callable(opp) else 'self_play' for opp in self.opponent_list]
        print(", ".join(self.opponent_names))
        self.opponent_stats = {name: {'games': 0, 'agent_wins': 0, 'opponent_wins': 0, 'draws': 0} for name in self.opponent_names}
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(self.width)
        self.config = ConnectFourEnv.Config()
    
        self.update_opponents()
    
    def _get_obs(self):

        mark = np.array([self.agent_piece], dtype=np.float32)

        return {
            "board": self.board.flatten(),
            "mark": mark,
        }
            

    def _get_connection_reward(self, row, col, player):
        """
        計算從指定位置 (row, col) 的最長連線長度，並返回對應的獎勵。
        檢查水平、垂直、對角線方向的連線。
        """
        def count_consecutive(r, c, dr, dc):
            """輔助函數：從 (r, c) 沿方向 (dr, dc) 計算連續棋子數"""
            count = 1  # 包括自己
            # 正方向
            nr, nc = r + dr, c + dc
            while 0 <= nr < self.height and 0 <= nc < self.width and self.board[nr, nc] == player:
                count += 1
                nr += dr
                nc += dc
            # 負方向
            nr, nc = r - dr, c - dc
            while 0 <= nr < self.height and 0 <= nc < self.width and self.board[nr, nc] == player:
                count += 1
                nr -= dr
                nc -= dc
            return count
        
        # 檢查四個方向的最長連線
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # 水平、垂直、對角線 /
        max_length = 1
        for dr, dc in directions:
            length = count_consecutive(row, col, dr, dc)
            if length > max_length:
                max_length = length
        
        # 根據連線長度給予獎勵（可調整）
        if max_length == 3:
            return 0.05
        else:
            return 0.0

    def _get_flipped_obs_for_opponent(self):
        # Flip board: swap 1 and 2
        opp_piece = 3 - self.agent_piece
        return {
            "board": self.board.flatten(),
            "mark": np.array([opp_piece], dtype=np.float32),
        }



    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board.fill(0)
        self.episode_count += 1
        # prob_agent_second = 1.0 if self.always_lose_2 > 10 else 0.7  # 調整這個值來控制機率 (0.5 = 50%, 1.0 = 100%)
        prob_agent_second = 1.0
        
        if random.random() < prob_agent_second:
            # Agent 為後手
            self.current_player = 1  # 先手給對手
            self.agent_piece = 2     # Agent 為玩家 2 (後手)
            self.is_second = True  # 正確定義：agent 是否為後手
        else:
            # Agent 為先手
            self.is_second = False  # 正確定義：agent 是否為後手
            self.current_player = 2  # 後手給對手
            self.agent_piece = 1     # Agent 為玩家 1 (先手)

        self.step_count = 0

        # Opponent selection logic (unchanged)
        weights = []
        for name in self.opponent_names:
            stats = self.opponent_stats[name]
            games = stats['games']
            if games == 0:
                weight = 1.0
            else:
                opp_win_rate = stats['opponent_wins'] / games
                weight = opp_win_rate
            weights.append(weight)
        total_weight = sum(weights)
        if total_weight == 0:
            probabilities = [1.0 / len(weights)] * len(weights)
        else:
            probabilities = [w / total_weight for w in weights]
        idx = np.random.choice(len(self.opponent_list), p=probabilities)
        self.opponent = self.opponent_list[idx]
        self._opponent_name_cached = self.opponent_names[idx]
        
        self._opponent_name_cached = getattr(self.opponent, "_source_file",
                    getattr(self.opponent, "_source_file", "callable_opponent"))
        import torch
        torch.cuda.empty_cache()
        gc.collect()
        self.games_count += 1


        return self._get_obs(), {}

    def _get_info(self):
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
        temp = self._get_flipped_obs_for_opponent()  # Use flipped for external opponents? No, for external, assume opponent handles its own view
        temp['board'] = temp['board'].astype(np.int8).tolist()
        temp['mark'] = temp['mark'].astype(np.int8).tolist()[0]
        try:
            action = self.opponent(temp, self.config)
        except Exception as e:
            print(f"\033[31m[get_opponent_action] Opponent error: {e}, falling back to random action\033[0m")
            valid_actions = np.where(temp['board'].reshape(6, 7)[0,:] == 0 , 1 ,0)
            action = np.random.choice(valid_actions) if len(valid_actions) > 0 else 0
        return action           


    def _update_info(self, info):
        opponent_name = self._opponent_name_cached
        if opponent_name not in self.opponent_stats:
            opponent_name = random.choice(list(self.opponent_stats.keys()))


        if info['game_result'] != 'ongoing':
            self.opponent_stats[opponent_name]['games'] += 1
            if info['winner'] == self.agent_piece:
                self.opponent_stats[opponent_name]['agent_wins'] += 1
            else:
                self.opponent_stats[opponent_name]['opponent_wins'] += 1
            if info['game_result'] == 'draw':
                self.opponent_stats[opponent_name]['draws'] += 1

    def _get_defense_reward(self, row, col, player):
        """
        計算防守獎勵：檢查放置棋子是否阻止了對手的連線
        返回防守獎勵值
        """
        opponent = 3 - player
        
        def count_opponent_consecutive_blocked(r, c, dr, dc):
            """計算在該位置放棋後，阻止對手在某方向的最長連線"""
            # 先移除我們剛放的棋子，看對手原本能連多長
            temp_board = self.board.copy()
            temp_board[r, c] = 0  # 暫時移除
            
            # 檢查如果對手在這個位置放棋，能連多長
            temp_board[r, c] = opponent
            
            count = 1  # 包括該位置
            # 正方向
            nr, nc = r + dr, c + dc
            while 0 <= nr < self.height and 0 <= nc < self.width and temp_board[nr, nc] == opponent:
                count += 1
                nr += dr
                nc += dc
            # 負方向
            nr, nc = r - dr, c - dc
            while 0 <= nr < self.height and 0 <= nc < self.width and temp_board[nr, nc] == opponent:
                count += 1
                nr -= dr
                nc += dc
            return count
        
        # 檢查四個方向：水平、垂直、兩個對角線
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        max_blocked_length = 1
        
        for dr, dc in directions:
            blocked_length = count_opponent_consecutive_blocked(row, col, dr, dc)
            if blocked_length > max_blocked_length:
                max_blocked_length = blocked_length
        
        # 根據阻止的連線長度給予獎勵
        if max_blocked_length == 3:
            return 0.5   # 阻止對手3連線（更重要）
        elif max_blocked_length >= 4:
            return 0.6   # 阻止對手即將獲勝（最重要）
        else:
            return 0.0


    def _get_missed_win_penalty(self, action_taken, player):
        """
        檢查是否錯失了勝利機會
        如果有其他動作能直接獲勝，但 agent 沒有選擇，則返回懲罰值
        """
        penalty = 0.0
        winning_actions = []
        
        # 檢查所有可能的動作，看是否有能直接獲勝的
        for col in range(self.width):
            if self._is_valid_action(col) and col != action_taken:
                # 模擬在該位置放棋
                row = self._next_open_row(col)
                if row >= 0:
                    temp_board = self.board.copy()
                    temp_board[row, col] = player
                    
                    # 檢查是否能獲勝
                    if self._is_winner(player, temp_board):
                        winning_actions.append(col)
        
        # 如果有獲勝機會但沒有選擇，給予懲罰
        if len(winning_actions) > 0:
            penalty = -0.35  # 錯失勝機的懲罰
            # print(f"[MISSED WIN] Agent could win with actions {winning_actions} but chose {action_taken}")
        
        return penalty

    def _get_missed_defense_penalty(self, action_taken, player):
        """
        檢查是否錯失了防守機會
        如果對手下一步可以獲勝，但 agent 沒有阻擋，則返回懲罰值
        """
        penalty = 0.0
        opponent = 3 - player
        critical_defense_actions = []
        
        # 檢查所有可能的動作，看對手是否能在某處獲勝
        for col in range(self.width):
            if self._is_valid_action(col):
                # 模擬對手在該位置放棋
                row = self._next_open_row(col)
                if row >= 0:
                    temp_board = self.board.copy()
                    temp_board[row, col] = opponent
                    
                    # 檢查對手是否能獲勝
                    if self._is_winner(opponent, temp_board):
                        critical_defense_actions.append(col)
        
        # 如果對手有獲勝機會，但 agent 沒有選擇阻擋其中任何一個
        if len(critical_defense_actions) > 0 and action_taken not in critical_defense_actions:
            penalty = -0.3  # 錯失防守的懲罰（比錯失勝機稍輕）
            # print(f"[MISSED DEFENSE] Opponent can win with actions {critical_defense_actions} but agent chose {action_taken}")
        
        return penalty



    def step(self, action=None):  # Allow None for opponent turn
        self.step_count = getattr(self, 'step_count', 0) + 1
        info = self._get_info()
        current_piece = self.agent_piece if self.current_player == 2 else 3 - self.agent_piece
        reward = 0.0  # 初始化 reward
        if self.current_player == 1:  # Opponent's turn
            action_to_use = self._get_opponent_action()
        else:  # Agent's turn
            action_to_use = action

        if not self._is_valid_action(action_to_use):
            if self.current_player == 2:  # Agent illegal move
                info.update({'game_result': 'loss', 'winner': 3 - self.agent_piece})
                info.update({'evaluation': -10.0})
                print(f'[ILLEGAL MOVE] Agent played illegal action {action_to_use}.')
                reward = -10.0
            else:  # Opponent illegal, agent wins
                info.update({'game_result': 'win', 'winner': self.agent_piece})
                info.update({'evaluation': 0.0})
                print(f'[ILLEGAL MOVE] Opponent played illegal action {action_to_use}.')
                reward = 0.0
            self._update_info(info)
            return self._get_obs(), reward, True, False, info

        row = self._next_open_row(action_to_use)
        self.board[row, action_to_use] = current_piece

        if self._is_winner(current_piece):
            if current_piece == self.agent_piece:
                self.win_count += 1
                if self.is_second:
                    temp_f = self.always_lose_2
                    self.always_lose_2 = 0
                else:
                    temp_f = self.always_lose_1
                    self.always_lose_1 = 0
                reward += 1.0 + temp_f * 0.01  # 勝利獎勵
                info.update({'game_result': 'win', 'winner': current_piece})
            else:
                temp_f = 0
                if self.is_second:
                    self.always_lose_2 += 1
                    temp_f = self.always_lose_2
                else:
                    self.always_lose_1 += 1
                    temp_f = self.always_lose_1
                # if temp_f > 10:
                #     print(f'Agent {self.agent_piece} has lost {temp_f} times in a row and is {self.is_second}.')
                reward += -1.0 - temp_f * 0.01  # 失敗獎勵
                info.update({'game_result': 'loss', 'winner': current_piece})
            info.update({'evaluation': reward})
            self._update_info(info)
            return self._get_obs(), reward, True, False, info

        if self._is_draw():
            reward += -0.01  # 平局輕微懲罰
            info.update({'evaluation': reward})
            info.update({'game_result': 'draw'})
            self._update_info(info)
            return self._get_obs(), reward, True, False, info
    
        
        if current_piece == self.agent_piece:
            # 檢查錯失勝機懲罰
            missed_win_penalty = self._get_missed_win_penalty(action_to_use, current_piece)
            reward += missed_win_penalty
            
            # 檢查錯失防守懲罰
            missed_defense_penalty = self._get_missed_defense_penalty(action_to_use, current_piece)
            reward += missed_defense_penalty

            # 攻擊獎勵：自己的連線
            connection_reward = self._get_connection_reward(row, action_to_use, current_piece)
            reward += connection_reward
            
            # 防守獎勵：阻止對手連線
            # defense_reward = self._get_defense_reward(row, action_to_use, current_piece)
            # reward += defense_reward

        self.current_player = 3 - self.current_player
        info.update({'evaluation': reward})
        return self._get_obs(), reward, False, False, info

    def update_opponents(self, must_saved=None):
        # 強制清理舊的 opponent_list 和相關引用
        try:
            import time
            old_opponents = self.opponent_list.copy()  # 複製一份以便清理
            self.opponent_list.clear()  # 清空 set
            self.opponent_names.clear()
            self.opponent_stats.clear()
            
            # 刪除舊 agent 的強引用（如果有）
            for opp in old_opponents:
                if hasattr(opp, '_source_file'):
                    del opp  # 嘗試刪除引用
            del old_opponents

            # 使用字典追蹤已載入的 agent，避免重複
            loaded_agents = {}  # key: file_path, value: weakref to agent
            selected_files = [f for f in os.listdir(self.folder_path) if f.endswith('.py')]
        
            
            # 限制載入數量（e.g., 最多 5 個外部 agent）
            # selected_files = random.sample(files, self.max_opponents)  # 簡單取前 5 個，或用 random.sample 如果需要隨機
            if must_saved != None:
                selected_files.append(must_saved)

            for f in selected_files:
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


            gc.collect()  # 再次清理
        except Exception as e:
            print(f"\033[31m[update_opponents] Error updating opponents: {e}\033[0m")


    def _is_valid_action(self, action):
        if action is None or not isinstance(action, (int, np.integer)) or action < 0 or action >= self.width:
            return False
        return self.board[0, action] == 0
    def _next_open_row(self, col):
        for r in range(self.height - 1, -1, -1):
            if self.board[r, col] == 0:
                return r
        return -1

    def action_masks(self):
        # 回傳合法動作的 bool array (shape = action_space.n)
        valid_moves_mask = np.zeros(self.width, dtype=bool)
        for col in range(self.width):
            if self.board[0, col] == 0:
                valid_moves_mask[col] = True
        return valid_moves_mask

    def _is_winner(self, player, board=None):
        if board is None:
            b = self.board
        else:
            b = board
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
        if self._renderer:
            try:
                pygame.quit()
                self._renderer.close()  # 使用 renderer 的 close 方法
                del self._renderer  # 明確刪除引用
                self._renderer = None

            except Exception as e:
                raise Exception(f"[ConnectFourEnv.close] Error during pygame.quit: {e}")
                print(f"[ConnectFourEnv.close] Error during renderer cleanup: {e}")
       
        # 清理對手列表（避免循環引用）
        if hasattr(self, 'opponent_list'):
            self.opponent_list.clear()
        if hasattr(self, 'opponent_names'):
            self.opponent_names.clear()

        super().close()
class ConnectFourRenderer:
    def __init__(self, width=7, height=6, _opponent_name_cached='Piyan'):
        self.width = width
        self.height = height
        self._opponent_name_cached = _opponent_name_cached
        self.pygame_initialized = False
        self.animations = []
        self.particles = []
        self.screen = None
        self.font_large = None
        self.font_medium = None
        self.font_small = None
        self.clock = None

    def initialize_pygame(self):

        if not self.pygame_initialized:
            try:
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
                # Use default font to avoid file handle issues
                self.font_large = pygame.font.Font(None, 40)
                self.font_medium = pygame.font.Font(None, 30)
                self.font_small = pygame.font.Font(None, 24)
            except Exception as e:
                print(f"[ConnectFourRenderer] Failed to initialize pygame: {e}")
                self.pygame_initialized = False

        else:
            return
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

    def close(self):
        if self.pygame_initialized:
            try:
                # 明確清理字體資源
                if self.font_large:
                    del self.font_large
                if self.font_medium:
                    del self.font_medium
                if self.font_small:
                    del self.font_small
                if self.clock:
                    del self.clock
                if self.screen:
                    del self.screen
                
                # 清理列表
                self.screen = None
                self.font_large = None
                self.font_medium = None
                self.font_small = None
                self.clock = None

                # 最後退出 pygame
                pygame.quit()

            except Exception as e:
                print(f"[ConnectFourRenderer.close] Error: {e}")
            finally:
                self.pygame_initialized = False