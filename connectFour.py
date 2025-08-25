import gymnasium as gym
import numpy as np
from gymnasium import spaces
import sys
import gc
import random
import os
from kaggle_environments import make, utils


class ConnectFourEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 20}

    def __init__(self, width=7, height=6, connect=4, render_mode=None):
        super().__init__()
        self.win_count = 0
        self.games_count = 0
        self.width = width
        self.height = height
        self.connect = connect
        self.episode_count = 0
        self.board = np.zeros((self.height, self.width), dtype=np.float32)  # 0=empty, 1=first, 2=second
        self.current_player = 1  # 當前輪到誰 (1 或 2)
        self.last_info = None
        self.label = 1
        self.observation_space = spaces.Dict({
            "board": spaces.Box(low=0, high=2, shape=(self.height * self.width,), dtype=np.float32),
            "mark": spaces.Box(low=1, high=2, shape=(1,), dtype=np.float32)
        })

        self.opponent_list = [self.load_agent(f) for f in os.listdir(os.path.join('opponent')) if f.endswith('.py')]
        self.opponent_list.append('self')
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(self.width)
        
        # submission_vMega 的配置
        self.config = {"rows": self.height, "columns": self.width, "inarow": self.connect}

    def _get_obs(self, b=None):
        # 回傳給 RL agent 的觀察：以 RL agent 為視角，mark 永遠是 1
        return {
            "board": self.board.flatten(),
            "mark": np.array([self.label] , dtype=np.float32)
        }

        super().reset(seed=seed)
        self.board.fill(0)    def reset(self, seed=None, options=None):

        self.episode_count += 1
        self.label = 1
        self.current_player = self.games_count % 2
        self.step_count = 0  # 添加步數計數器
        gc.collect()

        if self.current_player == 1:
            self.step(None)
        
        self.games_count += 1
        self.opponent = np.random.choice(self.opponent_list)
        
        if self.games_count % 200 == 0:
            print(f'win_rate: {(self.win_count / self.games_count):.3f}')
        return self._get_obs(), {}


    def _get_info(self):
        if callable(getattr(self, 'opponent', None)):
            opp_name = getattr(self.opponent, "_source_file",
                               getattr(self.opponent, "__name__", "callable_opponent"))
        else:
            opp_name = str(getattr(self, 'opponent', 'unknown'))

        info = {
            'game_result': None,
            'winner': None,
            'episode_length': getattr(self, 'step_count', 0),
            'total_games': self.games_count,
            'agent_wins': self.win_count,
            'win_rate': self.win_count / max(self.games_count, 1),
            'current_player': self.current_player,
            'board_filled_ratio': np.count_nonzero(self.board) / (self.height * self.width),
            'opponent_type': opp_name,
            'evaluation': 0
        }
        if self._is_winner(self.label):
            info['game_result'] = 'win'
            info['winner'] = self.label
            info['evaluation'] = 2
        elif self._is_winner(3 - self.label):
            info['game_result'] = 'loss'
            info['winner'] = 3 - self.label
            info['evaluation'] = -30
        elif self._is_draw():
            info['game_result'] = 'draw'
            info['evaluation'] = -0.1
        else:
            info['game_result'] = 'ongoing'
            info['evaluation'] = 0.7
        return info


    def load_agent(self, file_path):
        submission = utils.read_file(os.path.join('opponent', file_path))
        agent = utils.get_last_callable(submission)
        # 標記來源檔名，方便統計
        setattr(agent, "_source_file", file_path)
        return agent


    def _get_opponent_action(self):
        temp = self._get_obs()
        temp['board'] = temp['board'].astype(np.int8)
        temp['mark'] = temp['mark'].astype(np.int8).tolist()[0]
        return self.opponent(temp, self.config)

    def step(self, action):
        self.step_count = getattr(self, 'step_count', 0) + 1  # 增加步數計數

        if self.current_player == 1 or action == None: # if 1 then opponent
            if self.opponent == 'self':
                if not self._is_valid_action(action):
                    return self._get_obs(), 1, True, False, {'evaluation': 1} # submission輸出不合法 對手勝利
                
                row = self._next_open_row(action)
                self.board[row , action] = self.label
                info = self._get_info()
                if self._is_winner(self.label):
                    return self._get_obs(), -30, True, False, info   # submission贏了 給-1

                if self._is_draw():
                    return self._get_obs(), -0.1, True, False, info # submission平手 給一點點小懲罰

                self.current_player = 3 - self.current_player
                self.label = 3 - self.label
                return self._get_obs(), 0.01 , False , False, info  # 多活一點 給一點獎勵 

            else:
                action = self._get_opponent_action()
                if not self._is_valid_action(action):
                    return self._get_obs(), 1, True, False, {'evaluation': 1} # submission輸出不合法 對手勝利
                
                row = self._next_open_row(action)
                self.board[row , action] = self.label
                info = self._get_info()
                if self._is_winner(self.label):
                    info['evaluation'] = -info['evaluation']
                    return self._get_obs(), -30, True, False, info   # submission贏了 給-1

                if self._is_draw():
                    return self._get_obs(), -0.1, True, False, info # submission平手 給一點點小懲罰

                self.current_player = 3 - self.current_player
                self.label = 3 - self.label
                return self._get_obs(), 0.01 , False , False, info  # 多活一點 給一點獎勵 
        else:
            if not self._is_valid_action(action):
                return self._get_obs(), -10000, True, False, {} # 輸出不合法動作 給超大懲罰
            row = self._next_open_row(action)
            self.board[row, action] = self.label

            info = self._get_info()
            if self._is_winner(self.label):
                self.win_count+=1
                return self._get_obs(), 2, True, False, info   # 如果贏了給reward

            if self._is_draw():
                return self._get_obs(), -0.1, True, False, info  # 如果平手 給一點點小懲罰

            self.current_player = 3 - self.current_player
            self.label = 3 - self.label
            if self.opponent == 'self':
                return self._get_obs(), 0.01, False, False, info  # 多活一點 給一點獎勵
            ret , reward , terminated , truncated , info = self.step(None)    # 直接讓對手下棋
            return ret , reward , terminated, truncated , info



    def _is_valid_action(self, action):
        if action is None or action < 0 or action >= self.width:
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
        # 水平
        for r in range(H):
            for c in range(W - C + 1):
                if np.all(b[r, c:c+C] == player):
                    return True
        # 垂直
        for c in range(W):
            for r in range(H - C + 1):
                if np.all(b[r:r+C, c] == player):
                    return True
        # 斜線 ↘
        for r in range(H - C + 1):
            for c in range(W - C + 1):
                if all(b[r+i, c+i] == player for i in range(C)):
                    return True
        # 斜線 ↗
        for r in range(H - C + 1):
            for c in range(C - 1, W):
                if all(b[r+i, c-i] == player for i in range(C)):
                    return True
        return False

    def _is_draw(self):
        return np.all(self.board != 0)

        
    def render(self, mode='human'):
        if mode == 'ansi':
            print("\n" + "=" * (self.width * 4 + 1))
            for r in range(self.height):
                row_str = "|"
                for c in range(self.width):
                    v = self.board[r, c]
                    if v == 1:
                        row_str += " R |"  # Red for Agent
                    elif v == 2:
                        row_str += " Y |"  # Yellow for submission_vMega
                    else:
                        row_str += "   |"
                print(row_str)
            print("=" * (self.width * 4 + 1))
            print(" " + " ".join([f" {i} " for i in range(self.width)]))
            
            # 顯示當前狀態
            if self.last_info:
                if self.last_info.get("info") == "win":
                    print(f"🎉 Winner: {self.player_labels[self.last_info['winner']]}")
                elif self.last_info.get("info") == "draw":
                    print("🤝 Draw!")
                elif self.last_info.get("info") == "illegal":
                    print("❌ Illegal move!")
                else:
                    print(f"🎯 Current turn: {self.player_labels[self.current_player]}")
            else:
                print(f"🎯 Current turn: {self.player_labels[self.current_player]}")
            return

        # Human mode with pygame
    def render(self, mode='human'):
        self.opponent = self.load_agent('submission_vMega.py')
        if mode == 'ansi':
            print("\n" + "=" * (self.width * 4 + 1))
            for r in range(self.height):
                row_str = "|"
                for c in range(self.width):
                    v = self.board[r, c]
                    if v == 1:
                        row_str += " R |"  # Red for first player
                    elif v == 2:
                        row_str += " Y |"  # Yellow for second player
                    else:
                        row_str += "   |"
                print(row_str)
            print("=" * (self.width * 4 + 1))
            print(" " + " ".join([f" {i} " for i in range(self.width)]))
            
            # 顯示當前狀態
            player_name = "submission_vMega" if self.current_player == 1 else "RL Agent"
            print(f"🎯 Current turn: {player_name} (Player {self.current_player})")
            print(f"📊 Win rate: {(self.win_count / self.games_count if self.games_count > 0 else 0):.3f}")
            return
        else:
            # Human mode with pygame
            import pygame
            
            if not hasattr(self, 'pygame_initialized'):
                pygame.init()
                self.pygame_initialized = True
                
                # 顏色定義
                self.COLORS = {
                    'background': (240, 248, 255),  # AliceBlue
                    'board': (65, 105, 225),        # RoyalBlue
                    'board_shadow': (25, 25, 112),  # MidnightBlue
                    'player1': (220, 20, 60),       # Crimson (submission_vMega)
                    'player2': (255, 215, 0),       # Gold (RL Agent)
                    'empty': (255, 255, 255),       # White
                    'text': (25, 25, 112),          # MidnightBlue
                    'highlight': (50, 205, 50),     # LimeGreen
                    'border': (105, 105, 105)       # DimGray
                }
                
                # 尺寸設定
                self.CELL_SIZE = 80
                self.MARGIN = 12
                self.TOP_MARGIN = 140
                self.BOTTOM_MARGIN = 60
                
                self.screen_width = self.width * self.CELL_SIZE + (self.width + 1) * self.MARGIN
                self.screen_height = (self.height * self.CELL_SIZE + 
                                    (self.height + 1) * self.MARGIN + 
                                    self.TOP_MARGIN + self.BOTTOM_MARGIN)
                
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
                pygame.display.set_caption("🔴🟡 Connect Four - AI Training")
                self.clock = pygame.time.Clock()
                
                # 字體
                self.font_large = pygame.font.Font(None, 36)
                self.font_medium = pygame.font.Font(None, 28)
                self.font_small = pygame.font.Font(None, 24)

            # 處理pygame事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # 清空螢幕
            self.screen.fill(self.COLORS['background'])
            
            # 繪製標題
            title_text = self.font_large.render("Connect Four", True, self.COLORS['text'])
            title_rect = title_text.get_rect(center=(self.screen_width // 2, 25))
            self.screen.blit(title_text, title_rect)
            
            # 繪製玩家資訊
            player1_text = "🔴 Player 1: submission_vMega"
            player2_text = "🟡 Player 2: RL Agent"

            p1_surface = self.font_medium.render(player1_text, True, self.COLORS['player1'])
            p2_surface = self.font_medium.render(player2_text, True, self.COLORS['player2'])
            
            self.screen.blit(p1_surface, (20, 55))
            self.screen.blit(p2_surface, (20, 80))
            
            # 顯示當前回合
            current_player_name = "submission_vMega" if self.current_player == 1 else "RL Agent"
            turn_text = f"🎯 Turn: {current_player_name}"
            turn_color = self.COLORS['player1'] if self.current_player == 1 else self.COLORS['player2']
            
            turn_surface = self.font_medium.render(turn_text, True, turn_color)
            self.screen.blit(turn_surface, (20, 105))
            
            # 顯示勝率
            win_rate = self.win_count / self.games_count if self.games_count > 0 else 0
            stats_text = f"📊 Games: {self.games_count} | RL Agent Wins: {self.win_count} | Win Rate: {win_rate:.3f}"
            stats_surface = self.font_small.render(stats_text, True, self.COLORS['text'])
            stats_rect = stats_surface.get_rect(center=(self.screen_width // 2, self.TOP_MARGIN - 15))
            self.screen.blit(stats_surface, stats_rect)
            
            # 繪製棋盤背景（帶陰影效果）
            board_start_x = self.MARGIN
            board_start_y = self.TOP_MARGIN
            board_width = self.width * self.CELL_SIZE + (self.width - 1) * self.MARGIN
            board_height = self.height * self.CELL_SIZE + (self.height - 1) * self.MARGIN
            
            # 陰影
            shadow_offset = 4
            shadow_rect = pygame.Rect(
                board_start_x + shadow_offset, 
                board_start_y + shadow_offset, 
                board_width + self.MARGIN, 
                board_height + self.MARGIN
            )
            pygame.draw.rect(self.screen, self.COLORS['board_shadow'], shadow_rect, border_radius=12)
            
            # 主棋盤
            board_rect = pygame.Rect(board_start_x, board_start_y, board_width + self.MARGIN, board_height + self.MARGIN)
            pygame.draw.rect(self.screen, self.COLORS['board'], board_rect, border_radius=12)
            
            # 繪製棋子
            for r in range(self.height):
                for c in range(self.width):
                    # 計算位置
                    x = board_start_x + c * (self.CELL_SIZE + self.MARGIN) + self.MARGIN // 2
                    y = board_start_y + r * (self.CELL_SIZE + self.MARGIN) + self.MARGIN // 2
                    center_x = x + self.CELL_SIZE // 2
                    center_y = y + self.CELL_SIZE // 2
                    radius = self.CELL_SIZE // 2 - 8
                    
                    # 根據棋盤狀態選擇顏色
                    cell_value = self.board[r, c]
                    if cell_value == 1:
                        color = self.COLORS['player1']  # submission_vMega
                    elif cell_value == 2:
                        color = self.COLORS['player2']  # RL Agent
                    else:
                        color = self.COLORS['empty']
                    
                    # 繪製棋子陰影
                    if cell_value != 0:
                        shadow_center = (center_x + 2, center_y + 2)
                        pygame.draw.circle(self.screen, self.COLORS['board_shadow'], shadow_center, radius - 2)
                    
                    # 繪製棋子
                    pygame.draw.circle(self.screen, color, (center_x, center_y), radius)
                    
                    # 繪製棋子邊框
                    border_color = self.COLORS['border'] if cell_value == 0 else self.COLORS['board_shadow']
                    border_width = 2 if cell_value == 0 else 3
                    pygame.draw.circle(self.screen, border_color, (center_x, center_y), radius, border_width)
                    
                    # 為空格子添加光澤效果
                    if cell_value == 0:
                        highlight_radius = radius // 3
                        highlight_center = (center_x - radius // 3, center_y - radius // 3)
                        pygame.draw.circle(self.screen, (255, 255, 255, 100), highlight_center, highlight_radius)
            
            # 繪製列號
            for c in range(self.width):
                col_x = board_start_x + c * (self.CELL_SIZE + self.MARGIN) + self.CELL_SIZE // 2
                col_text = self.font_small.render(str(c), True, self.COLORS['text'])
                col_rect = col_text.get_rect(center=(col_x, self.screen_height - 30))
                self.screen.blit(col_text, col_rect)
            
            # 更新顯示
            pygame.display.flip()
            self.clock.tick(60)  # 60 FPS
