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
        self.board = np.zeros((self.height, self.width), dtype=np.float32)
        self.current_player = 1
        self.last_info = None
        self.label = 1
        self.observation_space = spaces.Dict({
            "board": spaces.Box(low=0, high=2, shape=(self.height * self.width,), dtype=np.float32),
            "mark": spaces.Box(low=1, high=2, shape=(1,), dtype=np.float32)
        })
        self.folder_path = 'opponents'
        self.opponent_list = [self.load_agent(f) for f in os.listdir(os.path.join(self.folder_path)) if f.endswith('.py')]
        self.opponent_list.append('self')
        temp = ", ".join([getattr(opponent, '_source_file', '') for opponent in self.opponent_list])
        print(temp)
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(self.width)
        self.config = {"rows": self.height, "columns": self.width, "inarow": self.connect}

    def _get_obs(self, b=None):
        return {
            "board": self.board.flatten(),
            "mark": np.array([self.label], dtype=np.float32)
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board.fill(0)
        self.episode_count += 1
        self.label = 1
        # 確保是 1 或 2
        self.current_player = (self.games_count % 2) + 1
        self.step_count = 0
        gc.collect()

        # 先決定對手
        if self.episode_count < 500000 and False:
            self.opponent = 'self'
        else:
            self.opponent = np.random.choice(self.opponent_list)

        if self.opponent == 'self':
            self._opponent_name_cached = 'self_play'
        elif callable(self.opponent):
            self._opponent_name_cached = getattr(self.opponent, "_source_file",
                                                 getattr(self.opponent, "_source_file", "callable_opponent"))
        else:
            self._opponent_name_cached = str(self.opponent)

        # 如果對手先手 (設定為 current_player == 1 視作對手)
        if self.current_player == 1:
            # 讓對手落子
            obs, reward, terminated, truncated, info = self.step(None)
            if terminated or truncated:
                # 立即結束 (極少見，但保險)
                self.games_count += 1
                return obs, {}
        self.games_count += 1
        if self.games_count % 200 == 0:
            print(f'win_rate: {(self.win_count / self.games_count):.3f}')
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
        temp['board'] = temp['board'].astype(np.int8)
        temp['mark'] = temp['mark'].astype(np.int8).tolist()[0]
        return self.opponent(temp, self.config)

    def step(self, action):
        self.step_count = getattr(self, 'step_count', 0) + 1

        # 對手行動 (current_player == 1)
        if self.current_player == 1 or action is None:
            info = self._get_info()
            if self.opponent == 'self':
                # 自對弈：用傳入 action (可能是上一個策略輸出)
                if not self._is_valid_action(action):
                    info.update({'evaluation': 0.0})
                    info.update({'game_result': 'win', 'winner': 2})  # 視作我方勝 (對手非法)
                    return self._get_obs(), 0.0, True, False, info
                row = self._next_open_row(action)
                self.board[row, action] = self.label
                # 勝負判斷 (這一步是對手下的，如果對手形成連線 → 我方 loss)
                if self._is_winner(self.label):
                    info.update({'evaluation': -30.0})
                    info.update({'game_result': 'loss', 'winner': self.label})
                    return self._get_obs(), -30.0, True, False, info
                if self._is_draw():
                    info.update({'game_result': 'draw'})
                    info.update({'evaluation': -0.1})

                    return self._get_obs(), -0.1, True, False, info
                # 換我方
                self.current_player = 2
                self.label = 2
                info.update({'evaluation': 0.07})

                return self._get_obs(), 0.07, False, False, info
            else:
                # 外部對手
                opp_action = self._get_opponent_action()
                if not self._is_valid_action(opp_action):
                    info.update({'evaluation': 0.0})
                    info.update({'game_result': 'win', 'winner': 2})
                    return self._get_obs(), 0.0, True, False, info
                row = self._next_open_row(opp_action)
                self.board[row, opp_action] = self.label
                if self._is_winner(self.label):
                    # 對手剛剛贏
                    info.update({'evaluation': -30.0})
                    info.update({'game_result': 'loss', 'winner': self.label})
                    return self._get_obs(), -30.0, True, False, info
                if self._is_draw():
                    info.update({'evaluation': -0.1})
                    info.update({'game_result': 'draw'})
                    return self._get_obs(), -0.1, True, False, info
                # 換我方
                self.current_player = 2
                self.label = 2
                info.update({'evaluation': 0.07})

                return self._get_obs(), 0.07, False, False, info

        # 我方行動 (current_player == 2)
        if not self._is_valid_action(action):
            info = self._get_info()
            info.update({'game_result': 'loss', 'winner': 1})
            return self._get_obs(), -10000.0, True, False, info

        row = self._next_open_row(action)
        self.board[row, action] = self.label  # self.label 應為 2
        info = self._get_info()

        if self._is_winner(self.label):
            self.win_count += 1
            info.update({'evaluation': 2})
            info.update({'game_result': 'win', 'winner': self.label})
            return self._get_obs(), 2.0, True, False, info
        if self._is_draw():
            info.update({'evaluation': -0.1})
            info.update({'game_result': 'draw'})
            return self._get_obs(), -0.1, True, False, info

        # 換對手
        self.current_player = 1
        self.label = 1
        info.update({'evaluation': 0.07})

        # 若自對弈則不自動馬上再呼叫 step(None)，讓外部控制流程；若想自動可遞迴
        return self._get_obs(), 0.07, False, False, info

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
            renderer = ConnectFourRenderer()
            renderer.render(self.board, self.current_player, self.win_count, self.games_count)

class ConnectFourRenderer:
    def __init__(self, width=7, height=6):
        self.width = width
        self.height = height
        self.pygame_initialized = False
        self.animations = []  # Store animation states
        self.particles = []   # Store particle effects

    def initialize_pygame(self):
        if not self.pygame_initialized:
            pygame.init()
            self.pygame_initialized = True

            # Colors with modern palette
            self.COLORS = {
                'background_start': (10, 10, 30),  # Dark space blue
                'background_end': (50, 50, 100),   # Lighter blue for gradient
                'board': (20, 20, 60, 200),       # Semi-transparent dark blue
                'board_shadow': (0, 0, 0, 100),    # Subtle shadow
                'player1': (255, 80, 80),          # Neon red
                'player2': (80, 255, 255),         # Neon cyan
                'empty': (255, 255, 255, 50),      # Transparent white
                'text': (200, 200, 255),           # Light blue text
                'highlight': (100, 255, 100, 150), # Glowing green
                'border': (50, 50, 80),            # Dark border
                'glow': (255, 255, 255, 80)        # Glow effect
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
            pygame.display.set_caption("🔴🟡 Connect Four - Cyber Edition")
            self.clock = pygame.time.Clock()

            # Fonts (use a modern system font or fallback)
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
            'speed': 20,  # Pixels per frame
            't': 0  # For easing
        })

    def update_animations(self):
        """Update all active animations"""
        for anim in self.animations[:]:
            anim['t'] += 0.05  # Animation progress
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

    def render(self, board, current_player, win_count, games_count):
        self.initialize_pygame()
        self.board = board
        self.current_player = current_player
        self.win_count = win_count
        self.games_count = games_count

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

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
        player1_text = "🔴 Player 1: CyberBot"
        player2_text = "🟡 Player 2: AI Agent"
        p1_surface = self.font_medium.render(player1_text, True, self.COLORS['player1'])
        p2_surface = self.font_medium.render(player2_text, True, self.COLORS['player2'])
        self.screen.blit(p1_surface, (20, 60))
        self.screen.blit(p2_surface, (20, 90))

        # Draw current turn with glow
        current_player_name = "CyberBot" if self.current_player == 1 else "AI Agent"
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
