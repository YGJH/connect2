import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.base_class import BaseAlgorithm  # 用來傳入 PPO 模型
class SelfPlayConnectFourEnv(gym.Env):
    def __init__(self, opponent_model: BaseAlgorithm = None):
        super(SelfPlayConnectFourEnv, self).__init__()
        self.rows = 6
        self.cols = 7
        self.board = np.zeros((self.rows, self.cols), dtype=int)  # 棋盤：0=空，1=玩家1，2=玩家2
        self.current_player = 1  # 1: 主代理 (玩家1), 2: 對手 (玩家2)
        self.opponent_model = opponent_model  # PPO 模型，用來當對手 (可為 None，則用隨機)
        
        # 觀察空間: 棋盤狀態 (6x7 的整數矩陣)
        # self.observation_space = spaces.Box(low=0, high=2, shape=(self.rows, self.cols), dtype=int)
        self.observation_space = spaces.Box(low=0.0, high=2.0, shape=(6, 7, 1), dtype=np.float32)

        # 動作空間: 7 個列 (0-6)
        self.action_space = spaces.Discrete(self.cols)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = 1
        return self._get_observation()
    
    def step(self, action):
        if self.current_player == 1:  # 主代理行動
            if not self._is_valid_action(action):
                return self._get_observation(), -1, True, {}  # 非法移動，懲罰並結束
            self._drop_piece(action, self.current_player)
        else:  # 對手行動 (self-play)
            if self.opponent_model:
                obs = self._get_observation()
                opponent_action, _ = self.opponent_model.predict(obs, deterministic=False)
                if self._is_valid_action(opponent_action):
                    self._drop_piece(opponent_action, self.current_player)
                else:
                    # 如果對手動作無效，隨機選擇
                    valid_actions = [a for a in range(self.cols) if self._is_valid_action(a)]
                    if valid_actions:
                        opponent_action = np.random.choice(valid_actions)
                        self._drop_piece(opponent_action, self.current_player)
            else:
                # 如果沒有模型，用隨機對手
                valid_actions = [a for a in range(self.cols) if self._is_valid_action(a)]
                if valid_actions:
                    opponent_action = np.random.choice(valid_actions)
                    self._drop_piece(opponent_action, self.current_player)
        
        done = self._is_game_over()
        reward = self._get_reward()
        
        # 切換玩家
        self.current_player = 3 - self.current_player  # 1 -> 2, 2 -> 1
        
        # 如果輪到對手，自動讓對手行動 (loop until 主代理回合)
        while not done and self.current_player == 2:
            self.step(None)  # 遞歸呼叫，但小心避免無限迴圈
        
        return self._get_observation(), reward, done, {}
    
    def _get_observation(self):
        # 返回棋盤狀態 (6x7)
        return self.board.astype(np.float32).reshape(6, 7, 1)
    
    def _is_valid_action(self, action):
        # 檢查列是否未滿
        return self.board[0, action] == 0
    
    def _drop_piece(self, action, player):
        # 在指定列放置棋子，掉到最底部
        for row in range(self.rows - 1, -1, -1):
            if self.board[row, action] == 0:
                self.board[row, action] = player
                break
    
    def _is_game_over(self):
        # 檢查是否有勝利者或平局
        return self._check_winner(1) or self._check_winner(2) or np.all(self.board != 0)
    
    def _check_winner(self, player):
        # 檢查水平、垂直、對角線是否連成 4 個
        # 水平
        for row in range(self.rows):
            for col in range(self.cols - 3):
                if np.all(self.board[row, col:col+4] == player):
                    return True
        # 垂直
        for col in range(self.cols):
            for row in range(self.rows - 3):
                if np.all(self.board[row:row+4, col] == player):
                    return True
        # 對角線 (左上到右下)
        for row in range(self.rows - 3):
            for col in range(self.cols - 3):
                if np.all([self.board[row+i, col+i] == player for i in range(4)]):
                    return True
        # 對角線 (右上到左下)
        for row in range(self.rows - 3):
            for col in range(3, self.cols):
                if np.all([self.board[row+i, col-i] == player for i in range(4)]):
                    return True
        return False
    
    def _get_reward(self):
        if self._check_winner(1):
            return 1  # 主代理贏
        elif self._check_winner(2):
            return -1  # 對手贏
        elif np.all(self.board != 0):
            return 0  # 平局
        return 0  # 遊戲中，無獎勵
    
    def render(self, mode='human'):
        print(self.board)
        print()