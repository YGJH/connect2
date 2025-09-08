#!/usr/bin/env python3
"""
ConnectX AI對戰程式
讓submission_simple.py與submission2.py進行對戰並可視化過程
"""

import sys
import os
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
import numpy as np
import importlib.util
from datetime import datetime

class ConnectXGame:
    def __init__(self, rows=6, columns=7, starting_player=1):
        self.rows = rows
        self.columns = columns
        self.starting_player = 1 if starting_player not in (1, 2) else starting_player
        self.reset()
        
    def reset(self, starting_player=None):
        self.board = np.zeros((self.rows, self.columns), dtype=int)
        # allow overriding starting player per reset; default to configured
        if starting_player in (1, 2):
            self.current_player = starting_player
        else:
            self.current_player = self.starting_player
        self.winner = None
        self.done = False
        self.moves_history = []
        
    def get_valid_moves(self):
        return [c for c in range(self.columns) if self.board[0][c] == 0]
    
    def make_move(self, column):
        if column not in self.get_valid_moves():
            return False
            
        # 找到該列的最低空位
        for row in range(self.rows-1, -1, -1):
            if self.board[row][column] == 0:
                self.board[row][column] = self.current_player
                self.moves_history.append((row, column, self.current_player))
                break
                
        # 檢查獲勝
        if self.check_win(row, column):
            self.winner = self.current_player
            self.done = True
        elif len(self.get_valid_moves()) == 0:
            self.done = True  # 平局
            
        # 切換玩家
        self.current_player = 2 if self.current_player == 1 else 1
        return True
        
    def check_win(self, row, col):
        piece = self.board[row][col]
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dr, dc in directions:
            count = 1
            # 向一個方向檢查
            r, c = row + dr, col + dc
            while 0 <= r < self.rows and 0 <= c < self.columns and self.board[r][c] == piece:
                count += 1
                r, c = r + dr, c + dc
            # 向相反方向檢查
            r, c = row - dr, col - dc
            while 0 <= r < self.rows and 0 <= c < self.columns and self.board[r][c] == piece:
                count += 1
                r, c = r - dr, c - dc
                
            if count >= 4:
                return True
        return False
    
    def get_observation(self, player):
        return {
            'board': self.board.flatten().tolist(),
            'mark': player
        }
    
    def get_config(self):
        class Config:
            def __init__(self, rows, columns):
                self.rows = rows
                self.columns = columns
        
        return Config(self.rows, self.columns)

class ConnectXVisualizer:
    def __init__(self, game, save_video=False, video_filename=None, player1_name='AI-1 KTK', player2_name='AI-2 Charles'):
        self.game = game
        self.save_video = save_video
        self.video_filename = video_filename or f"connectx_battle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        self.frames = []  # 儲存動畫幀
        self.player1_name = player1_name
        self.player2_name = player2_name
        
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.setup_board()
        
        if self.save_video:
            print(f"📹 影片模式已啟用，將保存為: {self.video_filename}")
        
    def setup_board(self):
        self.ax.set_xlim(-0.5, self.game.columns - 0.5)
        self.ax.set_ylim(-0.5, self.game.rows - 0.5)
        self.ax.set_aspect('equal')
        self.ax.invert_yaxis()  # 讓第0行在上方
        
        # 畫網格
        for i in range(self.game.rows + 1):
            self.ax.axhline(i - 0.5, color='black', linewidth=2)
        for j in range(self.game.columns + 1):
            self.ax.axvline(j - 0.5, color='black', linewidth=2)
            
        # 設置標題和標籤
        self.ax.set_title('ConnectX AI Battle', fontsize=16, fontweight='bold')
        self.ax.set_xlabel('Column', fontsize=12)
        self.ax.set_ylabel('Row', fontsize=12)
        
        # 隱藏刻度
        self.ax.set_xticks(range(self.game.columns))
        self.ax.set_yticks(range(self.game.rows))
        
    def update_display(self, capture_frame=True):
        self.ax.clear()
        self.setup_board()
        
        # 畫棋子
        for row in range(self.game.rows):
            for col in range(self.game.columns):
                if self.game.board[row][col] == 1:
                    circle = plt.Circle((col, row), 0.4, color='red', alpha=0.8)
                    self.ax.add_patch(circle)
                    self.ax.text(col, row, '1', ha='center', va='center', 
                               fontsize=16, fontweight='bold', color='white')
                elif self.game.board[row][col] == 2:
                    circle = plt.Circle((col, row), 0.4, color='blue', alpha=0.8)
                    self.ax.add_patch(circle)
                    self.ax.text(col, row, '2', ha='center', va='center', 
                               fontsize=16, fontweight='bold', color='white')
        
        # 顯示當前玩家
        player_color = 'red' if self.game.current_player == 1 else 'blue'
        player_name = self.player1_name if self.game.current_player == 1 else self.player2_name
        
        if not self.game.done:
            self.ax.text(self.game.columns/2, -1, f'Current Player: {player_name}', 
                        ha='center', va='center', fontsize=14, fontweight='bold', color=player_color)
        else:
            if self.game.winner:
                winner_name = self.player1_name if self.game.winner == 1 else self.player2_name
                winner_color = 'red' if self.game.winner == 1 else 'blue'
                self.ax.text(self.game.columns/2, -1, f'Winner: {winner_name}!', 
                            ha='center', va='center', fontsize=16, fontweight='bold', color=winner_color)
            else:
                self.ax.text(self.game.columns/2, -1, 'Draw!', 
                            ha='center', va='center', fontsize=16, fontweight='bold', color='purple')
        
        # 顯示移動歷史
        moves_text = f"Moves: {len(self.game.moves_history)}"
        self.ax.text(0, -1.5, moves_text, ha='left', va='center', fontsize=12)
        
        # 如果是影片模式，捕獲當前幀
        if self.save_video and capture_frame:
            # 將當前圖像保存到幀列表
            import io
            from PIL import Image
            buf = io.BytesIO()
            self.fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img = Image.open(buf)
            self.frames.append(np.array(img))
            buf.close()
        
        plt.draw()
        plt.pause(0.01)
    
    def save_video_file(self, fps=2):
        """保存影片文件"""
        if not self.frames:
            print("❌ 沒有幀數據可以保存影片")
            return False
            
        try:
            print(f"🎬 正在生成影片... ({len(self.frames)} 幀)")
            
            # 創建videos目錄
            os.makedirs('videos', exist_ok=True)
            video_path = os.path.join('videos', self.video_filename)
            
            # 使用matplotlib的動畫功能生成影片
            fig, ax = plt.subplots(figsize=(10, 8))
            
            def animate(frame_idx):
                ax.clear()
                ax.imshow(self.frames[frame_idx])
                ax.axis('off')
                return []
            
            anim = FuncAnimation(fig, animate, frames=len(self.frames), interval=1000//fps, blit=True)
            
            # 嘗試使用FFmpeg，如果不可用則使用Pillow
            try:
                writer = FFMpegWriter(fps=fps, metadata={'title': 'ConnectX AI Battle'})
                anim.save(video_path, writer=writer)
                print(f"✅ 影片已保存: {video_path}")
            except:
                # 如果FFmpeg不可用，嘗試使用Pillow (GIF格式)
                gif_path = video_path.replace('.mp4', '.gif')
                writer = PillowWriter(fps=fps)
                anim.save(gif_path, writer=writer)
                print(f"✅ GIF動畫已保存: {gif_path}")
            
            plt.close(fig)
            return True
            
        except Exception as e:
            print(f"❌ 保存影片失敗: {e}")
            return False

def load_agent_from_file(file_path):
    """從 .py 檔載入 Kaggle agent：使用 kaggle_environments.utils 解析出最後一個可呼叫物件。
    這樣不依賴 submission.py 內部實作細節，可避免 'output.weight' 類錯誤。
    """
    from kaggle_environments import utils as kaggle_utils
    try:
        code = kaggle_utils.read_file(file_path)
        return kaggle_utils.get_last_callable(code)
    except Exception as e:
        raise e

def create_simple_rule_agent():
    """創建一個簡單的規則型AI作為備用"""
    def agent(obs, config):
        import random
        import numpy as np
        
        def get_board():
            return np.array(obs['board']).reshape(config.rows, config.columns)
        
        def get_valid_moves():
            board = get_board()
            return [c for c in range(config.columns) if board[0][c] == 0]
        
        def check_win(board, piece, col):
            """檢查在col位置放置piece是否能獲勝"""
            row = -1
            for r in range(config.rows-1, -1, -1):
                if board[r][col] == 0:
                    row = r
                    break
            if row == -1:
                return False
                
            board[row][col] = piece
            
            def check_direction(r, c, dr, dc):
                count = 1
                nr, nc = r + dr, c + dc
                while 0 <= nr < config.rows and 0 <= nc < config.columns and board[nr][nc] == piece:
                    count += 1
                    nr, nc = nr + dr, nc + dc
                nr, nc = r - dr, c - dc
                while 0 <= nr < config.rows and 0 <= nc < config.columns and board[nr][nc] == piece:
                    count += 1
                    nr, nc = nr - dr, nc - dc
                return count >= 4
            
            directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
            win = any(check_direction(row, col, dr, dc) for dr, dc in directions)
            board[row][col] = 0
            return win
        
        board = get_board()
        valid_moves = get_valid_moves()
        
        if not valid_moves:
            return 0
        
        my_piece = obs['mark']
        opponent_piece = 1 if my_piece == 2 else 2
        
        # 1. 檢查是否能獲勝
        for col in valid_moves:
            if check_win(board.copy(), my_piece, col):
                return col
        
        # 2. 檢查是否需要阻擋對手獲勝
        for col in valid_moves:
            if check_win(board.copy(), opponent_piece, col):
                return col
        
        # 3. 偏好中心位置
        center_col = config.columns // 2
        if center_col in valid_moves:
            return center_col
        
        # 4. 選擇靠近中心的位置
        center_preference = sorted(valid_moves, key=lambda x: abs(x - center_col))
        return random.choice(center_preference[:3])
    
    return agent

def create_fixed_neural_agent():
    """創建一個修復版的神經網路AI"""
    def agent(obs, config):
        import numpy as np
        import random
        
        # 解析棋盤
        board = np.array(obs['board']).reshape(config.rows, config.columns)
        valid_moves = [c for c in range(config.columns) if board[0][c] == 0]
        
        if not valid_moves:
            return 0
        
        my_piece = obs['mark']
        opponent_piece = 1 if my_piece == 2 else 2
        
        # 簡單的神經網路啟發式評分
        def evaluate_position(col):
            score = 0.0
            
            # 找到放置位置
            row = -1
            for r in range(config.rows-1, -1, -1):
                if board[r][col] == 0:
                    row = r
                    break
            if row == -1:
                return -1000  # 無效位置
            
            # 中心偏好 (神經網路學習到的模式)
            center_distance = abs(col - config.columns // 2)
            score += (config.columns - center_distance) * 0.3
            
            # 檢查是否形成威脅
            test_board = board.copy()
            test_board[row][col] = my_piece
            
            # 簡單的連線檢查
            directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
            for dr, dc in directions:
                count = 1
                # 向兩個方向檢查
                for direction in [1, -1]:
                    r, c = row + dr * direction, col + dc * direction
                    while (0 <= r < config.rows and 0 <= c < config.columns and 
                           test_board[r][c] == my_piece):
                        count += 1
                        r, c = r + dr * direction, c + dc * direction
                
                if count >= 4:
                    score += 100  # 勝利
                elif count == 3:
                    score += 10   # 三連線
                elif count == 2:
                    score += 2    # 二連線
            
            # 檢查是否阻擋對手
            test_board[row][col] = opponent_piece
            for dr, dc in directions:
                count = 1
                for direction in [1, -1]:
                    r, c = row + dr * direction, col + dc * direction
                    while (0 <= r < config.rows and 0 <= c < config.columns and 
                           test_board[r][c] == opponent_piece):
                        count += 1
                        r, c = r + dr * direction, c + dc * direction
                
                if count >= 4:
                    score += 50  # 阻擋對手勝利
                elif count == 3:
                    score += 5   # 阻擋對手三連線
            
            # 添加一些隨機性 (模擬神經網路的不確定性)
            score += random.uniform(-0.5, 0.5)
            
            return score
        
        # 評分所有有效動作
        scores = [(col, evaluate_position(col)) for col in valid_moves]
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # 選擇最佳動作
        return scores[0][0]
    
    return agent

def run_battle(agent1_file, agent2_file, delay=1.0, show_animation=True, save_video=False, video_filename=None, swap_sides=False):
    """運行AI對戰"""
    print(f"🚀 開始AI對戰!")
    if save_video:
        print(f"📹 影片錄製模式已啟用")
    print("="*50)
    
    # 載入AI agent
    try:
        agent1 = load_agent_from_file(agent1_file)
        agent2 = load_agent_from_file(agent2_file)
        print(f"✅ 成功載入兩個AI!")
    except Exception as e:
        print(f"❌ 載入AI失敗: {e}")
        return None
    
    # 依據檔名給預設顯示名稱
    def default_name(path, fallback):
        try:
            base = os.path.basename(path)
            return os.path.splitext(base)[0]
        except Exception:
            return fallback

    agent1_name = default_name(agent1_file, "AI-1")
    agent2_name = default_name(agent2_file, "AI-2")
    
    # 依據是否對調，決定玩家與顏色對應
    # Player 1 = 紅色先手，Player 2 = 藍色後手
    if not swap_sides:
        player1_agent, player1_name = agent1, agent1_name
        player2_agent, player2_name = agent2, agent2_name
    else:
        player1_agent, player1_name = agent2, agent2_name
        player2_agent, player2_name = agent1, agent1_name

    print(f"Player 1 (紅色，先手): {player1_name}  <- {agent2_file if swap_sides else agent1_file}")
    print(f"Player 2 (藍色，後手): {player2_name}  <- {agent1_file if swap_sides else agent2_file}")

    # 創建遊戲和可視化（預設先手為Player 1）
    game = ConnectXGame(starting_player=1)
    
    visualizer = None
    if show_animation or save_video:
        visualizer = ConnectXVisualizer(
            game,
            save_video=save_video,
            video_filename=video_filename,
            player1_name=player1_name,
            player2_name=player2_name
        )
        if show_animation:
            plt.ion()  # 開啟互動模式
            plt.show()
        
        # 初始顯示
        visualizer.update_display()
        time.sleep(1)
    
    # 遊戲主循環
    move_count = 0
    while not game.done:
        move_count += 1
        current_agent = player1_agent if game.current_player == 1 else player2_agent
        agent_name = player1_name if game.current_player == 1 else player2_name
        
        try:
            # 獲取AI的動作
            obs = game.get_observation(game.current_player)
            config = game.get_config()
            action = current_agent(obs, config)
            
            print(f"Move {move_count}: {agent_name} chooses column {action}")
            
            # 執行動作
            if not game.make_move(action):
                print(f"❌ {agent_name} made invalid move: {action}")
                break
                
            # 更新顯示
            if visualizer:
                visualizer.update_display()
                if show_animation:
                    time.sleep(delay)
                
        except Exception as e:
            print(f"❌ {agent_name} error: {e}")
            print("🔄 這個錯誤已被AI對戰系統捕獲，遊戲繼續...")
            # 如果AI出錯，隨機選擇一個有效動作
            valid_moves = game.get_valid_moves()
            if valid_moves:
                import random
                action = random.choice(valid_moves)
                print(f"🎲 {agent_name} 使用隨機動作: {action}")
                if not game.make_move(action):
                    break
                if visualizer:
                    visualizer.update_display()
                    if show_animation:
                        time.sleep(delay)
            else:
                break
    
    # 添加最終狀態的額外幀（影片結尾停留）
    if visualizer and save_video:
        for _ in range(5):  # 添加5幀相同的結束畫面
            visualizer.update_display()
    
    # 顯示結果
    print("\n" + "="*50)
    print("🎯 對戰結果:")
    if game.winner == 1:
        print(f"🏆 {player1_name} 獲勝!")
    elif game.winner == 2:
        print(f"🏆 {player2_name} 獲勝!")
    else:
        print("🤝 平局!")
    
    print(f"總共進行了 {move_count} 步")
    print("="*50)
    
    # 保存影片
    if visualizer and save_video:
        print("\n📹 正在生成影片...")
        if visualizer.save_video_file(fps=2):
            print("✅ 影片生成成功!")
        else:
            print("❌ 影片生成失敗")
    
    if show_animation:
        print("\n按任意鍵關閉視窗...")
        plt.ioff()
        plt.show()  # 保持視窗開啟直到用戶關閉
    
    return {
        'winner': game.winner,
        'moves': move_count,
        'board': game.board.copy(),
        'history': game.moves_history.copy(),
        'video_saved': save_video
    }

def main():
    """主函數"""
    print("🎮 ConnectX AI 對戰程式")
    print("讓不同的AI進行ConnectX對戰並可視化過程")
    print()
    
    # 詢問用戶是否要錄製影片
    print("選擇模式:")
    print("1. 即時觀看對戰 (預設)")
    print("2. 錄製影片 (MP4/GIF)")
    print("3. 同時觀看和錄製")
    print("4. 僅生成影片 (不顯示視窗)")
    
    try:
        choice = input("\n請輸入選項 (1-4) [預設=1]: ").strip()
        if not choice:
            choice = "1"
    except KeyboardInterrupt:
        print("\n🛑 用戶取消")
        return
    
    # 根據選擇設置參數
    show_animation = choice in ["1", "3"]
    save_video = choice in ["2", "3", "4"]
    
    if choice == "4":
        show_animation = False  # 僅生成影片模式
    
    video_filename = None
    if save_video:
        try:
            custom_name = input("輸入影片檔名 (按Enter使用預設名稱): ").strip()
            if custom_name:
                if not custom_name.endswith(('.mp4', '.gif')):
                    custom_name += '.mp4'
                video_filename = custom_name
        except KeyboardInterrupt:
            print("\n🛑 用戶取消")
            return
    
    # 設置AI文件路徑
    agent1_file = "submission.py"      # 第一個AI檔案（標籤用）
    agent2_file = "submission_vMega.py"       # 第二個AI檔案（標籤用）
    
    # 檢查文件是否存在
    if not os.path.exists(agent1_file):
        print(f"❌ 找不到 {agent1_file}")
        return
    if not os.path.exists(agent2_file):
        print(f"❌ 找不到 {agent2_file}")
        return
    
    # 是否對調先後手
    try:
        swap_input = input("是否對調先後手順序? (y/N): ").strip().lower()
        swap_sides = swap_input in ("y", "yes")
    except KeyboardInterrupt:
        print("\n🛑 用戶取消")
        return

    # 設置播放速度
    delay = 1.5
    if save_video and not show_animation:
        delay = 0.1  # 僅錄製模式時加速
    
    # 開始對戰
    try:
        result = run_battle(
            agent1_file, 
            agent2_file, 
            delay=delay, 
            show_animation=show_animation,
            save_video=save_video,
            video_filename=video_filename,
            swap_sides=swap_sides
        )
        
        if result:
            print(f"\n🎊 對戰完成!")
            if result.get('video_saved'):
                print("📹 影片已成功保存到 videos/ 目錄")
            
    except KeyboardInterrupt:
        print("\n🛑 用戶中斷對戰")
    except Exception as e:
        print(f"\n❌ 對戰過程中發生錯誤: {e}")

if __name__ == "__main__":
    main()
