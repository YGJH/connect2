#!/usr/bin/env python3
"""
ConnectX Human vs AI Game - PyQt5 GUI Version

使用方法：
- 點擊列按鈕放置棋子
- 人類是紅色棋子，AI是黃色棋子
- 目標是連續四個棋子（水平、垂直或對角線）
"""

import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QGridLayout, QPushButton, QLabel, 
                            QMessageBox, QFrame, QRadioButton, QButtonGroup, QGraphicsOpacityEffect, QGraphicsColorizeEffect)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QEasingCurve, QPropertyAnimation
from PyQt5.QtGui import QFont, QPalette, QColor
from kaggle_environments import make, utils
import time
from gymnasium.envs.registration import register
from connectFour import *


# 導入AI模型
submission = utils.read_file("submission.py")
agent = utils.get_last_callable(submission)
register(id='ConnectFourEnv', entry_point='connectFour:ConnectFourEnv')  # Adjust if needed
class Config:
    columns = 7
    rows = 6
    inarow = 4
class AIThread(QThread):
    """AI思考線程"""
    move_calculated = pyqtSignal(int)
    
    def __init__(self, env, board, rows, cols, current_player):
        super().__init__()
        self.env = env  # 使用 ConnectFourEnv 實例
        self.board = board
        self.rows = rows
        self.cols = cols
        self.current_player = current_player  # 新增：當前玩家
    

    
    def run(self):
        try:            
            # 使用 ConnectFourEnv 獲取觀察
            self.env.board = self.board.copy()  # 同步棋盤
            self.env.agent_piece = self.current_player  # 同步當前玩家
            # 動態設定 mark 為當前玩家（AI 的角色）
            obs = self.env._get_obs()  # 使用當前狀態的觀察
            print(obs['board'].reshape((self.rows, self.cols)))
            print(obs['mark'])

            # 預測動作（這裡假設你有模型載入，否則需要從 submission.py 載入）
            # 從 submission.py 載入模型（假設 submission.py 包含模型）
            import importlib.util
            spec = importlib.util.spec_from_file_location("submission", "submission.py")
            submission_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(submission_module)
            agent = submission_module.agent  # 假設 submission.py 有這個函數
            
            # 使用模型預測動作
            action = agent(obs, Config())  # 根據你的模型調整參數
            
            # 驗證動作使用 action_mask
            action_mask = obs['action_mask']
            if action_mask[action] == 0:
                # 如果無效，選擇第一個有效動作
                valid_actions = np.where(action_mask == 1)[0]
                if len(valid_actions) > 0:
                    action = int(valid_actions[0])
                else:
                    action = -1  # 無效
            
            self.move_calculated.emit(action)
            
        except Exception as e:
            print(f"AI決策出錯: {e}")
            # 回退到第一個有效動作
            action_mask = obs.get('action_mask', np.ones(7))  # 預設全有效
            valid_actions = np.where(action_mask == 1)[0]
            if len(valid_actions) > 0:
                action = int(valid_actions[0])
            else:
                action = -1
            self.move_calculated.emit(action)


class ConnectXGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.rows = 6
        self.cols = 7
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = 1  # 1 = 人類, 2 = AI
        self.game_over = False
        self.ai_thinking = False
        self.ai_thread = None
        self.ai_starts = False
        # 動畫 / 視覺狀態
        self.animating = False
        self.animation_timer = None
        self.animation_interval_ms = 55
        self.last_move = None
        self.winning_cells = []
        self.win_flash_timer = None
        self.win_flash_state = False
        # 新增：勝利線動畫
        self.winning_line_timer = None
        self.winning_line_anim_step = 0
        self.winning_effects = []  # (cell,effect)
        # 狀態標籤 pulse 動畫
        self.status_pulse_anim = None
        
        # 創建 ConnectFourEnv 實例
        self.env = ConnectFourEnv()  # 使用自定義環境
        
        self.init_ui()
        self._init_status_pulse()
        
    def init_ui(self):
        """初始化用戶界面 (modernized)"""
        self.setWindowTitle("ConnectX - 人類 vs AI")
        self.setFixedSize(860, 770)
        # 應用全域樣式 (玻璃 & 漸層)
        self.setStyleSheet("""
            QMainWindow {background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #1f2d3a, stop:1 #10171e);} 
            QLabel {color: #ecf0f1;}
            QPushButton {font-weight:600; border-radius:8px; padding:6px 10px; background-color:#34495e; color:#ecf0f1; border:1px solid #2c3e50;}
            QPushButton:hover {background-color:#3d5d74;}
            QPushButton:pressed {background-color:#22313f;}
            QPushButton:disabled {background-color:#4b5b66; color:#bdc3c7;}
            QRadioButton {color:#ecf0f1;}
        """)
        
        # 中央Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主佈局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(16)
        main_layout.setContentsMargins(30, 20, 30, 20)
        
        # 標題
        title_label = QLabel("🎮 ConnectX - 人類 vs AI 對戰")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 18, QFont.Bold))
        title_label.setStyleSheet("color: white; margin: 10px;")
        main_layout.addWidget(title_label)
        
        # 狀態標籤
        self.status_label = QLabel("🔴 你的回合！")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.status_label.setStyleSheet("color: white; margin: 5px;")
        main_layout.addWidget(self.status_label)

        # 先手選擇
        starter_layout = QHBoxLayout()
        starter_layout.setAlignment(Qt.AlignCenter)
        starter_label = QLabel("先手：")
        starter_label.setFont(QFont("Arial", 11))
        self.rb_human_first = QRadioButton("我先手")
        self.rb_ai_first = QRadioButton("AI先手")
        self.rb_human_first.setChecked(True)
        self.rb_human_first.setObjectName('human_first')
        self.rb_ai_first.setObjectName('ai_first')
        self.starter_group = QButtonGroup(self)
        self.starter_group.addButton(self.rb_human_first)
        self.starter_group.addButton(self.rb_ai_first)
        self.rb_human_first.toggled.connect(self._on_starter_changed)
        self.rb_ai_first.toggled.connect(self._on_starter_changed)
        starter_layout.addWidget(starter_label)
        starter_layout.addWidget(self.rb_human_first)
        starter_layout.addWidget(self.rb_ai_first)
        main_layout.addLayout(starter_layout)
        
        # 遊戲棋盤框架
        board_frame = QFrame()
        board_frame.setStyleSheet("""
            QFrame {
                background-color: #34495e;
                border: 3px solid #34495e;
                border-radius: 10px;
                padding: 15px;
            }
        """)
        main_layout.addWidget(board_frame)
        
        # 棋盤佈局
        board_layout = QVBoxLayout(board_frame)
        board_layout.setSpacing(5)
        
        # 列按鈕佈局
        button_layout = QHBoxLayout()
        button_layout.setSpacing(5)
        
        # 創建列按鈕
        self.column_buttons = []
        for col in range(self.cols):
            btn = QPushButton(f"⬇️ {col}")
            btn.setFont(QFont("Arial", 12, QFont.Bold))
            btn.setFixedSize(80, 40)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #3498db;
                    color: white;
                    border: 2px solid #2980b9;
                }
                QPushButton:hover {
                    background-color: #5dade2;
                }
                QPushButton:pressed {
                    background-color: #2471a3;
                }
                QPushButton:disabled {
                    background-color: #7f8c8d;
                    border: 2px solid #5d6d7e;
                }
            """)
            btn.clicked.connect(lambda checked, c=col: self.human_move(c))
            button_layout.addWidget(btn)
            self.column_buttons.append(btn)
        
        board_layout.addLayout(button_layout)
        
        # 棋盤格子佈局
        grid_layout = QGridLayout()
        grid_layout.setSpacing(2)
        
        # 創建棋盤格子
        self.cells = []
        for row in range(self.rows):
            cell_row = []
            for col in range(self.cols):
                cell = QLabel("⚪")
                cell.setAlignment(Qt.AlignCenter)
                cell.setFont(QFont("Arial", 24, QFont.Bold))
                cell.setFixedSize(80, 80)
                cell.setStyleSheet("""
                    QLabel {
                        background-color: #ecf0f1;
                        border: 2px solid #bdc3c7;
                        border-radius: 5px;
                    }
                """)
                grid_layout.addWidget(cell, row, col)
                cell_row.append(cell)
            self.cells.append(cell_row)
        
        board_layout.addLayout(grid_layout)
        
        # 控制按鈕佈局
        control_layout = QHBoxLayout()
        control_layout.setSpacing(20)
        control_layout.setAlignment(Qt.AlignCenter)
        
        # 重新開始按鈕
        self.restart_button = QPushButton("🔄 重新開始")
        self.restart_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.restart_button.setFixedSize(120, 40)
        self.restart_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: 2px solid #c0392b;
            }
            QPushButton:hover {
                background-color: #ec7063;
            }
            QPushButton:pressed {
                background-color: #a93226;
            }
        """)
        self.restart_button.clicked.connect(self.restart_game)
        control_layout.addWidget(self.restart_button)
        
        # 退出按鈕
        self.quit_button = QPushButton("❌ 退出")
        self.quit_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.quit_button.setFixedSize(120, 40)
        self.quit_button.setStyleSheet("""
            QPushButton {
                background-color: #95a5a6;
                color: white;
                border: 2px solid #7f8c8d;
            }
            QPushButton:hover {
                background-color: #a6acaf;
            }
            QPushButton:pressed {
                background-color: #6c7b7f;
            }
        """)
        self.quit_button.clicked.connect(self.quit_game)
        control_layout.addWidget(self.quit_button)
        
        main_layout.addLayout(control_layout)
        
        # 遊戲說明
        info_label = QLabel("目標：連續四個棋子（水平、垂直或對角線）\n🔴 你是紅色  🟡 AI是黃色")
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setFont(QFont("Arial", 10))
        info_label.setStyleSheet("color: #bdc3c7; margin: 10px;")
        main_layout.addWidget(info_label)
        
        # 顯示歡迎消息
        QTimer.singleShot(100, self.show_welcome_message)

    def _on_starter_changed(self, checked: bool):
        if not checked:
            return
        # 根據選項更新狀態，並重開局以套用
        self.ai_starts = self.rb_ai_first.isChecked()
        self.restart_game()

    def show_welcome_message(self):
        """顯示歡迎消息"""
        msg = QMessageBox()
        msg.setWindowTitle("歡迎")
        msg.setText("🎮 歡迎來到 ConnectX！")
        msg.setInformativeText(
            "遊戲規則：\n"
            "• 目標：連續四個棋子（水平、垂直或對角線）\n"
            "• 🔴 你是紅色棋子\n"
            "• 🟡 AI是黃色棋子\n"
            "• 點擊列按鈕放置棋子\n"
            "• 人類先手\n\n"
            "祝你好運！"
        )
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()
        
    def _init_status_pulse(self):
        if not hasattr(self, 'status_label'):
            return
        effect = QGraphicsOpacityEffect(self.status_label)
        self.status_label.setGraphicsEffect(effect)
        self.status_pulse_anim = QPropertyAnimation(effect, b"opacity", self)
        self.status_pulse_anim.setDuration(1800)
        self.status_pulse_anim.setStartValue(0.35)
        self.status_pulse_anim.setEndValue(1.0)
        self.status_pulse_anim.setLoopCount(-1)
        self.status_pulse_anim.setEasingCurve(QEasingCurve.InOutQuad)
        self.status_pulse_anim.start()

    def update_board_display(self):
        """更新棋盤顯示 (加入最後一步與勝利高亮)"""
        base_empty = """QLabel {background-color:#ecf0f1; border:2px solid #bdc3c7; border-radius:6px;}"""
        style_p1 = """QLabel {background-color: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #ffb3b3, stop:1 #ff6f6f); border:2px solid #ff8f8f; border-radius:6px;}"""
        style_p2 = """QLabel {background-color: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #ffe9b3, stop:1 #ffc94d); border:2px solid #ffcf66; border-radius:6px;}"""
        style_last = "border:3px solid #ffffff; box-shadow:0 0 6px #fff;"
        style_flash_a = "border:3px solid #2ecc71; box-shadow:0 0 10px #2ecc71;"
        style_flash_b = "border:3px solid #27ae60; box-shadow:0 0 16px #27ae60;"
        flashing = set(self.winning_cells)
        for r in range(self.rows):
            for c in range(self.cols):
                cell = self.cells[r][c]
                v = self.board[r][c]
                if v == 0:
                    cell.setText("⚪")
                    cell.setStyleSheet(base_empty)
                elif v == 1:
                    cell.setText("🔴")
                    cell.setStyleSheet(style_p1)
                else:
                    cell.setText("🟡")
                    cell.setStyleSheet(style_p2)
                if self.last_move == (r,c):
                    # 疊加 last move 邊框
                    cell.setStyleSheet(cell.styleSheet() + style_last)
                if (r,c) in flashing:
                    cell.setStyleSheet(cell.styleSheet() + (style_flash_a if self.win_flash_state else style_flash_b))
    
    def _toggle_win_flash(self):
        self.win_flash_state = not self.win_flash_state
        self.update_board_display()
    
    def _start_win_flash(self):
        if self.win_flash_timer:
            self.win_flash_timer.stop(); self.win_flash_timer.deleteLater()
        self.win_flash_timer = QTimer(self)
        self.win_flash_timer.timeout.connect(self._toggle_win_flash)
        self.win_flash_timer.start(380)
    
    def _clear_winning_effects(self):
        for cell, eff in self.winning_effects:
            try:
                cell.setGraphicsEffect(None)
            except Exception:
                pass
        self.winning_effects.clear()
        if self.winning_line_timer:
            self.winning_line_timer.stop(); self.winning_line_timer.deleteLater(); self.winning_line_timer=None

    def _start_winning_line_animation(self):
        # 清掉舊的閃爍/特效
        if self.win_flash_timer:
            self.win_flash_timer.stop(); self.win_flash_timer.deleteLater(); self.win_flash_timer=None
        self._clear_winning_effects()
        if not self.winning_cells:
            return
        # 為每個勝利格套用 colorize effect
        self.winning_effects = []
        for (r,c) in self.winning_cells:
            cell = self.cells[r][c]
            eff = QGraphicsColorizeEffect(cell)
            eff.setColor(QColor("#2ecc71"))
            eff.setStrength(0.0)
            cell.setGraphicsEffect(eff)
            self.winning_effects.append((cell, eff))
        self.winning_line_anim_step = 0
        self.winning_line_timer = QTimer(self)
        self.winning_line_timer.timeout.connect(self._advance_winning_line_animation)
        self.winning_line_timer.start(80)

    def _advance_winning_line_animation(self):
        # 形成一個波動，依索引位移
        import math
        self.winning_line_anim_step += 1
        base_phase = self.winning_line_anim_step * 0.18
        n = len(self.winning_effects)
        for idx, (cell, eff) in enumerate(self.winning_effects):
            phase = base_phase - idx * 0.65
            strength = 0.55 + 0.45 * math.sin(phase)
            try:
                eff.setStrength(max(0.0, min(1.0, strength)))
            except Exception:
                pass
        # 同步更新邊框樣式（保留 last_move 判斷）
        self.update_board_display()
    
    def is_valid_move(self, col):
        """檢查移動是否有效"""
        return 0 <= col < self.cols and self.board[0][col] == 0
    
    def make_move(self, col, player):
        """在指定列放置棋子"""
        if not self.is_valid_move(col):
            return False
            
        for row in range(self.rows - 1, -1, -1):
            if self.board[row][col] == 0:
                self.board[row][col] = player
                return True
        return False
    
    def is_board_full(self):
        """檢查棋盤是否已滿"""
        return all(self.board[0][c] != 0 for c in range(self.cols))
    
    def _compute_winning_cells(self, player):
        lines = []
        # horizontal
        for r in range(self.rows):
            for c in range(self.cols-3):
                if all(self.board[r][c+i]==player for i in range(4)):
                    lines.append([(r,c+i) for i in range(4)])
        # vertical
        for r in range(self.rows-3):
            for c in range(self.cols):
                if all(self.board[r+i][c]==player for i in range(4)):
                    lines.append([(r+i,c) for i in range(4)])
        # diag ↘
        for r in range(self.rows-3):
            for c in range(self.cols-3):
                if all(self.board[r+i][c+i]==player for i in range(4)):
                    lines.append([(r+i,c+i) for i in range(4)])
        # diag ↙
        for r in range(self.rows-3):
            for c in range(3,self.cols):
                if all(self.board[r+i][c-i]==player for i in range(4)):
                    lines.append([(r+i,c-i) for i in range(4)])
        return lines[0] if lines else []

    def check_win(self, player):
        # 重寫：同時儲存勝利位置並啟動新動畫
        cells = self._compute_winning_cells(player)
        if cells:
            self.winning_cells = cells
            self._start_winning_line_animation()
            return True
        return False
    
    def human_move(self, col):
        """處理人類玩家移動（加入動畫）"""
        if self.game_over or self.ai_thinking or self.current_player != 1 or self.animating:
            return
            
        if not self.is_valid_move(col):
            msg = QMessageBox()
            msg.setWindowTitle("無效移動")
            msg.setText("該列已滿，請選擇其他列！")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return
        
        target_row = self.get_drop_row(col)
        if target_row < 0:
            return

        def after_animation():
            # 勝負檢查
            if self.check_win(1):
                self.game_over = True
                self.status_label.setText("🎉 恭喜！你贏了！")
                self.status_label.setStyleSheet("color: #27ae60; font-weight: bold;")
                self.disable_buttons()
                
                msg = QMessageBox()
                msg.setWindowTitle("遊戲結束")
                msg.setText("🎉 恭喜！你贏了！")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
                return
            
            # 檢查平局
            if self.is_board_full():
                self.game_over = True
                self.status_label.setText("🤝 平局！")
                self.status_label.setStyleSheet("color: #f39c12; font-weight: bold;")
                self.disable_buttons()
                
                msg = QMessageBox()
                msg.setWindowTitle("遊戲結束")
                msg.setText("🤝 平局！棋盤已滿。")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
                return
            
            # 交給 AI
            self.current_player = 2
            self.enable_buttons()  # 先解除，ai_turn 會再關閉
            self.ai_turn()

        # 啟動動畫
        self.animate_drop(col, target_row, 1, after_animation)
    
    def get_drop_row(self, col: int) -> int:
        """回傳該列可以落子的最底 row，若無則 -1"""
        if not (0 <= col < self.cols):
            return -1
        for r in range(self.rows - 1, -1, -1):
            if self.board[r][col] == 0:
                return r
        return -1

    def animate_drop(self, col: int, target_row: int, player: int, finished_cb):
        """以動畫方式將棋子由頂部落到 target_row。
        finished_cb: 動畫完成後呼叫 (會在最後真正寫入 board 並刷新 / 呼叫後續邏輯)
        """
        # 改良：模擬重力加速 (使用動態間隔)
        if self.animating:
            return
        self.animating = True
        self.disable_buttons()
        token = "🔴" if player == 1 else "🟡"
        path_rows = list(range(0, target_row + 1))
        current_index = {"i": 0}
        base_interval = self.animation_interval_ms
        def step():
            i = current_index["i"]
            if i > 0:
                pr = path_rows[i-1]
                if self.board[pr][col] == 0:
                    self.cells[pr][col].setText("⚪")
            cr = path_rows[i]
            self.cells[cr][col].setText(token)
            current_index["i"] += 1
            if current_index["i"] >= len(path_rows):
                if self.board[target_row][col] == 0:
                    self.board[target_row][col] = player
                self.last_move = (target_row, col)
                self.update_board_display()
                self.animating = False
                if self.animation_timer:
                    self.animation_timer.stop(); self.animation_timer.deleteLater(); self.animation_timer=None
                try:
                    if finished_cb:
                        finished_cb()
                except Exception as e:
                    print(f"Finished callback error: {e}")
                return
            # 動態調整速度 (後段加速)
            if self.animation_timer:
                speed_factor = 0.55 + 0.45 * (current_index["i"] / len(path_rows))
                self.animation_timer.setInterval(int(base_interval * (1.0 / speed_factor)))
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(step)
        self.animation_timer.start(base_interval)
        step()
    
    def ai_turn(self):
        """AI回合"""
        if self.game_over:
            return
            
        self.ai_thinking = True
        self.status_label.setText("🟡 AI思考中...")
        self.status_label.setStyleSheet("color: #f39c12; font-weight: bold;")
        self.disable_buttons()
        
        # 創建AI線程，傳遞環境實例和當前玩家
        self.ai_thread = AIThread(self.env, self.board.copy(), self.rows, self.cols, self.current_player)  # 新增：傳遞 self.current_player
        self.ai_thread.move_calculated.connect(self.execute_ai_move)
        self.ai_thread.start()


    def execute_ai_move(self, move):
        """執行AI移動（加入動畫）"""
        self.ai_thinking = False
        
        if move == -1 or self.game_over or self.animating:
            self.status_label.setText("❌ AI無法移動")
            self.status_label.setStyleSheet("color: #e74c3c; font-weight: bold;")
            return
        
        if not self.is_valid_move(move):
            # 找備選
            for c in range(self.cols):
                if self.is_valid_move(c):
                    move = c; break
            else:
                self.status_label.setText("❌ AI無法移動")
                return
        target_row = self.get_drop_row(move)
        if target_row < 0:
            self.status_label.setText("❌ AI無法移動")
            return

        def after_animation():
            if self.check_win(2):
                self.game_over = True
                self.status_label.setText("🤖 AI獲勝！")
                self.status_label.setStyleSheet("color: #e74c3c; font-weight: bold;")
                self.disable_buttons()
                
                msg = QMessageBox()
                msg.setWindowTitle("遊戲結束")
                msg.setText("🤖 AI獲勝！再接再厲！")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
                return
            
            # 檢查平局
            if self.is_board_full():
                self.game_over = True
                self.status_label.setText("🤝 平局！")
                self.status_label.setStyleSheet("color: #f39c12; font-weight: bold;")
                self.disable_buttons()
                
                msg = QMessageBox()
                msg.setWindowTitle("遊戲結束")
                msg.setText("🤝 平局！棋盤已滿。")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
                return
            
            self.current_player = 1
            self.status_label.setText("🔴 你的回合！")
            self.status_label.setStyleSheet("color: white; font-weight: bold;")
            self.enable_buttons()

        self.animate_drop(move, target_row, 2, after_animation)
    
    def disable_buttons(self):
        """禁用列按鈕"""
        for btn in self.column_buttons:
            btn.setEnabled(False)
    
    def enable_buttons(self):
        """啟用列按鈕"""
        if not self.game_over and not self.ai_thinking:
            for btn in self.column_buttons:
                btn.setEnabled(True)
    
    def restart_game(self):
        """重新開始遊戲"""
        # 停止AI線程
        if self.ai_thread and self.ai_thread.isRunning():
            self.ai_thread.terminate()
            self.ai_thread.wait()
        
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.game_over = False
        self.ai_thinking = False
        self.animating = False
        self.winning_cells = []
        if self.win_flash_timer:
            self.win_flash_timer.stop(); self.win_flash_timer.deleteLater(); self.win_flash_timer=None
        self.last_move = None
        self._clear_winning_effects()
        self.update_board_display()
        
        # 根據先手選擇設定當前玩家與狀態
        if self.ai_starts:
            self.current_player = 2  # AI 行動
            self.status_label.setText("🟡 AI思考中...")
            self.status_label.setStyleSheet("color: #f39c12; font-weight: bold;")
            self.disable_buttons()
            QTimer.singleShot(300, self.ai_turn)
        else:
            self.current_player = 1  # 人類行動
            self.status_label.setText("🔴 你的回合！")
            self.status_label.setStyleSheet("color: white; font-weight: bold;")
            self.enable_buttons()
    
    def quit_game(self):
        """退出遊戲"""
        reply = QMessageBox.question(
            self, 
            "退出", 
            "確定要退出遊戲嗎？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # 停止AI線程
            if self.ai_thread and self.ai_thread.isRunning():
                self.ai_thread.terminate()
                self.ai_thread.wait()
            self.close()
    
    def closeEvent(self, event):
        """窗口關閉事件"""
        # 停止AI線程
        if self.ai_thread and self.ai_thread.isRunning():
            self.ai_thread.terminate()
            self.ai_thread.wait()
        event.accept()

def main():
    """主函數"""
    app = QApplication(sys.argv)
    
    # 設置應用程序樣式
    app.setStyle('Fusion')
    
    # 設置深色主題
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(44, 62, 80))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(52, 73, 94))
    palette.setColor(QPalette.AlternateBase, QColor(66, 84, 103))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(52, 73, 94))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)
    
    try:
        game = ConnectXGUI()
        game.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"遊戲出現錯誤: {e}")
        msg = QMessageBox()
        msg.setWindowTitle("錯誤")
        msg.setText(f"遊戲出現錯誤: {e}\n請確認 ConnectFourEnv 和 submission.py 正確。")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

if __name__ == "__main__":
    main()