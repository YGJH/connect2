def agent(obs, config):
    from random import choice
    import time
    import numpy as np  # 添加這行
    columns = config['columns'] if isinstance(config, dict) else config.columns
    rows = config['rows'] if isinstance(config, dict) else config.rows
    size = rows * columns
    max_depth = 15  # 降低深度避免超時
    EMPTY = 0

    # 改善的轉置表 - 使用全域變數保持狀態
    if not hasattr(agent, 'transposition_table'):
        agent.transposition_table = {}
        agent.call_count = 0
        # print("🆕 首次載入agent，初始化轉置表")
    
    agent.call_count += 1
    # print(f"📞 Agent呼叫次數: {agent.call_count}")
    
    transposition_table = agent.transposition_table
    
    # 調試信息
    search_stats = {'nodes_searched': 0, 'max_depth_reached': 0}

    def flatten_nested(item):
        flat = []
        if isinstance(item, (list, tuple, np.ndarray)):
            for sub in item:
                flat.extend(flatten_nested(sub))
        else:
            flat.append(int(item))  # 強制轉為 python int
        return flat
    def board_hash(board):
        """安全的棋盤雜湊函數"""
        if hasattr(board, 'tolist'):  # numpy 陣列
            board_tuple = tuple(board.flatten().tolist())
        elif isinstance(board, (list, tuple)):
            board_tuple = tuple(board)
        else:
            board_tuple = tuple(list(board))
        return hash(board_tuple)
    
    def safe_board_copy(board):
        """安全的棋盤複製，確保返回 list"""
        if hasattr(board, 'tolist'):  # numpy 陣列
            return board.flatten().tolist()
        elif isinstance(board, list):
            return board[:]
        else:
            return list(board)

    def is_win(board, col, piece, config, has_played):
        """檢查是否獲勝（修正版）"""
        columns = config['columns'] if isinstance(config, dict) else config.columns
        rows = config['rows'] if isinstance(config, dict) else config.rows
        inarow = config['inarow'] if isinstance(config, dict) else config.inarow

        if not has_played:
            # 如果尚未下棋，先模擬下棋
            temp_board = board[:]
            row = get_next_open_row(temp_board, col)
            if row is None:
                return False
            drop_piece(temp_board, row, col, piece)
            board = temp_board

        # 檢查以該位置為起點的所有方向
        def check_direction(start_row, start_col, delta_row, delta_col):
            count = 0
            r, c = start_row, start_col

            # 向一個方向檢查
            while 0 <= r < rows and 0 <= c < columns and board[r * columns + c] == piece:
                count += 1
                r += delta_row
                c += delta_col

            # 向反方向檢查
            r, c = start_row - delta_row, start_col - delta_col
            while 0 <= r < rows and 0 <= c < columns and board[r * columns + c] == piece:
                count += 1
                r -= delta_row
                c -= delta_col

            return count >= inarow

        # 找到最後下棋的位置（UI格式：第0行是頂部）
        last_row = None
        for r in range(rows-1, -1, -1):  # 從底部開始尋找
            if board[r * columns + col] == piece:
                last_row = r
                break

        if last_row is None:
            return False

        # 檢查四個方向：水平、垂直、正對角線、反對角線
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for delta_row, delta_col in directions:
            if check_direction(last_row, col, delta_row, delta_col):
                return True

        return False

    def order_moves(board, mark):
        """智能移動排序 - 修正版"""
        moves = []
        center_col = columns // 2
        opp_mark = 1 if mark == 2 else 2

        for col in range(columns):
            if is_valid_location(board, col):
                score = 0
                
                # 檢查獲勝移動 - 最高優先級
                if check_winning_move(board, col, mark):
                    score += 1000000  # 非常高的分數
                
                # 檢查阻擋對手獲勝 - 次高優先級
                elif check_winning_move(board, col, opp_mark):
                    score += 100000   # 很高的分數
                
                # 中央優先 - 較低優先級
                score -= abs(col - center_col) * 10

                moves.append((score, col))

        moves.sort(reverse=True)
        return [col for score, col in moves]

    def negamax(board, mark, depth, alpha=-float('inf'), beta=float('inf'), is_maximizing=True):
        search_stats['nodes_searched'] += 1
        search_stats['max_depth_reached'] = max(search_stats['max_depth_reached'], max_depth - depth)
        
        # 確保 board 是一維列表
        board = safe_board_copy(board)
        
        moves = sum(1 if cell != EMPTY else 0 for cell in board)
        opp_mark = 1 if mark == 2 else 2

        # 檢查轉置表
        board_key = (board_hash(board), mark, depth)
        if board_key in transposition_table:
            stored_score, stored_move, stored_depth = transposition_table[board_key]
            if stored_depth >= depth:
                return (stored_score, stored_move)

        # 終止條件：平局
        if moves == size:
            result = (0, None)
            transposition_table[board_key] = (result[0], result[1], depth)
            return result

        # 檢查立即獲勝
        for column in range(columns):
            if is_valid_location(board, column) and check_winning_move(board, column, mark):
                score = float('inf')
                result = (score, column)
                transposition_table[board_key] = (result[0], result[1], depth)
                return result

        # 檢查對手威脅
        threat_columns = [c for c in range(columns) 
                        if is_valid_location(board, c) and check_winning_move(board, c, opp_mark)]
        
        if len(threat_columns) == 1:
            threat_col = threat_columns[0]
            temp_board = safe_board_copy(board)
            row = get_next_open_row(temp_board, threat_col)
            drop_piece(temp_board, row, threat_col, mark)
            
            (score, _) = negamax(temp_board, opp_mark, depth - 1, -beta, -alpha, False)
            score = -score
            
            result = (score, threat_col)
            transposition_table[board_key] = (result[0], result[1], depth)
            return result
        elif len(threat_columns) > 1:
            result = (-float('inf'), None)
            transposition_table[board_key] = (result[0], result[1], depth)
            return result

        # 如果達到最大深度，使用評估函數
        if depth <= 0:
            best_score = -float('inf')
            best_column = None
            ordered_columns = order_moves(board, mark)

            for column in ordered_columns:
                temp_board = safe_board_copy(board)
                row = get_next_open_row(temp_board, column)
                if row is not None:
                    drop_piece(temp_board, row, column, mark)
                    score = evaluate_position(temp_board, mark) - evaluate_position(temp_board, opp_mark)

                    if score > best_score:
                        best_score = score
                        best_column = column

            result = (best_score, best_column)
            transposition_table[board_key] = (result[0], result[1], depth)
            return result

        # 遞歸搜尋 - 修正：移動 return 到正確位置
        best_score = -float('inf')
        best_column = None
        ordered_columns = order_moves(board, mark)

        for column in ordered_columns:
            temp_board = safe_board_copy(board)
            row = get_next_open_row(temp_board, column)
            if row is not None:
                drop_piece(temp_board, row, column, mark)

                (score, _) = negamax(temp_board, opp_mark, depth - 1, -beta, -alpha, False)
                score = -score

                if score == float('inf'):
                    best_score = score
                    best_column = column
                    break
                elif score == -float('inf'):
                    continue

                if score > best_score:
                    best_score = score
                    best_column = column

                alpha = max(alpha, score)
                if beta <= alpha:
                    break

        # 修正：將 return 移到循環外部
        result = (best_score, best_column)
        transposition_table[board_key] = (result[0], result[1], depth)
        return result

    def iterative_deepening_search_with_timeout(board, mark, max_depth, timeout=1.8):
        """帶時間限制的迭代深化搜尋 - 修正版"""
        start_time = time.time()
        best_move = None
        best_score = -float('inf')
        
        # 重置搜尋統計
        search_stats['nodes_searched'] = 0
        search_stats['max_depth_reached'] = 0

        for depth in range(1, max_depth + 1):
            current_time = time.time()
            if current_time - start_time > timeout:
                print(f"時間限制達到，在深度 {depth-1} 停止搜尋")
                break

            try:
                score, move = negamax(board, mark, depth)
                if move is not None:
                    best_move = move
                    best_score = score
                    # print(f"深度 {depth}: 最佳移動 = {move}, 分數 = {score:.2f}")
                    
                    # 如果找到確定獲勝的移動，可以提前結束
                    if score == float('inf') or score > 10000:
                        # print(f"找到獲勝移動，提前結束搜尋")
                        break
                        
            except Exception as e:
                print(f"深度 {depth} 搜尋出錯: {e}")
                break

        # print(f"搜尋統計: 節點數 = {search_stats['nodes_searched']}, 最大深度 = {search_stats['max_depth_reached']}")
        return best_move



    def evaluate_position(board, mark):
        """重新設計的位置評估函數 - 強化威脅檢測"""
        score = 0
        opp_mark = 1 if mark == 2 else 2

        # 1. 檢查立即威脅和機會 - 最高優先級
        my_threats = 0
        opp_threats = 0
        
        for col in range(columns):
            if is_valid_location(board, col):
                # 我方立即獲勝機會
                if check_winning_move(board, col, mark):
                    my_threats += 1
                    score += 100000  # 極高分數
                
                # 對手立即威脅
                if check_winning_move(board, col, opp_mark):
                    opp_threats += 1
                    score -= 100000  # 極低分數

        # 2. 檢查潛在威脅（兩步內的威脅）
        potential_threats = evaluate_potential_threats(board, mark, opp_mark)
        score += potential_threats

        # 3. 評估棋型強度
        pattern_score = evaluate_patterns_enhanced(board, mark)
        score += pattern_score

        # 4. 中央控制 - 最低權重
        center_column = columns // 2
        center_count = sum(1 for r in range(rows) 
                          if board[center_column + r * columns] == mark)
        score += center_count * 2

        return score


    def evaluate_potential_threats(board, mark, opp_mark):
        """評估潛在威脅（需要兩步或更多步驟的威脅）"""
        score = 0
        
        # 檢查所有可能的四連線位置
        for direction in [(0, 1), (1, 0), (1, 1), (1, -1)]:  # 水平、垂直、對角線
            score += check_line_threats(board, mark, opp_mark, direction)
        
        return score

    def check_line_threats(board, mark, opp_mark, direction):
        """檢查特定方向的線性威脅"""
        score = 0
        dr, dc = direction
        
        # 根據方向設定檢查範圍
        if dr == 0:  # 水平
            start_positions = [(r, 0) for r in range(rows)]
            max_length = columns - 3
        elif dc == 0:  # 垂直
            start_positions = [(0, c) for c in range(columns)]
            max_length = rows - 3
        elif dr == 1 and dc == 1:  # 正對角線
            start_positions = [(r, 0) for r in range(rows - 3)] + [(0, c) for c in range(1, columns - 3)]
            max_length = min(rows, columns)
        else:  # 反對角線 (dr == 1, dc == -1)
            start_positions = [(r, columns - 1) for r in range(rows - 3)] + [(0, c) for c in range(3, columns - 1)]
            max_length = min(rows, columns)
        
        for start_r, start_c in start_positions:
            line_length = 0
            r, c = start_r, start_c
            
            # 計算這條線的最大長度
            while 0 <= r < rows and 0 <= c < columns:
                line_length += 1
                r += dr
                c += dc
            
            if line_length >= 4:  # 只檢查長度足夠的線
                score += evaluate_line_segment(board, mark, opp_mark, start_r, start_c, dr, dc, line_length)
        
        return score

    def evaluate_line_segment(board, mark, opp_mark, start_r, start_c, dr, dc, length):
        """評估一條線段的威脅值"""
        score = 0
        
        # 檢查所有可能的4格窗口
        for i in range(length - 3):
            window = []
            empty_positions = []
            
            for j in range(4):
                r = start_r + (i + j) * dr
                c = start_c + (i + j) * dc
                if 0 <= r < rows and 0 <= c < columns:
                    cell_value = board[r * columns + c]
                    window.append(cell_value)
                    if cell_value == 0:  # EMPTY
                        empty_positions.append((r, c))
            
            if len(window) == 4:
                # 評估這個窗口的威脅值
                my_count = window.count(mark)
                opp_count = window.count(opp_mark)
                empty_count = window.count(0)
                
                # 只有在沒有對手棋子阻擋時才是有效威脅
                if opp_count == 0:
                    if my_count == 3 and empty_count == 1:
                        # 檢查空格是否可以放置（重力規則）
                        empty_r, empty_c = empty_positions[0]
                        if can_place_piece(board, empty_r, empty_c):
                            score += 5000  # 強威脅
                    elif my_count == 2 and empty_count == 2:
                        # 檢查兩個空格是否都可以放置
                        valid_positions = sum(1 for er, ec in empty_positions if can_place_piece(board, er, ec))
                        if valid_positions >= 1:
                            score += 500  # 中等威脅
                
                # 檢查對手威脅
                if my_count == 0:
                    if opp_count == 3 and empty_count == 1:
                        empty_r, empty_c = empty_positions[0]
                        if can_place_piece(board, empty_r, empty_c):
                            score -= 8000  # 對手強威脅
                    elif opp_count == 2 and empty_count == 2:
                        valid_positions = sum(1 for er, ec in empty_positions if can_place_piece(board, er, ec))
                        if valid_positions >= 1:
                            score -= 800  # 對手中等威脅
        
        return score

    def can_place_piece(board, target_r, target_c):
        """檢查是否可以在指定位置放置棋子（考慮重力）- UI格式：第0行是頂部"""
        # 在UI格式中，第5行是底部，第0行是頂部
        if target_r == 5:  # 底部行（第5行）
            return True
        
        # 檢查下方是否有棋子支撐
        below_r = target_r + 1  # 下方行在UI格式中是+1
        below_pos = below_r * columns + target_c
        return board[below_pos] != 0  # 下方不是空的

    def evaluate_patterns_enhanced(board, mark):
        """增強的棋型評估"""
        score = 0
        opp_mark = 1 if mark == 2 else 2

        # 使用原來的窗口評估
        score += evaluate_window(board, mark, columns, rows)
        
        # 額外的棋型評估
        score += evaluate_special_patterns(board, mark, opp_mark)
        
        return score

    def evaluate_special_patterns(board, mark, opp_mark):
        """評估特殊棋型"""
        score = 0
        
        # 檢查fork（分叉）威脅 - 同時有多個威脅線
        fork_threats = count_fork_threats(board, mark)
        score += fork_threats * 2000
        
        # 檢查對手的fork威脅
        opp_fork_threats = count_fork_threats(board, opp_mark)
        score -= opp_fork_threats * 3000
        
        return score

    def count_fork_threats(board, mark):
        """計算分叉威脅數量"""
        threats = 0
        
        # 檢查每個可能的下棋位置
        for col in range(columns):
            if is_valid_location(board, col):
                # 模擬在這個位置下棋
                temp_board = safe_board_copy(board)  # 使用安全複製
                row = get_next_open_row(temp_board, col)
                if row is not None:
                    drop_piece(temp_board, row, col, mark)
                    
                    # 檢查下棋後是否創造了多個威脅
                    threat_count = 0
                    for test_col in range(columns):
                        if is_valid_location(temp_board, test_col) and check_winning_move(temp_board, test_col, mark):
                            threat_count += 1
                    
                    if threat_count >= 2:
                        threats += 1
        
        return threats

    def score_window(window, mark):
            """評估 4 格窗口的分數 - 加強版"""
            score = 0
            opp_mark = 1 if mark == 2 else 2

            if window.count(mark) == 4:
                score += 1000  # 四連線
            elif window.count(mark) == 3 and window.count(EMPTY) == 1:
                score += 50   # 三連線有一空格
            elif window.count(mark) == 2 and window.count(EMPTY) == 2:
                score += 10   # 二連線有兩空格
            elif window.count(mark) == 1 and window.count(EMPTY) == 3:
                score += 1    # 一連線有三空格

            # 對手的威脅
            if window.count(opp_mark) == 4:
                score -= 1000  # 對手四連線
            elif window.count(opp_mark) == 3 and window.count(EMPTY) == 1:
                score -= 100   # 對手三連線威脅
            elif window.count(opp_mark) == 2 and window.count(EMPTY) == 2:
                score -= 5     # 對手二連線

            return score

    def evaluate_window(board, mark, columns, rows):
        """評估所有可能的窗口"""
        score = 0

        # 水平檢查
        for r in range(rows):
            for c in range(columns - 3):
                window = [board[r * columns + c + i] for i in range(4)]
                score += score_window(window, mark)

        # 垂直檢查
        for c in range(columns):
            for r in range(rows - 3):
                window = [board[(r + i) * columns + c] for i in range(4)]
                score += score_window(window, mark)

        # 正對角線檢查
        for r in range(rows - 3):
            for c in range(columns - 3):
                window = [board[(r + i) * columns + c + i] for i in range(4)]
                score += score_window(window, mark)

        # 反對角線檢查
        for r in range(rows - 3):
            for c in range(columns - 3):
                window = [board[(r + 3 - i) * columns + c + i] for i in range(4)]
                score += score_window(window, mark)

        return score

    def evaluate_window(board, mark, columns, rows):
        """評估所有可能的窗口"""
        score = 0

        # 水平檢查
        for r in range(rows):
            for c in range(columns - 3):
                window = [board[r * columns + c + i] for i in range(4)]
                score += score_window(window, mark)

        # 垂直檢查
        for c in range(columns):
            for r in range(rows - 3):
                window = [board[(r + i) * columns + c] for i in range(4)]
                score += score_window(window, mark)

        # 正對角線檢查
        for r in range(rows - 3):
            for c in range(columns - 3):
                window = [board[(r + i) * columns + c + i] for i in range(4)]
                score += score_window(window, mark)

        # 反對角線檢查
        for r in range(rows - 3):
            for c in range(columns - 3):
                window = [board[(r + 3 - i) * columns + c + i] for i in range(4)]
                score += score_window(window, mark)

        return score

    def is_valid_location(board, col):
        """檢查該列是否可以下棋 - 檢查頂部行是否為空（UI格式：第0行是頂部）"""
        # 在UI格式中，第0行是頂部，索引是 0*columns + col
        return board[0 * columns + col] == EMPTY

    def get_next_open_row(board, col):
        """取得該列的下一個可下位置（UI格式：第0行是頂部）"""
        for r in range(rows-1, -1, -1):  # 從底部開始尋找
            if board[r * columns + col] == EMPTY:
                return r
        return None

    def drop_piece(board, row, col, piece):
        """在指定位置放置棋子（UI格式：第0行是頂部）"""
        board[row * columns + col] = piece

    def check_winning_move(board, col, piece):
        """檢查在該列下棋是否能獲勝"""
        if not is_valid_location(board, col):
            return False

        temp_board = safe_board_copy(board)  # 使用安全複製
        row = get_next_open_row(temp_board, col)
        if row is None:
            return False

        drop_piece(temp_board, row, col, piece)
        return is_win(temp_board, col, piece, {'columns': columns, 'rows': rows, 'inarow': 4}, True)
    
    
    if isinstance(obs, dict):
        board = obs['board']
        mark = obs['mark']
        if hasattr(mark, '__getitem__'):  # 如果是陣列
            mark = mark[0]
    else:
        board = obs.board
        mark = obs.mark
    
    # 確保 board 是 list 類型
    board = safe_board_copy(board)
    
    # print(f"Board type: {type(board)}")
    # print(f"Mark: {mark}")
    
    # 使用迭代深化搜尋
    column = iterative_deepening_search_with_timeout(board, mark, max_depth, timeout=1.8)
    if column is None:
        available_columns = [c for c in range(columns) if board[c] == EMPTY]
        if available_columns:
            column = min(available_columns, key=lambda x: abs(x - columns//2))
        else:
            column = choice(range(columns))
    return column