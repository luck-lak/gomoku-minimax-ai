from agent import Agent
import numpy as np


class Search(Agent):
    """使用 Alpha-Beta 剪枝和棋型估值选择落点的五子棋 Agent。"""

    def __init__(self, player):
        """初始化搜索参数、棋型奖励和技能状态。"""
        super().__init__(player)

        # 搜索参数：depth 越大看得越远，但耗时会快速增长。
        self.depth = 3
        self.k = 1

        # 每次搜索都会更新的根节点状态。
        self.move_result = None
        self.skill_target = None
        self.root_max = -float("inf")
        self.skill_used = False

        # 棋型奖励，从五子连珠到单边一子逐级递减。
        self.score_already_win = 10_000_000_000
        self.score_ex_hopeful = 1_000_000_000
        self.score_hopeful_plus = 100_000_000
        self.score_hopeful = 10_000_000
        self.score_good = 100_000
        self.score_encourage = 1_000
        self.score_slight_encourage = 100
        self.score_minor = 10

    def get_children(self, board, is_max_player):
        """生成已有棋子周围一格内的所有候选棋盘。"""
        current_player = self.player if is_max_player else 3 - self.player
        size = len(board)
        neighbor_offsets = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

        valid_moves = set()
        for row in range(size):
            for col in range(size):
                if board[row][col] == 0:
                    continue
                for row_offset, col_offset in neighbor_offsets:
                    next_row = row + row_offset
                    next_col = col + col_offset
                    if (
                        0 <= next_row < size
                        and 0 <= next_col < size
                        and board[next_row][next_col] == 0
                    ):
                        valid_moves.add((next_row, next_col))

        children = []
        for row, col in valid_moves:
            child = board.copy()
            child[row][col] = current_player
            children.append((child, (row, col)))
        return children

    def _is_open_end(self, board, row, col, player):
        """判断序列端点是否可用；己方技能标记沿用原实现的开放端语义。"""
        size = len(board)
        return (
            0 <= row < size
            and 0 <= col < size
            and board[row][col] in (0, player + 2)
        )

    def _score_sequence(self, count, open_ends):
        """根据连续棋子数和开放端数量返回棋型分数。"""
        if count >= 5:
            return self.score_already_win
        if count == 4 and open_ends == 2:
            return self.score_ex_hopeful
        if count == 3 and open_ends == 2:
            return self.score_hopeful
        if count == 4 and open_ends == 1:
            return self.score_hopeful_plus
        if count == 3 and open_ends == 1:
            return self.score_good
        if count == 2 and open_ends == 2:
            return self.score_encourage
        if (count == 2 and open_ends == 1) or (count == 1 and open_ends == 2):
            return self.score_slight_encourage
        if count == 1 and open_ends == 1:
            return self.score_minor
        return 0

    def _check_direction(self, board, row, col, player, row_step, col_step):
        """从一个连续序列的起点出发，统计指定方向的棋型。"""
        open_ends = int(
            self._is_open_end(board, row - row_step, col - col_step, player)
        )
        count = 0
        size = len(board)
        current_row, current_col = row, col

        while (
            0 <= current_row < size
            and 0 <= current_col < size
            and board[current_row][current_col] == player
        ):
            count += 1
            current_row += row_step
            current_col += col_step

        open_ends += int(
            self._is_open_end(board, current_row, current_col, player)
        )
        return self._score_sequence(count, open_ends)

    def check_vertical(self, board, pos_x, pos_y, can_play_player):
        """计算从给定起点向下延伸的竖直棋型分数。"""
        return self._check_direction(board, pos_x, pos_y, can_play_player, 1, 0)

    def check_level(self, board, pos_x, pos_y, can_play_player):
        """计算从给定起点向右延伸的水平棋型分数。"""
        return self._check_direction(board, pos_x, pos_y, can_play_player, 0, 1)

    def check_LU_to_RD(self, board, pos_x, pos_y, can_play_player):
        """计算从左上向右下延伸的对角棋型分数。"""
        return self._check_direction(board, pos_x, pos_y, can_play_player, 1, 1)

    def check_LD_to_RU(self, board, pos_x, pos_y, can_play_player):
        """计算从左下向右上延伸的对角棋型分数。"""
        return self._check_direction(board, pos_x, pos_y, can_play_player, -1, 1)

    def _board_score(self, board, player):
        """扫描一名玩家的所有连续序列并汇总棋型分数。"""
        score = 0
        size = len(board)

        for row, col in np.argwhere(board == player):
            if row == 0 or board[row - 1][col] != player:
                score += self.check_vertical(board, row, col, player)
            if col == 0 or board[row][col - 1] != player:
                score += self.check_level(board, row, col, player)
            if row == 0 or col == 0 or board[row - 1][col - 1] != player:
                score += self.check_LU_to_RD(board, row, col, player)
            if (
                row == size - 1
                or col == 0
                or board[row + 1][col - 1] != player
            ):
                score += self.check_LD_to_RU(board, row, col, player)

        return score

    def evaluate(self, board, can_play_player, pos):
        """从最大化玩家视角评价叶子棋盘。"""
        sign = 1 if can_play_player == self.player else -1

        # 最近一步已经连五时，直接返回终局分数，不再扫描整张棋盘。
        if pos is not None:
            from gomoku import check_win

            pos_x = pos[0][0]
            pos_y = pos[1][0]
            if check_win(board, pos_x, pos_y):
                winner = board[pos_x][pos_y]
                return (
                    self.score_already_win
                    if winner == self.player
                    else -self.score_already_win
                )

        current_score = self._board_score(board, can_play_player)
        opponent_score = self._board_score(board, 3 - can_play_player)

        # 活四优先于其他累计棋型，必须立即进攻或防守。
        if current_score >= self.score_ex_hopeful:
            return sign * self.score_ex_hopeful
        if opponent_score >= self.score_ex_hopeful:
            return -sign * self.score_ex_hopeful

        # 假设对手仍可用技能反制单边冲四，因此行动方的活三优先。
        if current_score >= self.score_hopeful:
            return sign * self.score_hopeful
        if opponent_score >= self.score_hopeful_plus:
            return -sign * self.score_hopeful_plus

        return sign * (current_score - opponent_score * self.k)

    def is_game_over(self, board, pos):
        """只围绕最近落点检查胜负，同时判断棋盘是否已满。"""
        board_full = np.all(
            (board == self.player) | (board == 3 - self.player)
        )
        if pos is None:
            return board_full

        from gomoku import check_win

        pos_x = pos[0][0]
        pos_y = pos[1][0]
        return check_win(board, pos_x, pos_y) or board_full

    def _find_skill_target_in_direction(
        self, board, move_result, row_step, col_step
    ):
        """检查假设落子后，指定方向是否形成可封锁端点的连续四子。"""
        pos_x, pos_y = move_result
        size = len(board)
        count = 1
        target = None

        row, col = pos_x + row_step, pos_y + col_step
        while (
            0 <= row < size
            and 0 <= col < size
            and board[row][col] == self.player
        ):
            count += 1
            row += row_step
            col += col_step
        if 0 <= row < size and 0 <= col < size and board[row][col] == 0:
            target = (row, col)

        row, col = pos_x - row_step, pos_y - col_step
        while (
            0 <= row < size
            and 0 <= col < size
            and board[row][col] == self.player
        ):
            count += 1
            row -= row_step
            col -= col_step
        if 0 <= row < size and 0 <= col < size and board[row][col] == 0:
            target = (row, col)

        return target if count == 4 else None

    def check_for_skill(self, board, move_result):
        """按水平、竖直和两个对角方向寻找技能封锁位置。"""
        directions = [(0, 1), (1, 0), (1, 1), (-1, 1)]
        for row_step, col_step in directions:
            target = self._find_skill_target_in_direction(
                board, move_result, row_step, col_step
            )
            if target is not None:
                self.skill_target = target
                self.skill_used = True
                return

    def alpha_beta(self, board, pos, n_depth, alpha, beta, is_max_player):
        """递归执行 Minimax 搜索，并使用 Alpha-Beta 边界剪枝。"""
        if n_depth == 0:
            next_player = self.player if is_max_player else 3 - self.player
            return self.evaluate(board, next_player, pos)

        if self.is_game_over(board, pos):
            pos_x = pos[0][0]
            pos_y = pos[1][0]

            # 根节点的某个候选落点直接获胜时，需要同步保存该落点。
            if n_depth == self.depth - 1:
                self.move_result = (pos_x, pos_y)

            winner = board[pos_x][pos_y]
            return (
                self.score_already_win
                if winner == self.player
                else -self.score_already_win
            )

        if is_max_player:
            max_score = -float("inf")
            for child, move in self.get_children(board, True):
                child_pos = (
                    np.array([move[0]]),
                    np.array([move[1]]),
                )
                score = self.alpha_beta(
                    child, child_pos, n_depth - 1, alpha, beta, False
                )

                # 深度为 1 时，根节点直接在最大化层选择落点。
                if self.depth == 1 and score > max_score:
                    self.move_result = move

                max_score = max(max_score, score)
                if n_depth == self.depth:
                    self.root_max = max(self.root_max, max_score)

                alpha = max(alpha, max_score)
                if alpha >= beta:
                    break
            return max_score

        min_score = float("inf")

        # 根节点下一层需要在完成对手应对搜索后，记录对应的首步落点。
        if n_depth == self.depth - 1:
            for child, move in self.get_children(board, False):
                child_pos = (
                    np.array([move[0]]),
                    np.array([move[1]]),
                )
                score = self.alpha_beta(
                    child, child_pos, n_depth - 1, alpha, beta, True
                )
                min_score = min(min_score, score)
                beta = min(beta, min_score)
                if beta <= alpha:
                    break

            if min_score > self.root_max:
                self.root_max = min_score
                self.move_result = (pos[0][0], pos[1][0])
            return min_score

        for child, move in self.get_children(board, False):
            child_pos = (
                np.array([move[0]]),
                np.array([move[1]]),
            )
            score = self.alpha_beta(
                child, child_pos, n_depth - 1, alpha, beta, True
            )
            min_score = min(min_score, score)
            beta = min(beta, min_score)
            if beta <= alpha:
                break
        return min_score

    def make_move(self, board):
        """搜索最佳普通落点，并在需要时附带一次技能目标。"""
        self.move_result = None
        self.skill_target = None
        self.root_max = -float("inf")

        # 空棋盘固定选择中心点，避免执行没有意义的完整搜索。
        if np.all(board == 0):
            center = len(board) // 2
            return (center, center), None

        self.alpha_beta(
            board,
            None,
            self.depth,
            -float("inf"),
            float("inf"),
            True,
        )

        # 理论上搜索应始终返回落点；兜底分支保证接口仍返回合法空位。
        if self.move_result is None:
            empty_cells = np.argwhere(board == 0)
            if len(empty_cells) == 0:
                return None, None
            self.move_result = tuple(empty_cells[0])

        if not self.skill_used:
            self.check_for_skill(board, self.move_result)

        return self.move_result, self.skill_target
