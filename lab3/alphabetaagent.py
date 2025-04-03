from __future__ import annotations
from typing import TYPE_CHECKING

from exceptions import AgentException

if TYPE_CHECKING:
    from connect4 import Connect4


class AlphaBetaAgent:
    def __init__(self, token='o', max_depth=3, use_heuristics=True):
        self.token = token
        self.max_depth = max_depth
        self.use_heuristics = use_heuristics

    def choose_move(self, game: Connect4):
        if game.who_moves != self.token:
            raise AgentException('Nie moja tura')

        best_score = float('-inf')
        best_move = None
        for col in game.possible_drops():
            game.drop_token(col)
            score = self._alpha_beta(game, self.max_depth, is_maximizing=False,
                                     alpha=float('-inf'), beta=float('inf'))
            game.undrop_token(col)
            if score > best_score:
                best_score = score
                best_move = col

        return best_move

    def _alpha_beta(self, game: Connect4, depth: int, is_maximizing: bool,
                    alpha: float, beta: float) -> int:
        if game.game_over:
            if game.wins == self.token:
                return 1
            elif game.wins is not None:
                return -1
            else:
                return 0

        if depth <= 0:
            return game.evaluate(self.token) if self.use_heuristics else 0

        if is_maximizing:
            value = float('-inf')
            for col in game.possible_drops():
                game.drop_token(col)
                value = max(value, self._alpha_beta(game, depth - 1, is_maximizing=False,
                                                    alpha=alpha, beta=beta))
                game.undrop_token(col)
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:
            value = float('inf')
            for col in game.possible_drops():
                game.drop_token(col)
                value = min(value, self._alpha_beta(game, depth - 1, is_maximizing=True,
                                                    alpha=alpha, beta=beta))
                game.undrop_token(col)
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value
