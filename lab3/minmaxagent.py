from __future__ import annotations
from typing import TYPE_CHECKING

from exceptions import AgentException

if TYPE_CHECKING:
    from connect4 import Connect4


class MinMaxAgent:
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
            score = self._minimax(game, self.max_depth, is_maximizing=False)
            game.undrop_token(col)
            if score > best_score:
                best_score = score
                best_move = col

        return best_move

    def _minimax(self, game: Connect4, depth: int, is_maximizing: bool) -> int:
        # Sprawdzenie zakończenia gry
        if game.game_over:
            if game.wins == self.token:
                return 1
            elif game.wins is not None:
                return -1
            else:
                return 0

        # Warunek głębokości
        if depth <= 0:
            return game.evaluate(self.token) if self.use_heuristics else 0

        if is_maximizing:
            best = float('-inf')
            for col in game.possible_drops():
                game.drop_token(col)
                best = max(best, self._minimax(game, depth - 1, is_maximizing=False))
                game.undrop_token(col)
            return best
        else:
            worst = float('inf')
            for col in game.possible_drops():
                game.drop_token(col)
                worst = min(worst, self._minimax(game, depth - 1, is_maximizing=True))
                game.undrop_token(col)
            return worst
