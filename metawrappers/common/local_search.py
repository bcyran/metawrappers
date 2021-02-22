from metawrappers.common.mask import random_mask, random_neighbor


class LSMixin:
    """Mixin for use in local search based metaheuristics. Provides method for easy getting random
    masks and neighbors with their scores.
    """

    def _random_mask_with_score(self, X, y):
        mask = random_mask(X.shape[1], random_state=self._rng)
        return mask, self._score_mask(mask, X, y)

    def _random_neighbor_with_score(self, cur_mask, X, y):
        neighbor = random_neighbor(self.neighborhood, cur_mask, random_state=self._rng)
        return neighbor, self._score_mask(neighbor, X, y)
