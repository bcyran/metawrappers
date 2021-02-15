import time


class RunTimeMixin:
    """Mixin for use in selectors which can be set to run for a specified time."""

    def _start_timer(self):
        self._start_time = time.perf_counter_ns()

    def _time(self):
        return time.perf_counter_ns() - self._start_time

    def _time_ms(self):
        return self._time() // (10 ** 6)

    def _should_end(self, iterations=None):
        if self.run_time:
            if self._time_ms() >= self.run_time:
                return True
        elif iterations:
            if iterations >= self.iterations:
                return True
        return False
