import math
from typing import Callable, Iterable, NewType, Tuple, Union

import numpy as np

Timestamp = NewType("Timestamp", float)


class UnivariateTimeseries:
    def __init__(self, timeseries, lhs=None, operator: Union[Callable, None] = None):
        self._rhs = timeseries
        self._lhs = lhs
        self._operator = operator

    def __add__(self, lhs):
        return UnivariateTimeseries(self, lhs, lambda x, y: x + y)

    def __mul__(self, lhs):
        return UnivariateTimeseries(self, lhs, lambda x, y: x * y)

    def __call__(self, timestamps: Iterable[Timestamp]) -> np.ndarray:
        if self._lhs is None:  # Leaf node of call tree.
            # Evaluate method is available only at derived class
            return self.evaluate(timestamps)  # pytype: disable=attribute-error

        return self._operator(np.array(self._rhs(timestamps)), np.array(self._lhs(timestamps)))


class GaussianNoise(UnivariateTimeseries):
    def __init__(self, standard_deviation: float):
        super(GaussianNoise, self).__init__(self)
        self._standard_deviation = standard_deviation

    def evaluate(self, timestamps: Iterable[Timestamp]) -> Iterable[float]:
        return [np.random.normal(0, self._standard_deviation) for _ in timestamps]


class Constant(UnivariateTimeseries):
    def __init__(self, amplitude: float):
        super(Constant, self).__init__(self)
        self._amplitude = amplitude

    def evaluate(self, timestamps: Iterable[Timestamp]) -> Iterable[float]:
        return [self._amplitude for _ in timestamps]


class Sin(UnivariateTimeseries):
    def __init__(self, phase: float, angular_frequency: float, amplitude: float):
        super(Sin, self).__init__(self)
        self._phase = phase
        self._angular_frequency = angular_frequency
        self._amplitude = amplitude

    def evaluate(self, timestamps: Iterable[Timestamp]) -> Iterable[float]:
        return [self._amplitude * math.sin(self._angular_frequency * t + self._phase) for t in timestamps]


class SquareWave(UnivariateTimeseries):
    def __init__(
        self,
        start_phase: float,
        first_level_length: float,
        second_level_length: float,
        first_level: float,
        second_level: float,
    ):
        super(SquareWave, self).__init__(self)
        self._start_phase = start_phase
        self._first_level_length = first_level_length
        self._second_level_length = second_level_length
        self._first_level = first_level
        self._second_level = second_level
        self._wave_length = self._first_level_length + self._second_level_length

    def evaluate(self, timestamps: Iterable[Timestamp]) -> Iterable[float]:
        return [self._evaluate_at(t) for t in timestamps]

    def _evaluate_at(self, t: Timestamp) -> float:
        # Need to figure out if timestamp corresponds to
        # first or second part

        effective_time = (t + self._start_phase) % self._wave_length

        if effective_time < self._first_level_length:
            return self._first_level

        return self._second_level


class MultivariateTimeseries:
    def __init__(self, univariate_timeseries: Iterable[UnivariateTimeseries]):
        self._univariate_timeseries = univariate_timeseries

    def __call__(self, timestamps: Iterable[Timestamp]) -> Tuple[Iterable[Timestamp], np.ndarray]:
        retvals = np.vstack([timeserie(timestamps) for timeserie in self._univariate_timeseries])
        return timestamps, retvals.transpose()
