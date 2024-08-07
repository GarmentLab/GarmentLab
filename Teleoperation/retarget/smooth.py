from collections import deque
from typing import Optional

import numpy as np


class LowPassFilter:
    def __init__(self, alpha: float) -> None:
        self._alpha = alpha
        self._last_value = None
        self._last_raw_value = None
        self._initialized = False

    def __call__(self, value: float, alpha: Optional[float] = None) -> float:
        self._last_raw_value = value

        if not self._initialized:
            self._last_value = value
            self._initialized = True
        else:
            alpha = self._alpha if alpha is None else alpha
            value = alpha * value + (1 - alpha) * self._last_value
            self._last_value = value
        return value

    @property
    def last_value(self) -> float:
        return self._last_value

    @property
    def last_raw_value(self) -> float:
        return self._last_raw_value

    @property
    def initialized(self) -> bool:
        return self._initialized


class OneEuroFilter:
    def __init__(
        self,
        frequency: float,
        min_cutoff: float = 1.0,
        beta: float = 0.0,
        derivate_cutoff: float = 1.0,
    ) -> None:
        self._frequency = frequency
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._derivate_cutoff = derivate_cutoff
        self._x_filter = LowPassFilter(self.compute_alpha(min_cutoff))
        self._dx_filter = LowPassFilter(self.compute_alpha(derivate_cutoff))
        self._last_timestamp = None

    def compute_alpha(self, cutoff: float) -> float:
        tau: float = 1.0 / (2 * np.pi * cutoff)
        te: float = 1.0 / self._frequency
        return 1.0 / (1.0 + tau / te)

    def __call__(
        self, value: float, value_scale: float, timestamp: Optional[int] = None
    ) -> float:
        if self._last_timestamp is not None and timestamp is not None:
            self._frequency = 1.0 / (timestamp - self._last_timestamp)
        self._last_timestamp = timestamp

        dvalue = (
            0.0
            if not self._x_filter.initialized
            else (value - self._x_filter.last_raw_value) * value_scale * self._frequency
        )
        edvalue = self._dx_filter(dvalue)
        cutoff = self._min_cutoff + self._beta * np.abs(edvalue)
        return self._x_filter(value, self.compute_alpha(cutoff))


class RelativeVelocityFilter:
    def __init__(
        self, window_size: int, velocity_scale: float, rate: float = 15
    ) -> None:
        self._max_window_size = window_size
        self._window = deque(maxlen=window_size)
        self._last_value = None
        self._last_value_scale = None
        self._last_timestamp = None
        self._rate = rate
        self._duration = 1.0 / rate
        self._velocity_scale = velocity_scale
        self._filter = LowPassFilter(1.0)
        self._initialized = False

    def __call__(
        self, value: float, value_scale: float, timestamp: Optional[int] = None
    ) -> float:
        if not self._initialized:
            alpha = 1.0
            self._initialized = True

        else:
            distance = value * value_scale - self._last_value * self._last_value_scale

            if timestamp is None:
                unit_scale = 1.0
                duration = self._duration
                assumed_max_duration = self._duration
            else:
                unit_scale = nanoseconds_per_second = 1_000_000_000
                duration = timestamp - self._last_timestamp
                assumed_max_duration = nanoseconds_per_second / self._rate

            cumulative_distance = distance
            cumulative_duration = duration

            max_cumulative_duration = (1 + len(self._window)) * assumed_max_duration

            for previous_distance, previous_duration in self._window:
                if cumulative_duration + previous_duration > max_cumulative_duration:
                    break
                cumulative_distance = cumulative_distance + previous_distance
                cumulative_duration = cumulative_duration + previous_duration

            velocity = cumulative_distance / (cumulative_duration / unit_scale)
            alpha = 1.0 - 1.0 / (1.0 + self._velocity_scale * np.abs(velocity))
            self._window.appendleft((distance, duration))

        self._last_value = value
        self._last_value_scale = value_scale
        self._last_timestamp = timestamp

        return self._filter(value, alpha)


class VelocityFilter:
    def __init__(self, window_size: int, velocity_scale: float) -> None:
        self._window_size = window_size
        self._velocity_scale = velocity_scale
        self._filter = RelativeVelocityFilter(window_size, velocity_scale)

    def __call__(
        self, value: np.ndarray, timestamp: Optional[float] = None
    ) -> np.ndarray:
        assert value.ndim == 2
        value_min = np.min(value, axis=0)
        value_max = np.max(value, axis=0)
        value_scale = 1.0 / (value_max - value_min).mean()
        return self._filter(value, value_scale, timestamp)


if __name__ == "__main__":
    import glob

    from natsort import natsorted

    from retarget import plot_hand_motion_keypoints

    velocity_filter = VelocityFilter(5, 5)
    filenames = glob.glob("hand_pose/*joint*.npy")
    filenames = natsorted(filenames)
    target = np.stack([np.load(filename) for filename in filenames])[:]
    target -= target[0, 0]

    plot_hand_motion_keypoints(target, "before_filtering.gif")
    # plot_hand_motion_keypoints(target)

    for i in range(target.shape[0]):
        print("Filtering frame", i)
        # print("  Before:", target[i])
        # target[i] = velocity_filter(target[i], i * 1_000_000_000 / 15)
        target[i] = velocity_filter(target[i])
        # print("  After:", target[i])

    plot_hand_motion_keypoints(target, "after_filtering.gif")
    # plot_hand_motion_keypoints(target)
