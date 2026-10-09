"""Independent native battle client; no Firstlight Python imports."""
from .batch import ActionReceipt, Batch, stop_resident
from .engine import Engine, Play, RenderedEngine
from .observation import Observer


def __getattr__(name):
    # Keep the lightweight engine client importable without Gym/SB3. Training
    # dependencies are loaded only when the VecEnv is requested.
    if name in ('NativeBatchVecEnv', 'native_observation_space'):
        from .native_vec_env import NativeBatchVecEnv, native_observation_space
        return {
            'NativeBatchVecEnv': NativeBatchVecEnv,
            'native_observation_space': native_observation_space,
        }[name]
    if name in ('NativeHybridExtractor', 'NativeHybridMaskedPolicy'):
        from .native_model import NativeHybridExtractor, NativeHybridMaskedPolicy
        return {
            'NativeHybridExtractor': NativeHybridExtractor,
            'NativeHybridMaskedPolicy': NativeHybridMaskedPolicy,
        }[name]
    raise AttributeError(name)


__all__ = [
    'ActionReceipt', 'Batch', 'Engine', 'NativeBatchVecEnv',
    'NativeHybridExtractor', 'NativeHybridMaskedPolicy', 'Observer', 'Play',
    'RenderedEngine', 'native_observation_space', 'stop_resident',
]
