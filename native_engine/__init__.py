"""Import-friendly package view over the repository's root Python modules.

The repository name contains hyphens, while Python import statements do not.
Point this package's module search path at the repository root so existing files
remain the single source of truth and ``import native_engine`` works directly
from a checkout without an installation step.
"""

from pathlib import Path


__path__ = [str(Path(__file__).resolve().parents[1])]

from .batch import ActionReceipt, Batch, stop_resident
from .engine import Engine, Play, RenderedEngine
from .observation import Observer


def __getattr__(name):
    if name == 'NativePlanningAgent':
        from .planning_policy import NativePlanningAgent
        return NativePlanningAgent
    if name in ("NativeBatchVecEnv", "native_observation_space"):
        from .native_vec_env import NativeBatchVecEnv, native_observation_space
        return {
            "NativeBatchVecEnv": NativeBatchVecEnv,
            "native_observation_space": native_observation_space,
        }[name]
    if name in ("NativeHybridExtractor", "NativeHybridMaskedPolicy"):
        from .native_model import NativeHybridExtractor, NativeHybridMaskedPolicy
        return {
            "NativeHybridExtractor": NativeHybridExtractor,
            "NativeHybridMaskedPolicy": NativeHybridMaskedPolicy,
        }[name]
    if name in (
        "NativeRoleAwareSpatialHeatmapRecurrentPolicy",
        "NativeRoleAwareWorldModelRecurrentPolicy",
        "NativeRoleAwareActionConditionedWorldModelRecurrentPolicy",
        "NativeCrossModalExtractor",
        "NativeCrossModalRecurrentPolicy",
        "NativeSpatialHeatmapExtractor",
        "NativeSpatialHeatmapRecurrentPolicy",
        "NativeWorldModelPPO",
        "NativeWorldModelRecurrentPolicy",
        "NativeActionConditionedWorldModelPPO",
        "NativeActionConditionedWorldModelRecurrentPolicy",
        "NativeRecurrentMaskedPolicy",
        "NativeRecurrentPPO",
    ):
        from .native_cross_modal import NativeCrossModalExtractor
        from .native_spatial_model import NativeSpatialHeatmapExtractor
        from .native_recurrent_model import (
            NativeRoleAwareSpatialHeatmapRecurrentPolicy,
            NativeRoleAwareWorldModelRecurrentPolicy,
            NativeRoleAwareActionConditionedWorldModelRecurrentPolicy,
            NativeCrossModalRecurrentPolicy,
            NativeRecurrentMaskedPolicy,
            NativeRecurrentPPO,
            NativeSpatialHeatmapRecurrentPolicy,
            NativeWorldModelPPO,
            NativeWorldModelRecurrentPolicy,
            NativeActionConditionedWorldModelPPO,
            NativeActionConditionedWorldModelRecurrentPolicy,
        )
        return {
            "NativeRoleAwareSpatialHeatmapRecurrentPolicy": NativeRoleAwareSpatialHeatmapRecurrentPolicy,
            "NativeRoleAwareWorldModelRecurrentPolicy": NativeRoleAwareWorldModelRecurrentPolicy,
            "NativeRoleAwareActionConditionedWorldModelRecurrentPolicy": NativeRoleAwareActionConditionedWorldModelRecurrentPolicy,
            "NativeCrossModalExtractor": NativeCrossModalExtractor,
            "NativeCrossModalRecurrentPolicy": NativeCrossModalRecurrentPolicy,
            "NativeSpatialHeatmapExtractor": NativeSpatialHeatmapExtractor,
            "NativeSpatialHeatmapRecurrentPolicy": (
                NativeSpatialHeatmapRecurrentPolicy
            ),
            "NativeRecurrentMaskedPolicy": NativeRecurrentMaskedPolicy,
            "NativeRecurrentPPO": NativeRecurrentPPO,
            "NativeWorldModelPPO": NativeWorldModelPPO,
            "NativeWorldModelRecurrentPolicy": NativeWorldModelRecurrentPolicy,
            "NativeActionConditionedWorldModelPPO": NativeActionConditionedWorldModelPPO,
            "NativeActionConditionedWorldModelRecurrentPolicy": NativeActionConditionedWorldModelRecurrentPolicy,
        }[name]
    raise AttributeError(name)


__all__ = [
    "NativeRoleAwareSpatialHeatmapRecurrentPolicy",
    "NativeRoleAwareWorldModelRecurrentPolicy",
    "NativeRoleAwareActionConditionedWorldModelRecurrentPolicy",
    "ActionReceipt",
    "Batch",
    "Engine",
    "NativeBatchVecEnv",
    "NativeCrossModalExtractor",
    "NativeCrossModalRecurrentPolicy",
    "NativeSpatialHeatmapExtractor",
    "NativeSpatialHeatmapRecurrentPolicy",
    "NativeWorldModelPPO",
    "NativeWorldModelRecurrentPolicy",
    "NativeActionConditionedWorldModelPPO",
    "NativeActionConditionedWorldModelRecurrentPolicy",
    "NativeHybridExtractor",
    "NativeHybridMaskedPolicy",
    "NativeRecurrentMaskedPolicy",
    "NativeRecurrentPPO",
    "Observer",
    "Play",
    "RenderedEngine",
    "native_observation_space",
    "stop_resident",
]
