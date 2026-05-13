"""Longitudinal MRI tumor segmentation and evolution modeling."""

__all__ = [
    "LongiTumorMamba",
    "LongiTumorMambaConfig",
    "LongiTumorMambaOutput",
    "OmniMamba4DMRI",
]


def __getattr__(name: str):
    if name in __all__:
        from .models import LongiTumorMamba, LongiTumorMambaConfig, LongiTumorMambaOutput, OmniMamba4DMRI

        exports = {
            "LongiTumorMamba": LongiTumorMamba,
            "LongiTumorMambaConfig": LongiTumorMambaConfig,
            "LongiTumorMambaOutput": LongiTumorMambaOutput,
            "OmniMamba4DMRI": OmniMamba4DMRI,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
