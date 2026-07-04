"""Tiny name -> class registry so the runner can resolve models/baselines by config."""

from __future__ import annotations

_REGISTRY: dict[str, type] = {}


def register(name: str):
    def _decorator(cls):
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            raise KeyError(f"model name {name!r} already registered to {_REGISTRY[name]!r}")
        _REGISTRY[name] = cls
        cls.name = name
        return cls

    return _decorator


def get_model(name: str) -> type:
    if name not in _REGISTRY:
        raise KeyError(f"unknown model {name!r}; available: {available()}")
    return _REGISTRY[name]


def available() -> list[str]:
    return sorted(_REGISTRY)
