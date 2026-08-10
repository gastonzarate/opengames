"""Resuelve los nombres que aparecen en los configs a implementaciones."""

from typing import Callable, TypeVar

from core.backend import Backend
from core.model import ModelAdapter

T = TypeVar("T")

_MODELS: dict[str, type] = {}
_BACKENDS: dict[str, type] = {}


class UnknownComponent(KeyError):
    """El nombre del config no corresponde a nada registrado."""


def _register(store: dict[str, type], kind: str, name: str) -> Callable[[type[T]], type[T]]:
    def decorator(cls: type[T]) -> type[T]:
        if name in store:
            raise ValueError(f"Ya hay un {kind} registrado como '{name}'")
        store[name] = cls
        return cls

    return decorator


def register_model(name: str) -> Callable[[type[T]], type[T]]:
    return _register(_MODELS, "modelo", name)


def register_backend(name: str) -> Callable[[type[T]], type[T]]:
    return _register(_BACKENDS, "backend", name)


def _get(store: dict[str, type], kind: str, name: str):
    if name not in store:
        known = ", ".join(sorted(store)) or "ninguno"
        raise UnknownComponent(f"No existe el {kind} '{name}'. Registrados: {known}")
    return store[name]()


def get_model(name: str) -> ModelAdapter:
    return _get(_MODELS, "modelo", name)


def get_backend(name: str) -> Backend:
    return _get(_BACKENDS, "backend", name)


def available_models() -> list[str]:
    return sorted(_MODELS)


def available_backends() -> list[str]:
    return sorted(_BACKENDS)


def reset() -> None:
    """Solo para tests: vacía ambos espacios de nombres."""
    _MODELS.clear()
    _BACKENDS.clear()
