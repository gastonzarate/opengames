"""Resuelve los nombres que aparecen en los configs a implementaciones."""

from typing import Callable, TypeVar

from core.backend import Backend
from core.model import ModelAdapter

T = TypeVar("T")

_MODELS: dict[str, type] = {}
_BACKENDS: dict[str, type] = {}


class UnknownComponent(KeyError):
    """El nombre del config no corresponde a nada registrado.

    Hereda de `KeyError`, cuyo `__str__` hace `repr()` del primer argumento
    -por eso, sin este override, el mensaje le llega al usuario entre
    comillas espurias: `Error: "No existe el modelo 'x'. Registrados: y"`.
    Es el camino de error más frecuente del CLI (un nombre mal escrito en
    el YAML), así que el mensaje tiene que salir limpio.
    """

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else super().__str__()


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


def _get_class(store: dict[str, type], kind: str, name: str) -> type:
    if name not in store:
        known = ", ".join(sorted(store)) or "ninguno"
        raise UnknownComponent(f"No existe el {kind} '{name}'. Registrados: {known}")
    return store[name]


def _get(store: dict[str, type], kind: str, name: str):
    return _get_class(store, kind, name)()


def get_model(name: str) -> ModelAdapter:
    return _get(_MODELS, "modelo", name)


def get_backend(name: str) -> Backend:
    return _get(_BACKENDS, "backend", name)


def get_backend_class(name: str) -> type[Backend]:
    """Como `get_backend`, pero sin instanciar.

    Sirve para los llamadores que necesitan la clase para construirla ellos
    mismos con argumentos propios (p. ej. `backend_options` de un experimento)
    sin pagar el costo de una instanciación descartable con el constructor
    por defecto, que explotaría contra un backend que exige argumentos.
    """
    return _get_class(_BACKENDS, "backend", name)


def available_models() -> list[str]:
    return sorted(_MODELS)


def available_backends() -> list[str]:
    return sorted(_BACKENDS)


def reset() -> None:
    """Solo para tests: vacía ambos espacios de nombres."""
    _MODELS.clear()
    _BACKENDS.clear()
