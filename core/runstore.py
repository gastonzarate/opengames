"""Persistencia de corridas: identidad por contenido, procedencia y caché.

`exists()` solo devuelve verdadero cuando `metrics.json` es JSON válido Y no
hay un marcador `.in-progress` presente. `write_metrics()` escribe
`metrics.json` de forma atómica y recién ahí borra el marcador: una corrida
interrumpida deja el directorio a medias y nunca se toma como cacheada.

## Por qué un marcador con dueño verificable

Un `run_id` es determinístico: dos intentos sobre el mismo contenido escriben
en el mismo directorio. Eso significa que un reintento después de un crash
puede encontrarse con outputs parciales del intento anterior, y hay que
decidir si son basura para limpiar o el trabajo legítimo de un intento que
sigue vivo. Nada en la firma pública `RunStore(root: Path)` impide construir
más de una instancia sobre la misma raíz — dentro del mismo proceso o desde
procesos distintos — así que esa decisión no puede depender de estado en
memoria de la instancia (eso fue el error de la ronda anterior).

El marcador `.in-progress` guarda el PID de quien lo creó. Un marcador se
considera huérfano -y por lo tanto seguro de limpiar- únicamente si:
- no se puede parsear (JSON inválido o sin campo `pid`), o
- el proceso dueño ya no existe (`os.kill(pid, 0)` falla con `ProcessLookupError`).

Si el proceso dueño sigue vivo -sea porque es un intento real en curso, sea
porque es OTRA instancia de `RunStore` en el mismo proceso que llama
`write_job()` de nuevo- el marcador nunca se trata como basura.

El marcador se reescribe con el mismo patrón de temporal + `os.replace()`
que usa `write_metrics()`. Es necesario: a diferencia de las rondas
anteriores, donde el marcador era un archivo vacío y un estado intermedio
truncado no significaba nada, ahora lleva JSON que hay que poder parsear.
Sin escritura atómica, un lector concurrente que cayera en la ventana entre
truncar y escribir vería el archivo vacío, el parseo fallaría, y se
trataría como huérfano un intento que en realidad está vivo -literalmente
la propiedad 6 rota bajo concurrencia real en lugar de por estado en
memoria.

Por la misma razón, `os.kill(pid, 0)` solo se usa para consultar en
plataformas POSIX. En Windows, `os.kill` con cualquier señal que no sea
`CTRL_C_EVENT`/`CTRL_BREAK_EVENT` invoca `TerminateProcess`: `os.kill(pid, 0)`
no consulta, **mata**, y lo hace sin lanzar ninguna excepción distinguible.
Windows no es una plataforma soportada hoy (CI en Ubuntu, desarrollo en
macOS), pero el costo de no blindar esto sería catastrófico y silencioso el
día que alguien lo ejecute ahí. Fuera de POSIX, un marcador ajeno siempre se
trata como vivo: no se limpia nada, nunca se arriesga terminar un proceso
ni destruir un artefacto por no poder confirmar sin efectos secundarios que
su dueño murió.

## Dos residuos conocidos, aceptados por diseño

Esto no es una lista de casos por resolver: es una lista de casos que se
decidió NO resolver porque cualquier solución automática es peor que el
problema. Quien retome este módulo debería tratarlos como decisiones
tomadas, no como descuidos.

1. **Reintento secuencial dentro del mismo proceso.** Si el "intento
   anterior" nunca llegó a crashear de verdad -mismo proceso, mismo PID,
   solo se está llamando `write_job()` de nuevo sobre restos que ese mismo
   proceso dejó- el marcador se ve vivo (porque lo está) y no se limpia.
   Deja basura de disco. Es el costo aceptado del lado seguro: nunca un
   falso positivo de caché, nunca la pérdida de un artefacto de un intento
   vivo, a cambio de algo de basura ocasional.

2. **Reutilización de PID.** Si el sistema operativo recicla el PID que
   quedó en un marcador huérfano y se lo asigna a otro proceso cualquiera
   que esté vivo en el momento de la consulta, `_pid_alive()` da verdadero
   y la limpieza se saltea: el intento nuevo escribe sus outputs junto a
   los restos del intento que en realidad sí murió. `load_artifacts()`
   devuelve la unión de ambos sin ninguna señal de que algo esté mal. No se
   intenta resolver: toda alternativa considerada (guardar más contexto de
   identidad del proceso, expirar marcadores por tiempo, bloquear
   reintentos ante duda) o arriesga borrar datos de un intento realmente
   vivo, o bloquea para siempre un reintento legítimo -ambas peores que
   basura ocasional mezclada con outputs nuevos.
"""

import hashlib
import json
import os
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.job import Artifacts, Job
from core.model import ModelSpec

_CHUNK = 1 << 20
_MARKER_NAME = ".in-progress"


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()


def compute_run_id(job: Job, spec: ModelSpec) -> str:
    """Identidad por contenido. No depende del tiempo ni de rutas."""
    payload = {
        "model": spec.name,
        "revision": spec.revision,
        "params": job.params,
        "export": job.export,
        "seed": job.seed,
        "inputs": {key: _hash_file(path) for key, path in sorted(job.inputs.items())},
    }
    return hashlib.sha256(_canonical(payload)).hexdigest()[:16]


def _repo_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def collect_provenance(job: Job, spec: ModelSpec, backend_name: str) -> dict[str, Any]:
    return {
        "model": spec.name,
        "model_revision": spec.revision,
        "docker_image": spec.docker_image,
        "backend": backend_name,
        "seed": job.seed,
        "repo_sha": _repo_sha(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _atomic_write_text(path: Path, text: str) -> None:
    """Escribe `text` en `path` sin dejar nunca un estado truncado visible.

    Escribe a un temporal y recién después reemplaza atómicamente con
    `os.replace()`. Un lector concurrente ve el contenido viejo completo o
    el nuevo completo, jamás algo a medio escribir.

    El nombre del temporal incorpora el PID y un componente aleatorio para
    que sea único por escritor. Un nombre fijo (`f"{path.name}.tmp"`) lo
    comparten todos los escritores del mismo archivo: dos procesos
    escribiendo el mismo `run_id` a la vez terminan pisándose el temporal
    del otro, y `os.replace()` de uno puede fallar con `FileNotFoundError`
    -o peor, publicar contenido a medio escribir del otro escritor bajo un
    nombre que parece atómico. El `run_id` en sí no puede depender del azar
    (tiene que ser determinístico por contenido), pero el temporal sí.
    """
    temp_path = path.parent / f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
    try:
        temp_path.write_text(text)
        os.replace(str(temp_path), str(path))
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def _marker_owner_pid(marker: Path) -> int | None:
    """PID guardado en el marcador, o None si no se puede parsear."""
    try:
        data = json.loads(marker.read_text())
        pid = data["pid"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return None
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return None
    return pid


def _pid_alive(pid: int) -> bool:
    """True si el proceso `pid` sigue existiendo (vivo o zombie, da igual).

    Solo consulta con `os.kill(pid, 0)` en POSIX. Fuera de POSIX (Windows),
    esa misma llamada no es una consulta: `os.kill` con cualquier señal que
    no sea `CTRL_C_EVENT`/`CTRL_BREAK_EVENT` invoca `TerminateProcess`, así
    que "preguntar" terminaría al proceso dueño real. Sin forma de
    consultar sin efectos secundarios, se asume vivo: nunca se limpia nada
    fuera de POSIX por esta vía.
    """
    if os.name != "posix":
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Existe pero no nos pertenece: sigue vivo.
        return True
    except OSError:
        # Cualquier otra falla del syscall: no podemos confirmar que murió,
        # así que no lo tratamos como huérfano. Fallar del lado seguro.
        return True
    return True


class RunStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _dir(self, run_id: str) -> Path:
        return self.root / run_id

    def create(self, run_id: str) -> Path:
        """Asegura los directorios de la corrida. Solo `mkdir -p`: idempotente
        y nunca destructivo, sin excepciones. No toca el marcador ni borra
        nada bajo ninguna circunstancia."""
        base = self._dir(run_id)
        (base / "inputs").mkdir(parents=True, exist_ok=True)
        (base / "outputs").mkdir(parents=True, exist_ok=True)
        return base

    def exists(self, run_id: str) -> bool:
        """Verdadero solo si la corrida completó: `metrics.json` es JSON
        válido y no queda un marcador `.in-progress`, sin importar de quién
        sea ese marcador — su sola presencia basta para decir "no cacheada".

        Si además hay un manifiesto de artefactos (`write_artifacts()` lo
        escribe antes de `write_metrics()`), se verifica que los archivos
        que declara existan de verdad en `outputs/`, y que el manifiesto no
        esté vacío. Sin esto, un backend que "termina bien" sin producir
        ningún archivo -un bug real, no un caso válido en este pipeline,
        donde toda corrida exitosa produce al menos un artefacto- quedaría
        cacheado para siempre con `outputs/` vacío. Si no hay manifiesto
        (por ejemplo, un caller que escribe outputs/ a mano y llama
        `write_metrics()` directo, como hacen algunos tests de este módulo),
        se conserva el comportamiento anterior: no se exige nada sobre los
        archivos.
        """
        base_dir = self._dir(run_id)
        if (base_dir / _MARKER_NAME).is_file():
            return False
        metrics_path = base_dir / "metrics.json"
        if not metrics_path.is_file():
            return False
        try:
            json.loads(metrics_path.read_text())
        except (json.JSONDecodeError, OSError):
            return False

        manifest_path = base_dir / "artifacts.json"
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text())
            except (json.JSONDecodeError, OSError):
                return False
            if not manifest:
                return False
            outputs = base_dir / "outputs"
            for filename in manifest.values():
                if not (outputs / filename).is_file():
                    return False
        return True

    def inputs_dir(self, run_id: str) -> Path:
        return self.create(run_id) / "inputs"

    def outputs_dir(self, run_id: str) -> Path:
        return self.create(run_id) / "outputs"

    def _reset_attempt(self, base: Path) -> None:
        """Descarta inputs/outputs de un intento confirmado huérfano."""
        for name in ("inputs", "outputs"):
            directory = base / name
            if directory.exists():
                shutil.rmtree(directory)
            directory.mkdir(parents=True, exist_ok=True)

    def write_job(self, run_id: str, job: Job) -> None:
        """Punto de entrada de un intento.

        - Si la corrida ya completó (`exists()`), es un acceso tardío: no se
          toca nada, ni siquiera el marcador. Nunca se pierde un resultado ya
          cacheado por una llamada posterior a `write_job()`.
        - Si hay un marcador y su dueño sigue vivo (incluida la posibilidad
          de que sea este mismo proceso, vía otra instancia de `RunStore`),
          se preserva todo: no es basura, es un intento en curso.
        - Si el marcador es huérfano (dueño muerto o marcador ilegible), se
          limpian los restos antes de empezar el intento nuevo.
        - Se puede llamar más de una vez dentro del mismo intento sin
          destruir lo que ese intento ya escribió: la segunda llamada ve el
          marcador que puso la primera, con el PID de este mismo proceso,
          vivo por definición.
        """
        base = self.create(run_id)
        if self.exists(run_id):
            (base / "job.json").write_text(job.model_dump_json(indent=2))
            return

        marker = base / _MARKER_NAME
        if marker.is_file():
            owner_pid = _marker_owner_pid(marker)
            if owner_pid is None or not _pid_alive(owner_pid):
                self._reset_attempt(base)

        _atomic_write_text(marker, json.dumps({"pid": os.getpid()}))
        (base / "job.json").write_text(job.model_dump_json(indent=2))

    def write_provenance(self, run_id: str, data: dict[str, Any]) -> None:
        (self.create(run_id) / "provenance.json").write_text(json.dumps(data, indent=2))

    def write_artifacts(self, run_id: str, files: dict[str, Path]) -> None:
        """Persiste el mapa de clave lógica a nombre de archivo en `outputs/`.

        La clave lógica es la que usa el adapter o el backend (p. ej.
        `"mesh"`, `"preview"`), no necesariamente el nombre del archivo. Sin
        este manifiesto, `load_artifacts()` solo puede reconstruir
        `Artifacts.files` indexando por basename -indistinguible de la
        clave lógica en el modelo simulado, donde coinciden, pero no contra
        un adapter real con claves distintas del nombre de archivo. Eso
        hace que una corrida fresca y la misma corrida cacheada devuelvan
        `Artifacts.files` con claves distintas: el peor modo de falla
        posible para quien hace `artifacts.files["mesh"]`.

        Se llama antes de `write_metrics()` (ver `execute()` en
        `core/runner.py`): así, si el proceso muere entre las dos
        escrituras, el marcador `.in-progress` sigue presente y `exists()`
        da falso de todos modos, sin depender del orden para la
        atomicidad.
        """
        base = self.create(run_id)
        manifest = {key: Path(path).name for key, path in files.items()}
        _atomic_write_text(base / "artifacts.json", json.dumps(manifest, indent=2))

    def write_metrics(self, run_id: str, metrics: dict[str, float]) -> None:
        base = self.create(run_id)
        # Atómico: si el proceso muere a mitad de la escritura, metrics.json
        # nunca queda truncado.
        _atomic_write_text(base / "metrics.json", json.dumps(metrics, indent=2))
        # Recién ahora, con metrics.json ya persistido, la corrida completó:
        # se borra el marcador de "en progreso".
        (base / _MARKER_NAME).unlink(missing_ok=True)

    def load_artifacts(self, run_id: str) -> Artifacts:
        """Lanza `FileNotFoundError` si la corrida nunca fue iniciada (no
        existe `job.json`). Para una corrida iniciada pero no completada,
        devuelve `Artifacts` con lo que haya (potencialmente vacío).

        Si `write_artifacts()` dejó un manifiesto (`artifacts.json`),
        `Artifacts.files` se reconstruye con las mismas claves lógicas que
        usó el adapter/backend -idénticas entre una corrida fresca y la
        misma corrida cacheada. Sin manifiesto (corridas escritas a mano
        sin pasar por `write_artifacts()`), se cae al comportamiento
        anterior: indexar por nombre de archivo.
        """
        base_dir = self._dir(run_id)
        job_path = base_dir / "job.json"
        if not job_path.is_file():
            raise FileNotFoundError(f"No existe la corrida {run_id}")

        metrics_path = base_dir / "metrics.json"
        metrics = {}
        if metrics_path.is_file():
            try:
                metrics = json.loads(metrics_path.read_text())
            except (json.JSONDecodeError, OSError):
                pass

        outputs = base_dir / "outputs"
        manifest_path = base_dir / "artifacts.json"
        manifest: dict[str, Any] | None = None
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text())
            except (json.JSONDecodeError, OSError):
                manifest = None

        files = {}
        if manifest is not None:
            for key, filename in manifest.items():
                candidate = outputs / filename
                if candidate.is_file():
                    files[key] = candidate
        elif outputs.is_dir():
            files = {path.name: path for path in sorted(outputs.iterdir()) if path.is_file()}

        return Artifacts(files=files, metrics=metrics)
