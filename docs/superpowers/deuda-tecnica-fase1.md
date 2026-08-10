# Deuda técnica aceptada — fase 1

**Fecha:** 2026-08-10
**Origen:** triaje de la revisión final del branch de la fase 1.

Todo lo que sigue se conoce, se evaluó y se decidió no arreglar antes de integrar.
No son descuidos. Cada entrada dice por qué se difirió y cuándo conviene revisarla.

Lo que la revisión final marcó como bloqueante ya se arregló y no figura acá.

## Bloquea la primera corrida real de un modelo

### `docker_image` guarda un tag, no un digest

El spec, sección 5, es explícito: *"El digest de la imagen Docker es obligatorio, no
opcional"*, porque las seis extensiones CUDA de TRELLIS.2 se compilan desde fuente y
no hay garantía de que dos builds produzcan binarios equivalentes. Hoy `ModelSpec.docker_image`
guarda `"opengames/mock:0.1.0"` y `collect_provenance()` lo copia tal cual.

En la fase 1 no hay imágenes que fijar, así que no rompe nada. **Hay que cerrarlo antes de
la primera corrida de TRELLIS.2**, o las corridas de la fase 2 no serán reproducibles.
Incluye renombrar `test_provenance_carries_the_docker_digest`, que hoy verifica un tag y
consagra la semántica equivocada.

## Revisar en la fase 2, con los backends remotos

### Sin exclusión mutua a nivel de runner

Dos procesos con el mismo experimento calculan el mismo `run_id`, los dos ven `exists()`
falso y los dos ejecutan, escribiendo en el mismo `outputs/`. El marcador `.in-progress`
existe en el almacén pero el runner nunca lo consulta. Con GPU por hora, esto es dinero.

### Sin timeout global de `poll()`

El bucle tiene una espera mínima para no ser un busy-loop, pero no hay límite de tiempo
total. Irrelevante mientras el único backend sea sincrónico; necesario en cuanto exista
RunPod o SageMaker.

### `execute()` puede lanzar fuera del contrato documentado

Se documentan `InsufficientVram` y `GenerationFailed`, pero un fallo de `fetch()` o de
`write_metrics()` propaga su propia excepción. Cerrarlo cuando se estabilice el contrato
de los backends remotos.

### Una excepción en `teardown()` enmascara la original

`teardown()` corre en un `finally`. Si lanza, se pierde la causa raíz. En un backend
efímero el error de liberación importa, pero perder la causa raíz importa más.

### El registro depende de que alguien importe los módulos

Solo `cli.py` importa `models.mock` y `backends.local` para que se registren. Quien use
`core.experiment` como biblioteca recibe "No existe el modelo 'mock'". Conviene un módulo
de wiring explícito o descubrimiento por entry points.

### `LocalBackend._runs` crece sin cota

`teardown()` limpia el disco pero no la entrada del diccionario. **Cuidado con el arreglo
obvio:** no se puede borrar la entrada, porque `test_teardown_is_idempotent` exige que
`poll()` siga funcionando después del `teardown()`. Hace falta un tombstone, no un `del`.

### "Manifiesto vacío significa no cacheable" asume que todo modelo produce archivos

La regla resuelve el envenenamiento de caché por éxito vacío, pero un modelo legítimo que
solo produjera métricas quedaría sin cachear para siempre. Hoy no existe tal modelo.

## Revisar en la fase 3, con la etapa de evaluación

### Un job fallido aborta el sweep entero

`run_experiment` es una comprensión de lista sin manejo por job. Un combo que falle
—por ejemplo un modelo que no entra en la VRAM del backend elegido— tira abajo el resto.

Se difirió a propósito: la caché acota el costo, porque al reintentar los jobs ya
completados no se re-ejecutan, así que el desperdicio máximo es el job que falló. Cambiar
el contrato de retorno de `run_experiment` ahora, con un solo modelo simulado, es especular
sobre una forma de reporte que recién se conoce cuando exista la etapa `evaluate` que pide
la sección 8 del spec.

### `spec.accepts` y `spec.produces` no se usan

La sección 4.1 del spec justifica `describe()` por la validación temprana, pero hoy solo se
valida la VRAM. Son campos muertos hasta que haya modelos con modalidades de entrada
distintas y valga la pena rechazar combinaciones incompatibles.

## Menores, sin urgencia

- **`_repo_sha()` no marca árbol sucio.** Una corrida desde un working tree modificado
  guarda un SHA que no describe el código que corrió. Un sufijo `-dirty` lo resuelve.
- **`ExperimentConfig` no usa `extra="forbid"`.** Un typo de clave en el YAML —`seed` en
  vez de `seeds`— se ignora en silencio y cae al valor por defecto.
- **`_resolve_relative_inputs` no expande `~`.** Falla ruidosa, no corrupción silenciosa.
- **Detección de colisión de basenames duplicada** entre `core/runner.py` y
  `backends/local.py`, por haber nacido en tareas distintas. Extraer a un helper de `core/`.
- **`fetch()` hace `mkdir()` antes de chequear colisiones**, así que un fallo por colisión
  deja un directorio vacío creado.
- **Falta un test de `teardown()` con un `RunHandle` fabricado** que nunca pasó por
  `submit()`. El código lo maneja bien; falta la cobertura.
- **El test de capas no detecta imports dinámicos** (`importlib.import_module("boto3")`).
  Es una limitación estructural de cualquier chequeo por AST, ya documentada en el propio
  docstring del test.

## Desvío del spec a corregir en el documento

`ModelAdapter.generate(job, workdir)` recibe `workdir`, mientras la sección 4.1 del spec
declara `generate(job)`. El desvío es correcto —el adapter necesita saber dónde escribir—
pero el spec quedó desactualizado.
