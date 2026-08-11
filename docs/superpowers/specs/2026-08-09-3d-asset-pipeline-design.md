# Pipeline modular de generación de assets 3D

**Fecha:** 2026-08-09
**Estado:** diseño aprobado, pendiente de plan de implementación

## 1. Objetivo

Construir un banco de experimentación para generación de assets 3D por IA, que permita
probar distintos modelos sobre distinta infraestructura sin reescribir código, y comparar
resultados de forma reproducible.

El objetivo final es producir assets para un videojuego. Este spec cubre solamente la
infraestructura de experimentación. Las decisiones sobre el juego —género, motor, estilo
artístico— siguen abiertas y no condicionan este diseño.

## 2. Alcance

**Entra:**

- Interfaces `ModelAdapter` y `Backend`, y el contrato `Job` que las conecta.
- Backends: `local`, `runpod`, `ec2`, `sagemaker`.
- Adapters: TRELLIS.2 (primero), TripoSG, PartCrafter, UniRig.
- Almacén de corridas reproducible y con caché por contenido.
- Etapa `evaluate` con métricas objetivas y renderizado bajo HDRI controlado.
- Una imagen Docker por adapter.

**No entra, y por qué:**

- **Etapa texto → imagen.** Es necesaria para el proyecto: los assets parten de ideas en
  texto y casi todos los generadores 3D piden una imagen. Se difiere porque define el
  estilo artístico del juego, que es una decisión de producto y no de infraestructura. La
  arquitectura la admite sin cambios: es un adapter más, que declara entrada `text` y
  salida `image`.
- **Modelos de escena** (HY-World 2.0, WorldGen). Requieren multi-GPU o exceden el
  hardware disponible, y su licencia es restrictiva en el caso de Tencent.
- **Animación.** HY-Motion pide 26 GB y excluye personajes no humanoides por
  documentación propia. Sin cobertura abierta hoy.
- **Motor de juego e integración.** Fuera del banco de experimentación.

## 3. Hallazgos que fundamentan el diseño

Verificados contra las APIs de GitHub y Hugging Face, el endpoint en vivo del 3D Arena y
los README oficiales, el 2026-08-09.

### 3.1 Licencias

Los modelos de Tencent (Hunyuan3D, HY-World, HY-Motion) usan la Tencent Community
License: territorio mundial **excepto Unión Europea, Reino Unido y Corea del Sur**, tope de
un millón de usuarios activos mensuales, y prohibición de usar las salidas para entrenar
otros modelos.

TRELLIS, TRELLIS.2, TripoSG, PartCrafter, Direct3D-S2, UniRig y SkinTokens son MIT.
UniLat3D, InstantMesh, WorldGen y GEN3C son Apache-2.0.

**Decisión: el pipeline se construye solo sobre modelos MIT o Apache-2.0.** Los adapters de
modelos Tencent quedan permitidos por la arquitectura pero fuera del alcance inicial.

### 3.2 Calidad medida

El 3D Arena ubica a TRELLIS.2-4B en el puesto 17 de 23 (ELO 1124), por debajo de TRELLIS
v1 (1302) y de Hunyuan3D-2 (1284).

Análisis de los GLB que el arena usa para votar:

| | TRELLIS v1 | TRELLIS.2-4B |
|---|---|---|
| Triángulos | 40.738 | 289.032 |
| Mapas de material | solo baseColor | baseColor + metallicRoughness |
| Rugosidad | uniforme (0,904) | por téxel |

El arena renderiza con iluminación por defecto del componente Model3D de Gradio, sin mapa
de entorno documentado, y su propio paper reconoce un sesgo análogo: los splats reciben
renderizado sin iluminar y ganan 16,6 ELO sobre las mallas. El ejemplo oficial de TRELLIS.2
carga un HDRI y el repositorio incluye ocho.

**Hipótesis no verificada:** el arena penaliza el PBR por renderizarlo sin iluminación
basada en imagen. Resolverla es el primer experimento del banco (sección 7.3). Hasta
entonces, la elección de TRELLIS.2 se apoya en un argumento independiente del ELO: una
textura con la iluminación horneada es un defecto en un motor de juego con luz dinámica,
porque la sombra pintada no coincide con la sombra calculada.

### 3.3 Hardware disponible

`gaston-pc`: RTX 3070 con 8 GB de VRAM (cerca de 1 GB ocupado por el escritorio de
Windows), 27 GB de RAM, 738 GB libres, Ubuntu 24.04 sobre WSL2 con passthrough de CUDA
funcionando. Corre además los bots de trading `opentrading` de forma permanente.

Entran en 8 GB: TripoSG, PartCrafter y UniRig. No entran: TRELLIS.2 (24 GB),
Hunyuan3D-2.1 texturizado (21 GB), HY-Motion (26 GB), HY-World (multi-GPU).

Costos de referencia por hora, a verificar al contratar: RunPod RTX 4090 unos USD 0,34;
AWS `g6e.xlarge` con L40S de 48 GB unos USD 1,86.

## 4. Arquitectura

Dos ejes de variación independientes: **qué modelo** y **dónde corre**. Se mantienen
desacoplados mediante tres interfaces.

```
opengames/
├── core/
│   ├── model.py       # ModelAdapter
│   ├── backend.py     # Backend
│   ├── job.py         # Job y Artifacts
│   ├── registry.py    # nombre en config → implementación
│   └── runstore.py    # persistencia y caché de corridas
├── models/
│   ├── trellis2.py
│   ├── triposg.py
│   ├── partcrafter.py
│   └── unirig.py
├── backends/
│   ├── local.py
│   ├── runpod.py
│   ├── ec2.py
│   └── sagemaker.py
├── eval/
│   ├── metrics.py     # análisis de GLB
│   └── render.py      # renderizado bajo HDRI
├── experiments/       # configs declarativos
└── docker/            # un Dockerfile por adapter
```

### 4.1 ModelAdapter

```python
class ModelAdapter(Protocol):
    def describe(self) -> ModelSpec: ...
    def load(self) -> None: ...
    def generate(self, job: Job) -> Artifacts: ...
```

`ModelSpec` declara: identificador, revisión de los pesos, VRAM mínima en GB, modalidades
de entrada aceptadas (`image`, `multiview`, `text`, `mesh`), artefactos que produce, y la
etiqueta de la imagen Docker que necesita.

**Regla de acoplamiento:** un adapter lee y escribe rutas locales. No conoce S3, RunPod ni
SageMaker. Un `import boto3` dentro de `models/` es un error de diseño.

`describe()` habilita la validación temprana: el runner compara la VRAM requerida contra la
del backend elegido y **falla antes de aprovisionar nada**. Pedir TRELLIS.2 sobre `local`
debe fallar en un segundo con un mensaje explícito, no reventar por falta de memoria
después de veinte minutos.

### 4.2 Backend

```python
class Backend(Protocol):
    def capabilities(self) -> BackendSpec: ...
    def submit(self, job: Job) -> RunHandle: ...
    def poll(self, handle: RunHandle) -> RunStatus: ...
    def fetch(self, handle: RunHandle, dest: Path) -> Artifacts: ...
    def teardown(self, handle: RunHandle) -> None: ...
```

`BackendSpec` declara VRAM disponible y si el backend es efímero. `teardown()` es
obligatorio y debe ser idempotente: con GPU por hora, un pod que queda encendido por un
error es dinero perdido de forma silenciosa.

### 4.3 Job

Contrato serializable a JSON, único punto de contacto entre las dos capas.

Entrada: identificador del modelo, rutas de los archivos de entrada, parámetros del modelo,
parámetros de exportación y semilla.

Salida (`Artifacts`): rutas de los archivos generados, métricas de ejecución (duración,
pico de VRAM) y la procedencia descrita en la sección 5.

## 5. Reproducibilidad

Cada corrida escribe un directorio inmutable bajo `runs/<run_id>/`:

```
runs/<run_id>/
├── job.json         # config resuelto, con todos los defaults explícitos
├── provenance.json  # seed · revisión HF · digest de la imagen · SHA del repo
├── inputs/
├── outputs/         # .glb, .mp4
└── metrics.json
```

**El digest de la imagen Docker es obligatorio, no opcional.** Las seis extensiones CUDA de
TRELLIS.2 se compilan desde fuente y no hay garantía de que dos builds produzcan binarios
equivalentes. Sin el digest, la corrida no es reproducible.

`run_id` es el hash de `(modelo, revisión, parámetros resueltos, hash de las entradas)`.
Repetir un config ya ejecutado devuelve el resultado cacheado sin consumir GPU.

## 6. Docker: una imagen por adapter

TRELLIS.2 requiere CUDA 12.4 con PyTorch 2.6.0; Hunyuan3D-2.1 requiere CUDA 12.4 con
PyTorch 2.5.1; HY-World requiere CUDA 12.8. **Unificarlos en un solo entorno es un error
conocido**: produce conflictos de dependencias sin aportar nada.

Se construye una imagen por adapter, sobre una base común con CUDA, Python y utilidades
compartidas. Cada `ModelSpec` nombra su imagen y el backend levanta la que corresponda.

La imagen es el artefacto común entre backends: RunPod, EC2 y SageMaker ejecutan
contenedores. SageMaker impone además su propio contrato (`/opt/ml`, `/ping`,
`/invocations`), que se resuelve con un entrypoint distinto sobre la misma imagen base.

## 7. Modelo de referencia: TRELLIS.2

### 7.1 Requisitos

Linux (única plataforma testeada por el repositorio), GPU NVIDIA con 24 GB mínimo,
CUDA Toolkit 12.4, Conda, Python 3.8 o superior, PyTorch 2.6.0 con cu124. Backend de
atención `flash-attn` por defecto, con `ATTN_BACKEND=xformers` como alternativa.

`setup.sh` compila desde fuente: `flash-attn`, `nvdiffrast`, `nvdiffrec`, `cumesh`,
`o-voxel` y `flexgemm`. El repositorio no incluye Dockerfile.

Pesos: 16,24 GB en nueve archivos safetensors. Espacio de disco recomendado, incluyendo
entorno y artefactos de compilación: 100 GB.

### 7.2 Interfaz

`pipeline.run()` acepta una imagen PIL y, opcionalmente, `num_samples`, `seed`,
`pipeline_type`, `preprocess_image`, `max_num_tokens` y parámetros de sampler para cada
una de las tres etapas. Devuelve una lista de `MeshWithVoxel` **en memoria, no archivos**.

La exportación es una decisión separada: `o_voxel.postprocess.to_glb()` recibe
`decimation_target`, `texture_size`, `remesh`, `remesh_band` y `remesh_project`. Una misma
generación puede exportarse a varios presupuestos de polígonos sin regenerar. El adapter
expone ambas etapas por separado para aprovecharlo.

`pipeline_type` determina qué checkpoints se cargan:

| Valor | Descarga aproximada |
|---|---|
| `512` | 11 GB |
| `1024` | 11 GB |
| `1024_cascade` | 13,5 GB |
| `1536_cascade` | 16,24 GB |

Las primeras pruebas usan `512`.

**Punto de partida sugerido en 24 GB:** `pipeline_type='512'` y `texture_size=2048`. El
ejemplo oficial usa `texture_size=4096` y `decimation_target=1000000`, valores que en una
GPU de 24 GB tienen riesgo de agotar la memoria.

### 7.3 Primer experimento

Comparar TRELLIS v1 y TRELLIS.2 sobre el mismo conjunto de imágenes, renderizando cada
salida con y sin mapa de entorno. Resuelve la hipótesis de la sección 3.2: si v2 sigue
viéndose peor con HDRI, el puesto 17 refleja calidad real y la elección de modelo debe
revisarse.

## 8. Evaluación

Etapa `evaluate`, independiente del modelo, sobre cualquier GLB producido.

**Métricas objetivas:** triángulos, vértices, cantidad de mallas y materiales, mapas de
material presentes, resolución de texturas, tamaño del archivo, estanqueidad de la malla,
caja contenedora, duración de la generación y pico de VRAM.

**Renderizado controlado:** cada salida se renderiza bajo el mismo conjunto de HDRIs. El
repositorio de TRELLIS.2 incluye ocho (`city`, `courtyard`, `forest`, `interior`, `night`,
`studio`, `sunrise`, `sunset`). Comparar modelos sin fijar la iluminación no produce
conclusiones válidas — es exactamente el error que se sospecha en el 3D Arena.

Un experimento es un config declarativo que cruza N modelos con M imágenes de referencia y
produce una tabla comparable.

## 9. Backends

**Restricción que ordena esta sección: hay créditos de AWS disponibles.** Eso invierte la
comparación de costos que sostenía la versión anterior de este spec. RunPod es más barato
en dólares de lista —unos USD 0,34 la hora contra 1,86 de `g6e.xlarge`— pero con créditos
el costo efectivo de AWS es cero hasta agotarlos, así que AWS va primero y RunPod queda
como plan alternativo.

Hay además una ventaja técnica que no depende de los créditos: `g6e.xlarge` monta una
L40S de 48 GB, mientras que la RTX 4090 de RunPod tiene exactamente los 24 GB que
TRELLIS.2 declara como mínimo. Trabajar con el doble de la memoria requerida elimina el
riesgo de quedarse sin VRAM al subir la resolución o el tamaño de textura.

| Backend | Uso | Notas |
|---|---|---|
| `local` | Modelos que entran en 8 GB, y desarrollo | TripoSG, PartCrafter, UniRig sobre `gaston-pc` |
| `ec2` | **Ciclo de iteración principal** | Cubierto por créditos. `g6e.xlarge` con L40S de 48 GB. Requiere aumento de cuota de instancias G |
| `sagemaker` | Generación por lotes y bajo demanda | Asynchronous Inference. Escala a cero |
| `runpod` | Plan alternativo | Si los créditos se agotan o la cuota de AWS se demora |

**SageMaker:** se usa Asynchronous Inference, no endpoints en tiempo real. Los endpoints
real-time exigen que la inferencia termine en 60 segundos y aceptan payloads de hasta 6 MB;
cargar 16 GB de pesos y generar no entra en ese presupuesto. Async admite payloads de
hasta 1 GB y procesamiento de hasta una hora (`InvocationTimeoutSeconds` hasta 3600
segundos), y escala a cero cuando no hay trabajo.

Instancias con VRAM suficiente: `ml.g5.xlarge` (A10G, 24 GB, en el límite) o
`ml.g6e.xlarge` (L40S, 48 GB).

**Limitación aceptada:** escalar desde cero con 16 GB de pesos implica un arranque en frío
de varios minutos. Es irrelevante para generación por lotes y descarta el uso interactivo.

## 10. Orden de construcción

### La cuota disponible decide el orden

Relevamiento de la cuenta **872154182820** (perfil `macacoai`), el 2026-08-11:

| Cuota | us-east-1 |
|---|---|
| EC2 · G y VT On-Demand | **0 vCPU** |
| EC2 · G y VT Spot | 0 vCPU |
| EC2 · P On-Demand y Spot | 0 vCPU |
| SageMaker · `ml.g6e.xlarge` endpoint | 0 |
| **SageMaker · `ml.g5.xlarge` endpoint** | **1** |

Todo lo de EC2 está en cero, así que el camino cómodo está cerrado hasta que se apruebe un
aumento. La única puerta abierta hoy es un endpoint de SageMaker sobre `ml.g5.xlarge`
—A10G con 24 GB, exactamente el mínimo que declara TRELLIS.2—, y por decisión explícita se
va por ahí en lugar de esperar el trámite.

Eso invierte el argumento de la versión anterior de este documento, que ponía `ec2` antes
que `sagemaker` porque iterar sobre la compilación de las seis extensiones CUDA es más
cómodo con acceso directo a la máquina. El argumento sigue siendo válido; la cuota manda.

1. ~~`core` con las interfaces, el backend `local` y un adapter simulado.~~ **Completado**
   el 2026-08-10: 20 commits, 71 tests, CI en verde.
2. Dockerfile de TRELLIS.2 y adapter, con el backend `sagemaker` sobre Asynchronous
   Inference en `ml.g5.xlarge`.
3. Etapa `evaluate` y el experimento de la sección 7.3.
4. Adapters de TripoSG, PartCrafter y UniRig sobre `local`.
5. Backend `ec2`, si en algún momento se aprueba la cuota y se quiere el ciclo de iteración
   cómodo o instancias más grandes que la A10G.
6. Backend `runpod`, solo si AWS deja de ser viable.

**Consecuencia de trabajar sobre `ml.g5.xlarge`:** 24 GB es el mínimo exacto, no hay
margen. Las primeras corridas van a 512³ con `texture_size=2048`, no con los valores del
ejemplo oficial, que usa 4096 y un `decimation_target` de un millón.

**Lo que no está bloqueado por la cuota:** el Dockerfile. La compilación de `flash-attn`,
`nvdiffrast`, `nvdiffrec`, `cumesh`, `o-voxel` y `flexgemm` desde fuente es la parte más
riesgosa de toda la fase, y se puede validar en `gaston-pc` —que compila e importa bien con
8 GB, aunque no pueda correr inferencia— antes de gastar un minuto de GPU paga.

## 11. Criterios de aceptación

1. Un config declarativo ejecuta un modelo sobre un backend sin código específico del par.
2. Agregar un backend nuevo no modifica ningún archivo de `models/`, y agregar un modelo
   nuevo no modifica ningún archivo de `backends/`.
3. Pedir un modelo cuya VRAM excede la del backend falla antes de aprovisionar recursos.
4. Repetir un config ya ejecutado devuelve el resultado cacheado sin consumir GPU.
5. Toda corrida deja `provenance.json` con seed, revisión de pesos, digest de imagen y SHA
   del repositorio.
6. `teardown()` es idempotente y no deja recursos facturables encendidos.
7. El paso 1 del orden de construcción corre en CI sin GPU.
8. El experimento de la sección 7.3 produce una tabla comparativa y una conclusión
   explícita sobre la hipótesis del renderizado.

## 12. Riesgos

| Riesgo | Mitigación |
|---|---|
| La compilación de las seis extensiones CUDA falla o es irreproducible | Fijar el digest de la imagen. El README sugiere instalar los flags de a uno ante fallas |
| 24 GB es el mínimo exacto en una RTX 4090 | Empezar en `512` con `texture_size=2048` y subir desde ahí |
| La hipótesis del renderizado resulta falsa y TRELLIS.2 es peor de verdad | El experimento 7.3 es barato y ocurre temprano. La arquitectura permite cambiar de modelo sin reescribir |
| Costo de GPU por error de operación | `teardown()` idempotente, caché por contenido y validación previa a aprovisionar |
| El self-hosting cuesta más que pagar por generación | Meshy cuesta unos USD 0,40 por modelo y la API de Tripo unos USD 0,25. Reevaluar tras el paso 3 con datos de costo real |

## 13. Preguntas abiertas

Ninguna bloquea la implementación de los pasos 1 a 3.

- Si el GGUF comunitario de TRELLIS.2 entra en 8 GB. Solo hay reportes de blogs y videos,
  ninguno de Microsoft. De confirmarse, `gaston-pc` cubriría el pipeline completo.
- Calidad de las UV que produce TRELLIS.2. Relevante solo si se piensa pintar a mano encima.
- Motor de juego. Determina el formato final y el presupuesto de polígonos; hasta
  definirlo, glTF y GLB son el formato portable de trabajo.
