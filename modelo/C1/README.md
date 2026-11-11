# Circuito C1 — Rot + CNOT de alcance variable (QAE)

> **Actualización de preprocesamiento:** el notebook usa ahora `RobustScaler(quantile_range=(10, 90))` para evitar que RPM y Fuerza_N dominen la normalización L₂. Los resultados numéricos de este README fueron obtenidos con la escala anterior y deben volver a medirse.

> Contexto técnico **solo del circuito C1**. Notebook: [`Refinamiento_c1.ipynb`](Refinamiento_c1.ipynb). Los datos y el pipeline se comparten con el resto del proyecto (ver sección 1).

## 1. Marco del problema (resumen)
- **Tarea:** detección de fallas en rodamientos (dataset Paderborn) con un Autoencoder Cuántico entrenado de forma no supervisada / one-class (solo datos sanos).
- **Datos:** `../../paderborn_features_tratadas.csv`, 59 867 filas × 71 features. **Todas son sanas (`Label = 0`)**; las anomalías de evaluación son **sintéticas** (`X[:, 10:25] += N(0.4, 0.1)` y renormalizar L2).
- **Pipeline:** split 70/15/15 (`random_state=42`) → `RobustScaler(quantile_range=(10, 90))` (fit solo en train) → zero-padding a 128 → norma L2 → **Amplitude Encoding en 7 qubits**.
- **Autoencoder:** ansatz U(θ) sobre 7 qubits; los últimos `trash` qubits son los *trash*, el resto son latentes. Pérdida `1 − E[P(|0⟩^trash)]`; fidelidad `F = P(|0⟩^trash)`; error de reconstrucción = `1 − F`.
- **Detección:** score = `1 − F`; umbral τ = percentil 95 del score en sanos (≈ 5 % de FPR por construcción). Positivo = anomalía.
- **Entorno:** PennyLane 0.45.1 (`default.qubit`, backprop), scikit-learn, pandas; Adam lr 0.12, batch 25, 20 épocas, 100 muestras de entrenamiento, pesos iniciales U[0, 2π].

## 2. Definición de C1 (`ansatz_c1`)
Pesos `(n_layers, 7, 3)`. Por capa `l`:
1. `Rot(φ,θ,ω)` en cada qubit (3 parámetros/qubit).
2. CNOT de **alcance variable**: `i → (i + r) mod 7`, con `r = (l mod 6) + 1`. La capa 0 es el anillo de vecinos; las siguientes conectan qubits cada vez más lejanos.

Parámetros = `n_layers × 7 × 3`. **Baseline original** (referencia): `RY+RZ` por qubit + anillo CNOT fijo, 2 capas, 1 trash (28 parámetros).
La configuración vigente se fija en `CFG_POR_CIRCUITO` (celda 8). Estructuralmente, C1 con este patrón equivale a `StronglyEntanglingLayers` de PennyLane con rangos por defecto.

## 3. Hallazgos
### Resultados del laboratorio (celda de la sección 9; 3 semillas [42, 1, 2], 200 muestras sanas + 200 anomalías, split test)
| Config (ep=20, n=100) | Params | Fid sanos | Fid anom | TPR | FPR | Prec | F1 | MCC | AUC | AP |
|---|---|---|---|---|---|---|---|---|---|---|
| RY+RZ/anillo, 2 capas, 1 trash (baseline) | 28 | 91.3 % | 72.1 % | 62.5 % | 5.0 % | 90.3 % | 70.5 % | 0.609 | 0.946 | 0.924 |
| Rot/stride, 4 capas, 1 trash | 84 | 93.2 % | 76.2 % | 35.3 % | 5.0 % | 49.9 % | 36.1 % | 0.271 | 0.814 | 0.742 |
| Rot/stride, 4 capas, 2 trash | 84 | 80.0 % | 35.8 % | 88.0 % | 5.0 % | 94.4 % | 90.3 % | 0.841 | 0.984 | 0.969 |
| **Rot/stride, 6 capas, 2 trash** | 126 | **86.9 %** | 47.1 % | **97.8 %** | 5.0 % | 95.1 % | **96.4 %** | **0.929** | **0.992** | 0.990 |

Brecha train–val ≈ 0 (|Δ| < 0.6 pp) en todas las configuraciones: **sin sobreentrenamiento**. Matriz de confusión de la mejor (6 capas, 2 trash): TP/FP/TN/FN = 196/10/190/4.

### Conclusiones
- **Más capas con 1 trash qubit sube la fidelidad pero baja la detección:** el circuito aprende a comprimir también las anomalías.
- **`trash = 2` es la palanca principal** (separación sanos–anomalías de ~17–19 pp a ~40–44 pp); añadir capas con `trash = 2` mejora tanto fidelidad como detección (4→6 capas: fidelidad 80.0→86.9 %, TPR 88→97.8 %).
- **Configuración recomendada actual: 6 capas, 2 trash, Rot/stride** (fidelidad 86.9 %, TPR 97.8 %, FPR 5 %).
- Resultados con alta varianza entre semillas (≈80 pasos de Adam); usar ≥ 3 semillas.
- Antes de fijar el hallazgo conviene validar la selección en un split distinto al usado para decidir.

## 4. Estructura del notebook
- Celda 3: `ansatz_c1`, `ws_c1`, registro `CIRCUITOS`.
- Celda 7: `entrenar_qae` (pérdida vectorizada por batch).
- Celda 8: `CFG` y `CFG_POR_CIRCUITO`.
- Celda 10: evaluación con `metricas_deteccion` (TP/FP/TN/FN, TPR, FPR, TNR, FNR, precisión, F1, exactitud, exactitud balanceada, MCC, AUC-ROC, AP). Celda 11b: matrices de confusión, ROC y PR.
- Sección 9: laboratorio con `PARAMETROS` (`n_layers`, `trash`, `epochs`, `lr`, `batch`, `n_samp`, `rot` = `rot|ryrz|ryrzry`, `ent` = `ring|stride|ring+stride`), `SEMILLAS` y `BARRIDO`.

## 5. Pendiente
Más épocas / registros de entrenamiento, `lr` menor, otras puertas (CRZ, IsingZZ) y evaluación con anomalías no sintéticas cuando existan.
