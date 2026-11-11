# Circuito C3 — StronglyEntanglingLayers (QAE)

> **Actualización de preprocesamiento:** el notebook usa ahora `RobustScaler(quantile_range=(10, 90))` para evitar que RPM y Fuerza_N dominen la normalización L₂. Los resultados numéricos previos citados aquí corresponden a la escala anterior y deben volver a medirse.

> Contexto técnico **solo del circuito C3**. Notebook: [`Refinamiento_c3.ipynb`](Refinamiento_c3.ipynb) (copia del notebook de C1 adaptada). **Estado: la búsqueda de configuración está escrita pero aún no se ha ejecutado; no hay resultados de C3 refinado.**

## 1. Marco del problema (resumen)
- **Tarea:** detección de fallas en rodamientos (dataset Paderborn) con un Autoencoder Cuántico no supervisado / one-class (solo datos sanos).
- **Datos:** `../../paderborn_features_tratadas.csv`, 59 867 filas × 71 features, **todas sanas**; las anomalías de evaluación son **sintéticas** (`X[:, 10:25] += N(0.4, 0.1)` + renormalizar L2).
- **Pipeline:** split 70/15/15 (`random_state=42`) → `RobustScaler(quantile_range=(10, 90))` (fit solo en train) → zero-padding a 128 → norma L2 → **Amplitude Encoding en 7 qubits**.
- **Autoencoder:** ansatz sobre 7 qubits; los últimos `trash` son *trash*, el resto latentes. Pérdida `1 − E[P(|0⟩^trash)]`; fidelidad `F = P(|0⟩^trash)`; error = `1 − F`.
- **Detección:** score `1 − F`, τ = percentil 95 en sanos (≈ 5 % FPR por construcción), positivo = anomalía.
- **Entorno:** PennyLane 0.45.1, Adam, batch 25, pesos iniciales U[0, 2π].

## 2. Circuito C3
`qml.StronglyEntanglingLayers`, pesos `(n_layers, 7, 3)`: por capa `l`, `Rot(φ,θ,ω)` en cada qubit y una cadena de compuertas entrelazantes `i → (i + r_l) mod 7` con `r_l = (l mod 6) + 1` por defecto. Parámetros = `n_layers × 7 × 3`.

- **C3-base** (referencia, ya medido en el notebook de C1): 2 capas, 1 trash (42 parámetros). Resultado observado en entrenamiento: fidelidad train 76.7 % / val 81.0 %.
- **C3-ref** (en refinamiento): `hacer_ansatz_c3(ranges, imprimitive)` expone dos hiperparámetros estructurales de PennyLane (verificado en `StronglyEntanglingLayers.__init__(weights, wires, ranges=None, imprimitive=CNOT)`):
  - `ranges`: `'default'`, `'ring'` (r=1), `'alt'` (r alterna 1 y 3), `'skip2'` (r=2).
  - `imprimitive`: `'cnot'` o `'cz'`.
- **Nota importante:** con `ranges='default'` y `'cnot'`, C3 es **idéntico** al C1 refinado (Rot + CNOT de alcance variable). Para C1 con 6 capas y 2 trash se obtuvo (3 semillas, test): fidelidad sanos 86.9 %, TPR 97.8 %, FPR 5 %, F1 96.4 %, AUC 0.992. Es el punto de partida de `C3_CONFIG`, no un resultado de C3 por confirmar.

## 3. Búsqueda de configuración (sección 9 del notebook)
Se ejecuta en orden; cada etapa usa el split de **validación** y solo la validación final usa **test**.

| Celda | Etapa | Qué barre | Semillas |
|---|---|---|---|
| 13a | Funciones y criterio | — | — |
| 13b | A | `n_layers ∈ {2,3,4,6,8}` × `trash ∈ {1,2,3}` | 3 |
| 13c | B | Top-2 de A × `ranges` (4) × `imprimitive` (2) | 3 |
| 13d | C | Mejor de A+B × `epochs ∈ {20,40}` × `n_samp ∈ {100,200}` (`lr` opcional) | 5 |
| 13e | Final | Mejor vs C3-base en test (500 sanos + 500 anomalías) | 8 |

**Criterio:** entre las configuraciones que cumplen `RESTRICCIONES` (`fid_min = 80 %` de fidelidad en sanos, `gap_max = 3 pp` brecha train–sanos como control de sobreentrenamiento, `fpr_max = 10 %`) se maximiza `Score = 0.7·F1 + 0.3·Fidelidad_sanos` (`W_FID`).
**Métricas reportadas:** fidelidad (train, sanos, anomalías), brecha, separación, TPR, FPR, precisión, F1 (± std entre semillas), exactitud balanceada, MCC, AUC, AP y TP/FP/TN/FN. El historial se guarda en `busqueda_c3_historial.csv`; gráficas: `c3_confusion_final.png` y `c3_pareto_busqueda.png`.

**Cómo adoptar el resultado:** la celda 13e imprime un bloque `C3_CONFIG = dict(...)`; pegarlo en la celda 3 y re-ejecutar las celdas 3–12 para comparar C3-base vs C3-ref con las métricas completas.

## 4. Estructura relevante del notebook
- Celda 3: `ansatz_c3`/`ws_c3` (base), `hacer_ansatz_c3`, `C3_CONFIG`, `CIRCUITOS = [C3-base, C3-ref]`, `CFG_POR_CIRCUITO`.
- Celda 7: `entrenar_qae` (pérdida vectorizada). Celda 8: `CFG` y entrenamiento.
- Celda 10: `metricas_deteccion` y tabla de evaluación. Celda 11b: matrices de confusión, ROC y PR.
- Celdas 13a–13e: búsqueda en etapas.

## 5. Hallazgos previos aplicables (de C1)
- Más capas con 1 trash sube la fidelidad pero baja la detección; `trash = 2` mejora mucho la separación.
- Alta varianza entre semillas → se promedia sobre ≥ 3 semillas y se reporta la desviación del F1.

## 6. Resultados
_Pendiente: completar tras ejecutar la búsqueda (configuración ganadora y métricas en test)._

