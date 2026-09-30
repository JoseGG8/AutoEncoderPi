"""
Análisis de Entropía de Shannon columna por columna
====================================================
Dataset: paderborn_features_tratadas.csv

Objetivo: Identificar qué columnas aportan información valiosa
para el entrenamiento de un modelo de detección de anomalías en
vibraciones de motores.

NOTA IMPORTANTE: Las columnas fft_0..fft_63 están almacenadas en
escala ln(x+1). Esta transformación comprime valores grandes y expande
los cercanos a 0, lo que distorsiona los bins de equal-width y puede
alterar la entropía aparente. Por ello, el script calcula la entropía
en AMBAS escalas:
  - Escala transformada (ln(x+1)) → tal cual están en el CSV.
  - Escala original (expm1)       → revirtiendo la transformación.

Criterio: Columnas con entropía muy baja (~0) contienen poca variabilidad
(posiblemente constantes o casi constantes) y podrían descartarse.
Columnas con entropía alta contienen mayor diversidad de valores.

Método: Discretización por equal-width binning (50 bins) y cálculo
de entropía de Shannon sobre la distribución de frecuencias resultante.
Se normaliza la entropía dividiéndola entre log2(num_bins) para obtener
valores entre 0 y 1.

NO modifica el dataset en ningún momento.
"""

import os
import sys
import time
import numpy as np
import pandas as pd

# ─────────────────────────── Encoding fix (Windows) ──────────────────
sys.stdout.reconfigure(encoding="utf-8")

# ─────────────────────────── Configuración ───────────────────────────
CSV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "paderborn_features_tratadas.csv",
)
NUM_BINS = 50  # bins para discretización (buen balance precisión/costo)
UMBRAL_BAJA_ENTROPIA = 0.15  # entropía normalizada por debajo = "baja"
FFT_PREFIX = "fft_"  # prefijo de columnas FFT (escala ln(x+1))

# ─────────────────────────── Funciones ───────────────────────────────

def entropia_shannon(serie: pd.Series, num_bins: int = NUM_BINS) -> dict:
    """
    Calcula la entropía de Shannon normalizada para una serie numérica.

    Returns
    -------
    dict con claves: entropia_raw, entropia_norm, num_unicos, min, max, std
    """
    serie_limpia = serie.dropna()
    n = len(serie_limpia)

    if n == 0:
        return {
            "entropia_raw": 0.0,
            "entropia_norm": 0.0,
            "num_unicos": 0,
            "min": np.nan,
            "max": np.nan,
            "std": 0.0,
            "pct_nulos": 100.0,
        }

    num_unicos = serie_limpia.nunique()

    # Si la columna es constante, entropía = 0
    if num_unicos <= 1:
        return {
            "entropia_raw": 0.0,
            "entropia_norm": 0.0,
            "num_unicos": num_unicos,
            "min": serie_limpia.min(),
            "max": serie_limpia.max(),
            "std": 0.0,
            "pct_nulos": (1 - n / len(serie)) * 100,
        }

    # Discretizar en bins (equal-width)
    bins_reales = min(num_bins, num_unicos)
    counts, _ = np.histogram(serie_limpia.values, bins=bins_reales)

    # Probabilidades (solo bins no vacíos)
    probs = counts[counts > 0] / n

    # Entropía de Shannon: -Σ p·log2(p)
    entropia_raw = -np.sum(probs * np.log2(probs))
    entropia_max = np.log2(bins_reales) if bins_reales > 1 else 1.0
    entropia_norm = entropia_raw / entropia_max

    return {
        "entropia_raw": round(entropia_raw, 6),
        "entropia_norm": round(entropia_norm, 6),
        "num_unicos": num_unicos,
        "min": round(serie_limpia.min(), 6),
        "max": round(serie_limpia.max(), 6),
        "std": round(serie_limpia.std(), 6),
        "pct_nulos": round((1 - n / len(serie)) * 100, 2),
    }


def barra_ascii(valor: float, ancho: int = 30) -> str:
    """Genera una barra visual ASCII para un valor entre 0 y 1."""
    llenos = int(valor * ancho)
    return "█" * llenos + "░" * (ancho - llenos)


# ─────────────────────────── Main ────────────────────────────────────

def main():
    print("=" * 80)
    print("  ANÁLISIS DE ENTROPÍA DE SHANNON — COLUMNA POR COLUMNA")
    print("  Dataset: paderborn_features_tratadas.csv")
    print("=" * 80)

    # Cargar dataset
    t0 = time.time()
    print(f"\n⏳ Cargando dataset desde:\n   {CSV_PATH}")

    if not os.path.exists(CSV_PATH):
        print(f"\n❌ ERROR: No se encontró el archivo en:\n   {CSV_PATH}")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    t_carga = time.time() - t0
    print(f"   ✅ Cargado en {t_carga:.2f}s — {df.shape[0]:,} filas × {df.shape[1]} columnas")

    # Separar Label del análisis (es la variable objetivo, no un feature)
    columnas_features = [c for c in df.columns if c != "Label"]
    n_features = len(columnas_features)

    columnas_fft = [c for c in columnas_features if c.startswith(FFT_PREFIX)]
    columnas_no_fft = [c for c in columnas_features if not c.startswith(FFT_PREFIX)]

    print(f"\n📊 Analizando entropía de {n_features} features (excluyendo Label)...")
    print(f"   Columnas FFT (escala ln(x+1)): {len(columnas_fft)}")
    print(f"   Columnas no-FFT: {len(columnas_no_fft)}")
    print(f"   Bins de discretización: {NUM_BINS}")
    print(f"   Umbral baja entropía: {UMBRAL_BAJA_ENTROPIA}")

    # ─── Calcular entropía por columna (escala del CSV) ───
    t0 = time.time()
    resultados = []
    for col in columnas_features:
        info = entropia_shannon(df[col])
        info["columna"] = col
        info["es_fft"] = col.startswith(FFT_PREFIX)
        resultados.append(info)

    df_resultados = pd.DataFrame(resultados)
    df_resultados = df_resultados.sort_values("entropia_norm", ascending=False).reset_index(drop=True)
    t_analisis = time.time() - t0
    print(f"   ✅ Análisis completado en {t_analisis:.2f}s\n")

    # ─── Tabla principal (escala del CSV) ───
    print("─" * 105)
    print(f"{'#':>3}  {'COLUMNA':<20} {'ESCALA':<8}  {'ENT.NORM':>8}  {'ENT.RAW':>8}  "
          f"{'ÚNICOS':>7}  {'STD':>12}  {'BARRA':>30}")
    print("─" * 105)

    for i, row in df_resultados.iterrows():
        escala = "ln(x+1)" if row["es_fft"] else "lineal"
        marca = "⚠️" if row["entropia_norm"] < UMBRAL_BAJA_ENTROPIA else "  "
        print(
            f"{i+1:>3}  {row['columna']:<20} {escala:<8}  {row['entropia_norm']:>8.4f}  "
            f"{row['entropia_raw']:>8.4f}  {row['num_unicos']:>7}  "
            f"{row['std']:>12.6f}  {barra_ascii(row['entropia_norm'])} {marca}"
        )

    print("─" * 105)

    # ─── Resumen por categorías ───
    baja = df_resultados[df_resultados["entropia_norm"] < UMBRAL_BAJA_ENTROPIA]
    media = df_resultados[
        (df_resultados["entropia_norm"] >= UMBRAL_BAJA_ENTROPIA)
        & (df_resultados["entropia_norm"] < 0.6)
    ]
    alta = df_resultados[df_resultados["entropia_norm"] >= 0.6]

    print(f"\n{'=' * 80}")
    print("  RESUMEN (escala del CSV)")
    print(f"{'=' * 80}")
    print(f"\n  🔴 ENTROPÍA BAJA  (< {UMBRAL_BAJA_ENTROPIA})  → {len(baja):>3} columnas  "
          f"— Candidatas a eliminación")
    print(f"  🟡 ENTROPÍA MEDIA ({UMBRAL_BAJA_ENTROPIA}–0.6)     → {len(media):>3} columnas  "
          f"— Revisar relevancia")
    print(f"  🟢 ENTROPÍA ALTA  (≥ 0.6)        → {len(alta):>3} columnas  "
          f"— Buena variabilidad")

    if len(baja) > 0:
        print(f"\n  ⚠️  Columnas con baja entropía (candidatas a eliminar):")
        for _, row in baja.iterrows():
            print(f"      • {row['columna']:<20}  H_norm={row['entropia_norm']:.4f}  "
                  f"únicos={row['num_unicos']}  rango=[{row['min']:.4f}, {row['max']:.4f}]")

    # ══════════════════════════════════════════════════════════════════════
    # COMPARACIÓN: entropía en escala ln(x+1) vs escala original (expm1)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 90}")
    print("  COMPARACIÓN FFT: ENTROPÍA ln(x+1) vs ESCALA ORIGINAL (expm1)")
    print(f"{'=' * 90}")
    print(f"  La transformación ln(x+1) comprime valores grandes y expande")
    print(f"  los cercanos a 0. Una caída grande de entropía al revertir")
    print(f"  indica que la distribución original está muy concentrada")
    print(f"  (pocos valores dominan).")
    print()

    comparacion = []
    for col in columnas_fft:
        h_ln = entropia_shannon(df[col])
        # Revertir ln(x+1) → escala original usando expm1
        serie_original = np.expm1(df[col].dropna())
        h_orig = entropia_shannon(serie_original)
        delta = h_orig["entropia_norm"] - h_ln["entropia_norm"]
        comparacion.append({
            "columna": col,
            "H_ln": h_ln["entropia_norm"],
            "H_orig": h_orig["entropia_norm"],
            "delta": delta,
            "std_ln": h_ln["std"],
            "std_orig": h_orig["std"],
        })

    df_comp = pd.DataFrame(comparacion)
    # Ordenar por delta (más negativo = más distorsión por la transformación)
    df_comp = df_comp.sort_values("delta", ascending=True).reset_index(drop=True)

    print(f"{'─' * 90}")
    print(f"{'#':>3}  {'COLUMNA':<10}  {'H ln(x+1)':>9}  {'H original':>10}  "
          f"{'DELTA':>7}  {'STD ln':>10}  {'STD orig':>10}  {'IMPACTO'}")
    print(f"{'─' * 90}")

    for i, row in df_comp.iterrows():
        # Indicador visual del impacto de la transformación
        if row["delta"] < -0.10:
            impacto = "🔴 ALTA distorsión"
        elif row["delta"] < -0.03:
            impacto = "🟡 Moderada"
        else:
            impacto = "🟢 Mínima"

        print(
            f"{i+1:>3}  {row['columna']:<10}  {row['H_ln']:>9.4f}  {row['H_orig']:>10.4f}  "
            f"{row['delta']:>+7.4f}  {row['std_ln']:>10.6f}  {row['std_orig']:>10.6f}  {impacto}"
        )

    print(f"{'─' * 90}")

    # Resumen de la comparación
    alta_distorsion = df_comp[df_comp["delta"] < -0.10]
    mod_distorsion = df_comp[(df_comp["delta"] >= -0.10) & (df_comp["delta"] < -0.03)]
    baja_distorsion = df_comp[df_comp["delta"] >= -0.03]

    print(f"\n  📊 Resumen de distorsión por ln(x+1):")
    print(f"     🔴 Alta distorsión (Δ < -0.10): {len(alta_distorsion):>3} columnas")
    print(f"     🟡 Moderada (-0.10 ≤ Δ < -0.03): {len(mod_distorsion):>3} columnas")
    print(f"     🟢 Mínima (Δ ≥ -0.03):          {len(baja_distorsion):>3} columnas")

    if len(alta_distorsion) > 0:
        print(f"\n  ⚠️  Columnas FFT con ALTA distorsión por ln(x+1):")
        for _, row in alta_distorsion.iterrows():
            print(f"      • {row['columna']:<10}  H_ln={row['H_ln']:.4f} → H_orig={row['H_orig']:.4f}  "
                  f"(Δ={row['delta']:+.4f})")

    # ─── Análisis de Label (informativo) ───
    if "Label" in df.columns:
        print(f"\n{'─' * 80}")
        print("  📌 DISTRIBUCIÓN DE LABEL (variable objetivo)")
        print(f"{'─' * 80}")
        dist = df["Label"].value_counts().sort_index()
        total = len(df)
        for val, count in dist.items():
            pct = count / total * 100
            print(f"      Label={val}:  {count:>7,} muestras  ({pct:5.1f}%)  "
                  f"{barra_ascii(pct / 100, ancho=20)}")

    # ─── Estadísticas de entropía ───
    print(f"\n{'─' * 80}")
    print("  📈 ESTADÍSTICAS DE ENTROPÍA NORMALIZADA")
    print(f"{'─' * 80}")
    ent_vals = df_resultados["entropia_norm"]
    print(f"      Media:    {ent_vals.mean():.4f}")
    print(f"      Mediana:  {ent_vals.median():.4f}")
    print(f"      Mín:      {ent_vals.min():.4f}  ({df_resultados.iloc[-1]['columna']})")
    print(f"      Máx:      {ent_vals.max():.4f}  ({df_resultados.iloc[0]['columna']})")
    print(f"      Std:      {ent_vals.std():.4f}")

    print(f"\n{'=' * 80}")
    print(f"  ✅ Análisis finalizado — Tiempo total: {t_carga + t_analisis:.2f}s")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
