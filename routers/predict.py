# routers/predict.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from starlette.responses import FileResponse
from starlette.background import BackgroundTask
import pandas as pd
import joblib
import tempfile
import os

# --- Configuración del router ---
router = APIRouter(prefix="/api", tags=["Predicción desde CSV"])

# === Ruta del modelo entrenado ===
MODEL_PATH = os.path.join("models", "gradientboost_exoplanets.pkl")

# === Cargar modelo entrenado con joblib ===
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"No se pudo cargar el modelo: {e}")

# === Mapeo de nombres del CSV a las columnas esperadas por el modelo ===
FEATURE_COLS = {
    "object_of_interest_name": "koi_score",
    "disposition_score": "koi_score",
    "false_positive_not_transit": "koi_fpflag_nt",
    "false_positive_stellar_eclipse": "koi_fpflag_ss",
    "false_positive_contamination": "koi_fpflag_co",
    "false_positive_ephemeris_match": "koi_fpflag_ec",
    "orbital_perod": "koi_period",
    "impact_parameter": "koi_impact",
    "transit_duration": "koi_duration",
    "transit_depth": "koi_depth",
    "planet_radius": "koi_prad",
    "equilibrium_temperature": "koi_teq",
    "incident_flux": "koi_insol",
    "model_signal_to_radio": "koi_model_snr",
    "effective_temperature": "koi_steff",
    "surface_gravity": "koi_slogg",
    "stellar_radius": "koi_srad",
}


@router.post("/predict/from_csv")
async def predict_from_csv(file: UploadFile = File(...)):
    """
    Recibe un archivo CSV con datos de estrellas y devuelve un nuevo CSV
    con las predicciones generadas por el modelo (una por fila).
    """
    try:
        # Validar formato del archivo
        if not file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="El archivo debe ser un CSV válido.")

        # Leer CSV
        df = pd.read_csv(file.file)

        # === Aplicar mapeo de columnas ===
        # Renombra las columnas del CSV si coinciden con las del diccionario FEATURE_COLS
        df.rename(columns={k: v for k, v in FEATURE_COLS.items() if k in df.columns}, inplace=True)

        # === Verificar que el modelo contenga el 'imputer' ===
        try:
            feature_names = model.named_steps['imputer'].feature_names_in_
        except Exception:
            raise HTTPException(
                status_code=500,
                detail="El modelo cargado no contiene el paso 'imputer' en su pipeline."
            )

        # === Validar que todas las columnas requeridas estén presentes ===
        missing = [c for c in feature_names if c not in df.columns]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Faltan columnas requeridas: {', '.join(missing)}"
            )

        # === Generar predicciones ===
        X_pred = df[list(feature_names)].copy()
        y_pred_proba = model.predict_proba(X_pred)[:, 1]
        y_pred = model.predict(X_pred)

        # === Agregar resultados al DataFrame original ===
        df["prob_confirme_planeta"] = y_pred_proba
        df["prediccion"] = y_pred

        # === Guardar resultados a un archivo temporal ===
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        df.to_csv(tmp_file.name, index=False)

        # === Retornar el CSV generado como respuesta ===
        return FileResponse(
            tmp_file.name,
            media_type="text/csv",
            filename="predicciones_exoplanetas.csv",
            background=BackgroundTask(lambda p=tmp_file.name: os.remove(p) if os.path.exists(p) else None)
        )

    # --- Manejo de errores ---
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {e}")
    finally:
        file.file.close()
