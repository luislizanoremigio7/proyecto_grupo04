import sys
import os

# Añade la ruta raíz al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_endpoint_salud():
    """Verifica que el endpoint /salud responda con el estado correcto."""
    response = client.get("/salud")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "modelo_cargado" in data
    assert "tipo_modelo" in data

def test_predecir_datos_vacios():
    """Verifica que el endpoint /predecir devuelva error de validación (422) si falta información."""
    response = client.post("/predecir", json={})
    assert response.status_code == 422