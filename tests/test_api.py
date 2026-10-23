import sys
import os

# Añade la ruta raíz al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from fastapi.testclient import TestClient
from src.api import app

CLIENTE_VALIDO = {"poutcome": "success", "month": "oct", "contact": "cellular", "housing": "no", "job": "management"}


@pytest.fixture(scope="module")
def client():
    """Crea el cliente de pruebas; el 'with' ejecuta la carga del modelo al arrancar."""
    with TestClient(app) as c:
        yield c

def test_endpoint_salud(client):
    """Verifica que el endpoint /salud responda con el estado correcto."""
    response = client.get("/salud")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["modelo_cargado"] is True
    assert "tipo_modelo" in data
    assert data["variables"] == ["poutcome", "month", "contact", "housing", "job"]

def test_predecir_responde_si_o_no(client):
    """Verifica que una solicitud válida devuelva 'sí' o 'no' según el umbral."""
    response = client.post("/predecir", json=CLIENTE_VALIDO)
    assert response.status_code == 200
    data = response.json()
    assert data["respuesta"] in ["sí", "no"]
    assert data["predice_suscripcion"] == (data["puntaje"] >= data["umbral"])

def test_predecir_datos_vacios(client):
    """Verifica que el endpoint /predecir devuelva error 400 si falta información."""
    response = client.post("/predecir", json={})
    assert response.status_code == 400
    assert "Falta el campo 'job'." in response.json()["detalles"]

def test_predecir_valor_fuera_de_opciones(client):
    """Un valor que no existe en el dataset (por ejemplo 'octubre') devuelve 400."""
    response = client.post("/predecir", json={**CLIENTE_VALIDO, "month": "octubre"})
    assert response.status_code == 400

def test_predecir_rechaza_duration(client):
    """'duration' se conoce después de la llamada (fuga): la API no la acepta."""
    response = client.post("/predecir", json={**CLIENTE_VALIDO, "duration": 250})
    assert response.status_code == 400
    assert "no está permitido" in response.json()["detalles"][0]
