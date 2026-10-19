import sys
import os

# Añade la ruta raíz al path para asegurar la importación correcta
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocesamiento_api import ClienteEntrada

def test_estructura_datos_cliente():
    """Valida que el esquema de datos de entrada (ClienteEntrada) acepte correctamente los tipos de datos esperados."""
    datos_prueba = {
        "age": 35,
        "job": "management",
        "marital": "married",
        "education": "tertiary",
        "default": "no",
        "balance": 1200.0,
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "day": 15,
        "month": "may",
        "campaign": 1,
        "pdays": -1,
        "previous": 0,
        "poutcome": "unknown"
    }
    
    # Validar que Pydantic procese los datos de los slots/datos sin errores
    cliente = ClienteEntrada(**datos_prueba)
    assert cliente.age == 35
    assert cliente.job == "management"
    assert cliente.balance == 1200.0