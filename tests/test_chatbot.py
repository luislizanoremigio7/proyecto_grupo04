import sys
import os

# Añade la ruta raíz al path para evitar errores de importación
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Nota: Cambia 'chatbot' por 'chatbot_2' si tu archivo se llama así en la carpeta src/
from src.chatbot import MAPEO_OPCIONES, PREGUNTAS_SLOTS

def test_mapeo_ocupacion_chatbot():
    """Valida las traducciones correctas de las ocupaciones en el chatbot."""
    assert MAPEO_OPCIONES["job"]["administrador"] == "admin."
    assert MAPEO_OPCIONES["job"]["obrero"] == "blue-collar"
    assert MAPEO_OPCIONES["job"]["estudiante"] == "student"

def test_mapeo_estado_civil_chatbot():
    """Valida la traducción del estado civil."""
    assert MAPEO_OPCIONES["marital"]["casado"] == "married"
    assert MAPEO_OPCIONES["marital"]["soltero"] == "single"

def test_mapeo_yes_no_chatbot():
    """Valida las respuestas de afirmación y negación."""
    assert MAPEO_OPCIONES["yes_no"]["si"] == "yes"
    assert MAPEO_OPCIONES["yes_no"]["no"] == "no"

def test_cantidad_y_claves_slots():
    """Verifica que existan exactamente los 15 slots requeridos por el modelo."""
    assert len(PREGUNTAS_SLOTS) == 15
    claves = [slot["clave"] for slot in PREGUNTAS_SLOTS]
    assert "age" in claves
    assert "job" in claves
    assert "balance" in claves
    assert "duration" not in claves  # Como se maneja internamente, no debe estar en los slots del usuario