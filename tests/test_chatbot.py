import sys
import os

# Añade la ruta raíz al path para evitar errores de importación
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import chatbot
from src.chatbot import MAPEO_OPCIONES, PREGUNTAS_SLOTS, MAX_REINTENTOS
from src.datos import VARIABLES_SELECCIONADAS

def test_mapeo_ocupacion_chatbot():
    """Valida las traducciones correctas de las ocupaciones en el chatbot."""
    assert MAPEO_OPCIONES["job"]["administrador"] == "admin."
    assert MAPEO_OPCIONES["job"]["obrero"] == "blue-collar"
    assert MAPEO_OPCIONES["job"]["estudiante"] == "student"

def test_mapeo_mes_chatbot():
    """Valida la traducción de los meses (nombre o número)."""
    assert MAPEO_OPCIONES["month"]["octubre"] == "oct"
    assert MAPEO_OPCIONES["month"]["5"] == "may"

def test_mapeo_yes_no_chatbot():
    """Valida las respuestas de afirmación y negación."""
    assert MAPEO_OPCIONES["yes_no"]["si"] == "yes"
    assert MAPEO_OPCIONES["yes_no"]["no"] == "no"

def test_cantidad_y_claves_slots():
    """Verifica que los slots sean exactamente las 5 variables que usa el modelo."""
    claves = [slot["clave"] for slot in PREGUNTAS_SLOTS]
    assert len(PREGUNTAS_SLOTS) == 5
    assert sorted(claves) == sorted(VARIABLES_SELECCIONADAS)
    assert "duration" not in claves

def test_payload_sin_valores_inventados_ni_duration():
    """El JSON enviado a la API solo lleva los 5 datos respondidos por el usuario."""
    datos = {"poutcome": "success", "month": "oct", "contact": "cellular", "housing": "no", "job": "retired"}
    payload = chatbot.construir_payload_completo(datos)
    assert payload == datos
    assert "duration" not in payload

def test_fallback_deriva_al_tercer_intento(monkeypatch):
    """Si no entiende la respuesta 3 veces, el chatbot deriva a un asesor humano."""
    respuestas = iter(["banana", "xyz", "???"])
    monkeypatch.setattr(chatbot.Prompt, "ask", lambda *a, **k: next(respuestas))
    valor, estado = chatbot.capturar_slot_interactivo(PREGUNTAS_SLOTS[1], {})
    assert valor is None
    assert estado == "MAX_REINTENTOS"
    assert MAX_REINTENTOS == 3

def test_respuesta_valida_tras_un_error(monkeypatch):
    """Después de un intento fallido, una respuesta válida se acepta."""
    respuestas = iter(["banana", "octubre"])
    monkeypatch.setattr(chatbot.Prompt, "ask", lambda *a, **k: next(respuestas))
    assert chatbot.capturar_slot_interactivo(PREGUNTAS_SLOTS[1], {}) == ("oct", "OK")

def test_pedir_humano_deriva(monkeypatch):
    """Escribir 'humano' en cualquier pregunta pide la derivación."""
    monkeypatch.setattr(chatbot.Prompt, "ask", lambda *a, **k: "humano")
    assert chatbot.capturar_slot_interactivo(PREGUNTAS_SLOTS[0], {}) == (None, "SOLICITUD_USUARIO")
