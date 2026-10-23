import sys
import requests
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

console = Console()
API_URL = "http://127.0.0.1:8000/predecir"
MAX_REINTENTOS = 3

MAPEO_OPCIONES = {
    "job": {
        "administrador": "admin.", "administrativo": "admin.", "admin": "admin.",
        "obrero": "blue-collar", "operario": "blue-collar",
        "empresario": "entrepreneur", "emprendedor": "entrepreneur",
        "trabajadora del hogar": "housemaid", "trabajador del hogar": "housemaid", "empleada domestica": "housemaid", "empleada doméstica": "housemaid",
        "gerente": "management", "gerencia": "management", "jefe": "management",
        "jubilado": "retired", "jubilada": "retired",
        "independiente": "self-employed", "autonomo": "self-employed", "autónomo": "self-employed",
        "servicios": "services",
        "estudiante": "student",
        "tecnico": "technician", "técnico": "technician",
        "desempleado": "unemployed", "desempleada": "unemployed",
        "desconocido": "unknown"
    },
    "month": {
        "enero": "jan", "febrero": "feb", "marzo": "mar", "abril": "apr", "mayo": "may", "junio": "jun",
        "julio": "jul", "agosto": "aug", "septiembre": "sep", "setiembre": "sep", "octubre": "oct",
        "noviembre": "nov", "diciembre": "dec",
        "1": "jan", "2": "feb", "3": "mar", "4": "apr", "5": "may", "6": "jun",
        "7": "jul", "8": "aug", "9": "sep", "10": "oct", "11": "nov", "12": "dec"
    },
    "yes_no": {
        "si": "yes", "sí": "yes", "s": "yes", "yes": "yes",
        "no": "no", "n": "no"
    },
    "contact": {
        "celular": "cellular", "movil": "cellular", "móvil": "cellular",
        "telefono": "telephone", "teléfono": "telephone", "fijo": "telephone",
        "desconocido": "unknown"
    },
    "poutcome": {
        "exito": "success", "éxito": "success", "exitoso": "success",
        "fallo": "failure", "fracaso": "failure", "fallido": "failure",
        "desconocido": "unknown",
        "otro": "other"
    }
}

PREGUNTAS_SLOTS = [
    {
        "clave": "poutcome",
        "pregunta": "1. ¿Cuál fue el RESULTADO de la campaña anterior?",
        "tipo": str, "mapeo": "poutcome",
        "opciones_humanas": "éxito, fallo, desconocido, otro",
        "ayuda": "Opciones válidas: éxito, fallo, desconocido (si no hubo campaña anterior) u otro."
    },
    {
        "clave": "month",
        "pregunta": "2. ¿En qué MES se realizará la llamada?",
        "tipo": str, "mapeo": "month",
        "opciones_humanas": "enero ... diciembre, o su número",
        "ayuda": "Escriba el nombre del mes (por ejemplo: octubre) o su número (10)."
    },
    {
        "clave": "contact",
        "pregunta": "3. ¿Por qué MEDIO DE CONTACTO se le comunicará?",
        "tipo": str, "mapeo": "contact",
        "opciones_humanas": "celular, teléfono, desconocido",
        "ayuda": "Opciones válidas: celular, teléfono o desconocido."
    },
    {
        "clave": "housing",
        "pregunta": "4. ¿Cuenta el cliente con un CRÉDITO HIPOTECARIO activo?",
        "tipo": str, "mapeo": "yes_no",
        "opciones_humanas": "sí, no",
        "ayuda": "Por favor responda únicamente 'sí' o 'no'."
    },
    {
        "clave": "job",
        "pregunta": "5. ¿Cuál es la OCUPACIÓN del cliente?",
        "tipo": str, "mapeo": "job",
        "opciones_humanas": "administrativo, obrero, gerente, técnico, servicios, jubilado, estudiante, ...",
        "ayuda": "Opciones: administrativo, obrero, empresario, trabajador del hogar, gerente, jubilado, "
                 "independiente, servicios, estudiante, técnico, desempleado o desconocido."
    }
]

def derivar_a_humano(motivo: str):
    """
    Gestiona la derivación de la sesión actual hacia un asesor humano 
    cuando se detecta una solicitud explícita o un fallo crítico.
    """
    console.print("\n" + "=" * 60, style="bold yellow")
    console.print(Panel(
        f"[bold red]📞 DERIVACIÓN A ASESOR COMERCIAL HUMANO[/bold red]\n\n"
        f"[bold white]Motivo:[/bold white] {motivo}\n"
        f"[bold yellow]Acción:[/bold yellow] Transfiriendo la sesión a un ejecutivo de Ventas...\n"
        f"[dim]Anexo interno: #4402[/dim]",
        title="[bold yellow]TRANSFERENCIA DE ATENCIÓN[/bold yellow]",
        border_style="yellow"
    ))
    console.print("=" * 60 + "\n", style="bold yellow")

def capturar_slot_interactivo(config_slot, datos_acumulados: dict):
    """
    Captura, valida y mapea la entrada del usuario para un slot específico,
    gestionando los comandos globales y un máximo de 3 intentos por pregunta.
    """
    intentos_fallidos = 0
    while True:
        if intentos_fallidos >= MAX_REINTENTOS:
            return None, "MAX_REINTENTOS"
        if "opciones_humanas" in config_slot:
            prompt_txt = f"\n[bold cyan]{config_slot['pregunta']}[/bold cyan] [dim]({config_slot['opciones_humanas']})[/dim]"
        else:
            prompt_txt = f"\n[bold cyan]{config_slot['pregunta']}[/bold cyan]"

        entrada = Prompt.ask(prompt_txt).strip()
        entrada_clean = entrada.lower().strip()

        if entrada_clean in ["humano", "asesor", "persona", "agente"]:
            return None, "SOLICITUD_USUARIO"
        if entrada_clean == "salir":
            console.print("\n[bold red]👋 Proceso terminado.[/bold red]")
            sys.exit(0)

        if "mapeo" in config_slot:
            dicc_tipo = MAPEO_OPCIONES[config_slot["mapeo"]]
            val_mapeado = dicc_tipo.get(entrada_clean) or (entrada_clean if entrada_clean in dicc_tipo.values() else None)

            if val_mapeado:
                return val_mapeado, "OK"
            else:
                intentos_fallidos += 1
                console.print(f"[bold yellow]⚠️ Entrada no comprendida (intento {intentos_fallidos} de {MAX_REINTENTOS}).[/bold yellow] {config_slot['ayuda']}")
                continue

        try:
            valor_procesado = config_slot["tipo"](entrada_clean)
            if config_slot.get("minimo") is not None and valor_procesado < config_slot["minimo"]:
                console.print(f"[bold yellow]⚠️ El valor debe ser mayor o igual a {config_slot['minimo']}.[/bold yellow]")
                continue
            if config_slot.get("maximo") is not None and valor_procesado > config_slot["maximo"]:
                console.print(f"[bold yellow]⚠️ El valor debe ser menor o igual a {config_slot['maximo']}.[/bold yellow]")
                continue

            return valor_procesado, "OK"
        except ValueError:
            console.print(f"[bold yellow]⚠️ Entrada no válida.[/bold yellow] {config_slot['ayuda']}")

def construir_payload_completo(datos_colectados: dict) -> dict:
    """
    Construye el JSON para la API solo con los 5 datos que respondió el usuario.
    No se completan valores supuestos ni se envía 'duration' (dato del futuro).
    """
    return {slot["clave"]: datos_colectados[slot["clave"]] for slot in PREGUNTAS_SLOTS}

def interpretar_resultado_no_tecnico(data_api: dict):
    """
    Procesa la respuesta JSON de la API y muestra una recomendación comercial
    comprensible (sí o no) orientada a usuarios no técnicos.
    """
    llamar = data_api.get("respuesta") == "sí"

    console.print("\n" + "=" * 60, style="bold green" if llamar else "bold yellow")

    if llamar:
        mensaje_claro = (
            f"[bold green]✅ SÍ, CONVIENE LLAMAR A ESTE CLIENTE[/bold green]\n\n"
            f"💡 [bold]Sugerencia:[/bold] Inclúyalo en su lista de llamadas y presente la tasa promocional."
        )
        estilo_borde = "green"
    else:
        mensaje_claro = (
            f"[bold yellow]❌ NO, NO PRIORIZAR POR AHORA[/bold yellow]\n\n"
            f"💡 [bold]Sugerencia:[/bold] Atienda primero a otros clientes de su lista."
        )
        estilo_borde = "yellow"

    console.print(Panel(mensaje_claro, title="[bold]RECOMENDACIÓN DE NEGOCIO[/bold]", border_style=estilo_borde))
    console.print("=" * 60 + "\n")

def ejecutar_chatbot():
    """
    Controlador principal del flujo conversacional del chatbot, encargado de orquestar 
    la recolección de datos, la consulta al backend y la presentación de resultados.
    """
    console.clear()
    console.print(Panel.fit(
        "[bold white]🤖 ASISTENTE VIRTUAL DE EVALUACIÓN COMERCIAL - BANCO MARKETING[/bold white]\n"
        "[dim]Herramienta interactiva para asesores de venta de Depósitos a Plazo.[/dim]\n",
        border_style="blue"
    ))

    datos_colectados = {}

    for slot in PREGUNTAS_SLOTS:
        valor, estado = capturar_slot_interactivo(slot, datos_colectados)

        if estado == "SOLICITUD_USUARIO":
            derivar_a_humano("El usuario solicitó atención personalizada.")
            return
        if estado == "MAX_REINTENTOS":
            derivar_a_humano(f"No se logró entender la respuesta después de {MAX_REINTENTOS} intentos.")
            return

        datos_colectados[slot["clave"]] = valor

    payload_completo = construir_payload_completo(datos_colectados)

    console.print("\n[bold green]⏳ Consultando el modelo predictivo en la API...[/bold green]")
    try:
        respuesta = requests.post(API_URL, json=payload_completo, timeout=10)

        if respuesta.status_code == 200:
            interpretar_resultado_no_tecnico(respuesta.json())
        else:
            console.print(f"\n[bold red]⚠️ Error ({respuesta.status_code}):[/bold red] No se pudo procesar la predicción.")
            derivar_a_humano("Error interno en el backend.")

    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        console.print("\n[bold red]❌ Error de conexión con la API en http://127.0.0.1:8000/predecir.[/bold red]")
        derivar_a_humano("Servicio no disponible.")

if __name__ == "__main__":
    ejecutar_chatbot()