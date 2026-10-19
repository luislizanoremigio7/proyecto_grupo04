import sys
import requests
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

console = Console()
API_URL = "http://127.0.0.1:8000/predecir"

# Mapeo español -> API
MAPEO_OPCIONES = {
    "job": {
        "administrador": "admin.", "administrativo": "admin.", "admin": "admin.",
        "obrero": "blue-collar", "manual": "blue-collar", "operario": "blue-collar",
        "empresario": "entrepreneur", "emprendedor": "entrepreneur",
        "empleada domestica": "housemaid", "empleada doméstica": "housemaid", "domestico": "housemaid",
        "gerente": "management", "gerencia": "management", "director": "management",
        "jubilado": "retired", "pensionista": "retired",
        "autonomo": "self-employed", "autónomo": "self-employed", "independiente": "self-employed",
        "servicios": "services",
        "estudiante": "student",
        "tecnico": "technician", "técnico": "technician",
        "desempleado": "unemployed", "sin empleo": "unemployed",
        "desconocido": "unknown"
    },
    "marital": {
        "casado": "married", "casada": "married",
        "soltero": "single", "soltera": "single",
        "divorciado": "divorced", "divorciada": "divorced", "viudo": "divorced", "viuda": "divorced"
    },
    "education": {
        "primaria": "primary",
        "secundaria": "secondary",
        "superior": "tertiary", "universitaria": "tertiary", "universitario": "tertiary", "tecnica": "tertiary", "técnica": "tertiary",
        "desconocido": "unknown"
    },
    "yes_no": {
        "si": "yes", "sí": "yes", "s": "yes",
        "no": "no", "n": "no"
    },
    "contact": {
        "celular": "cellular", "movil": "cellular", "móvil": "cellular",
        "telefono": "telephone", "teléfono": "telephone", "fijo": "telephone",
        "desconocido": "unknown"
    },
    "month": {
        "enero": "jan", "ene": "jan", "jan": "jan",
        "febrero": "feb", "feb": "feb",
        "marzo": "mar", "mar": "mar",
        "abril": "apr", "abr": "apr", "apr": "apr",
        "mayo": "may", "may": "may",
        "junio": "jun", "jun": "jun",
        "julio": "jul", "jul": "jul",
        "agosto": "aug", "ago": "aug", "aug": "aug",
        "septiembre": "sep", "setiembre": "sep", "sep": "sep",
        "octubre": "oct", "oct": "oct",
        "noviembre": "nov", "nov": "nov",
        "diciembre": "dec", "dic": "dec", "dec": "dec"
    },
    "poutcome": {
        "exito": "success", "éxito": "success",
        "fallo": "failure", "fracaso": "failure",
        "desconocido": "unknown",
        "otro": "other"
    }
}

# 15 porandu añónte (ndaipóri duration)
PREGUNTAS_SLOTS = [
    {
        "clave": "age",
        "pregunta": "1. ¿Cuál es la EDAD del cliente?",
        "tipo": int, "minimo": 18, "maximo": 120,
        "ayuda": "Ingrese un número entero entre 18 y 120."
    },
    {
        "clave": "job",
        "pregunta": "2. ¿A qué se dedica u OCUPACIÓN principal?",
        "tipo": str, "mapeo": "job",
        "opciones_humanas": "gerente, obrero, técnico, administrativo, servicios, jubilado, independiente, estudiante, desempleado, empresario",
        "ayuda": "Ejemplos: gerente, administrativo, técnico, obrero, servicios, estudiante, etc."
    },
    {
        "clave": "marital",
        "pregunta": "3. ¿Cuál es su ESTADO CIVIL?",
        "tipo": str, "mapeo": "marital",
        "opciones_humanas": "casado, soltero, divorciado",
        "ayuda": "Opciones: casado, soltero o divorciado."
    },
    {
        "clave": "education",
        "pregunta": "4. ¿Cuál es su NIVEL EDUCATIVO?",
        "tipo": str, "mapeo": "education",
        "opciones_humanas": "superior, secundaria, primaria, desconocido",
        "ayuda": "Opciones: superior, secundaria o primaria."
    },
    {
        "clave": "default",
        "pregunta": "5. ¿Tiene reportes de CRÉDITO EN MORA actualmente?",
        "tipo": str, "mapeo": "yes_no",
        "opciones_humanas": "sí, no",
        "ayuda": "Responda 'sí' o 'no'."
    },
    {
        "clave": "balance",
        "pregunta": "6. ¿Cuál es su SALDO PROMEDIO ANUAL en cuenta (€/$)?",
        "tipo": float,
        "ayuda": "Ingrese el monto numérico (ejemplo: 1500)."
    },
    {
        "clave": "housing",
        "pregunta": "7. ¿Cuenta con un CRÉDITO HIPOTECARIO activo?",
        "tipo": str, "mapeo": "yes_no",
        "opciones_humanas": "sí, no",
        "ayuda": "Responda 'sí' o 'no'."
    },
    {
        "clave": "loan",
        "pregunta": "8. ¿Tiene algún PRÉSTAMO PERSONAL vigente?",
        "tipo": str, "mapeo": "yes_no",
        "opciones_humanas": "sí, no",
        "ayuda": "Responda 'sí' o 'no'."
    },
    {
        "clave": "contact",
        "pregunta": "9. ¿Por qué MEDIO DE CONTACTO se le comunicó?",
        "tipo": str, "mapeo": "contact",
        "opciones_humanas": "celular, teléfono, desconocido",
        "ayuda": "Opciones: celular, teléfono o desconocido."
    },
    {
        "clave": "day",
        "pregunta": "10. ¿Qué DÍA del mes fue el último contacto?",
        "tipo": int, "minimo": 1, "maximo": 31,
        "ayuda": "Ingrese un día del 1 al 31."
    },
    {
        "clave": "month",
        "pregunta": "11. ¿En qué MES se realizó el último contacto?",
        "tipo": str, "mapeo": "month",
        "opciones_humanas": "enero, febrero, marzo, abril, mayo, junio, julio, agosto, septiembre, octubre, noviembre, diciembre",
        "ayuda": "Escriba el mes en español (ejemplo: mayo)."
    },
    {
        "clave": "campaign",
        "pregunta": "12. ¿Cuántas veces se le ha CONTACTADO en esta campaña?",
        "tipo": int, "minimo": 1,
        "ayuda": "Ingrese un número entero (mínimo 1)."
    },
    {
        "clave": "pdays",
        "pregunta": "13. ¿DÍAS transcurridos desde la campaña anterior?",
        "tipo": int, "minimo": -1,
        "ayuda": "Ingrese los días o '-1' si es cliente nuevo."
    },
    {
        "clave": "previous",
        "pregunta": "14. Número de CONTACTOS PREVIOS antes de esta campaña:",
        "tipo": int, "minimo": 0,
        "ayuda": "Ingrese 0 si no hubo contactos anteriores."
    },
    {
        "clave": "poutcome",
        "pregunta": "15. ¿Cuál fue el RESULTADO de la campaña anterior?",
        "tipo": str, "mapeo": "poutcome",
        "opciones_humanas": "éxito, fallo, desconocido, otro",
        "ayuda": "Opciones: éxito, fallo, desconocido u otro."
    }
]

def derivar_a_humano(motivo: str):
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

def capturar_slot_con_fallback(config_slot) -> tuple:
    reintentos = 0
    max_reintentos = 3

    while reintentos < max_reintentos:
        entrada = Prompt.ask(f"\n[bold cyan]{config_slot['pregunta']}[/bold cyan]").strip()
        entrada_clean = entrada.lower().strip()

        if entrada_clean in ["humano", "asesor", "persona", "agente"]:
            return None, "SOLICITUD_USUARIO"
        if entrada_clean == "salir":
            console.print("\n[bold red]👋 Proceso terminado.[/bold red]")
            sys.exit(0)

        if "mapeo" in config_slot:
            dicc_tipo = MAPEO_OPCIONES[config_slot["mapeo"]]
            if entrada_clean in dicc_tipo:
                return dicc_tipo[entrada_clean], "OK"
            elif entrada_clean in dicc_tipo.values():
                return entrada_clean, "OK"
            else:
                reintentos += 1
                console.print(f"[bold yellow]⚠️ Opción no reconocida.[/bold yellow] Elija entre: {config_slot['opciones_humanas']}")
                continue

        try:
            valor_procesado = config_slot["tipo"](entrada_clean)
            if config_slot.get("minimo") is not None and valor_procesado < config_slot["minimo"]:
                raise ValueError()
            if config_slot.get("maximo") is not None and valor_procesado > config_slot["maximo"]:
                raise ValueError()
            return valor_procesado, "OK"
        except ValueError:
            reintentos += 1
            console.print(f"[bold yellow]⚠️ Dato inválido.[/bold yellow] {config_slot['ayuda']}")

    return None, "EXCESO_REINTENTOS"

def interpretar_resultado_no_tecnico(data_api: dict):
    probabilidad = data_api.get("probabilidad_suscripcion", 0.0)
    etiqueta = data_api.get("etiqueta", "no")

    console.print("\n" + "=" * 60, style="bold green" if etiqueta == "yes" else "bold yellow")

    if etiqueta == "yes":
        mensaje_claro = (
            f"[bold green]✨ ¡ALTA PROBABILIDAD DE SUSCRIPCIÓN![/bold green]\n\n"
            f"El cliente tiene un [bold]{probabilidad}% de probabilidad[/bold] de aceptar el producto.\n"
            f"💡 [bold]Sugerencia:[/bold] Presentar la tasa promocional e incentivar el cierre."
        )
        estilo_borde = "green"
    else:
        mensaje_claro = (
            f"[bold yellow]⚠️ BAJA PROBABILIDAD DE ACEPTACIÓN[/bold yellow]\n\n"
            f"El cliente tiene un [bold]{probabilidad}% de probabilidad[/bold].\n"
            f"💡 [bold]Sugerencia:[/bold] Ofrecer productos alternativos con liquidez inmediata."
        )
        estilo_borde = "yellow"

    console.print(Panel(mensaje_claro, title="[bold]RECOMENDACIÓN DE NEGOCIO[/bold]", border_style=estilo_borde))
    console.print("=" * 60 + "\n")

def ejecutar_chatbot():
    console.clear()
    console.print(Panel.fit(
        "[bold white]🤖 ASISTENTE VIRTUAL DE EVALUACIÓN COMERCIAL - BANCO MARKETING[/bold white]\n"
        "[dim]Herramienta interactiva para asesores de venta de Depósitos a Plazo[/dim]\n",
        border_style="blue"
    ))

    datos_colectados = {}

    for slot in PREGUNTAS_SLOTS:
        valor, estado = capturar_slot_con_fallback(slot)

        if estado == "SOLICITUD_USUARIO":
            derivar_a_humano("El usuario solicitó atención personalizada.")
            return
        elif estado == "EXCESO_REINTENTOS":
            derivar_a_humano(f"Dificultad al recolectar '{slot['clave']}'.")
            return

        datos_colectados[slot["clave"]] = valor

    # Mbojuapy interno ikatu haguarã API omba'apo hekopete
    datos_colectados["duration"] = 250.0

    console.print("\n[bold green]⏳ Consultando el modelo predictivo en la API...[/bold green]")
    try:
        respuesta = requests.post(API_URL, json=datos_colectados, timeout=10)

        if respuesta.status_code == 200:
            interpretar_resultado_no_tecnico(respuesta.json())
        else:
            console.print(f"\n[bold red]⚠️ Error ({respuesta.status_code}):[/bold red] No se pudo procesar la predicción.")
            derivar_a_humano("Error interno en el backend.")

    except requests.exceptions.ConnectionError:
        console.print("\n[bold red]❌ Error de conexión con la API en http://127.0.0.1:8000/.[/bold red]")
        derivar_a_humano("Servicio no disponible.")

if __name__ == "__main__":
    ejecutar_chatbot()