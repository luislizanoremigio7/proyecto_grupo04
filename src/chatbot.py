import sys
import requests
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

console = Console()
API_URL = "http://127.0.0.1:8000/predecir"

MAPEO_OPCIONES = {
    "education": {
        "primaria": "primary",
        "secundaria": "secondary",
        "superior": "tertiary", "universitaria": "tertiary", "universitario": "tertiary", "tecnica": "tertiary", "técnica": "tertiary",
        "desconocido": "unknown"
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
        "clave": "housing",
        "pregunta": "1. ¿Cuenta el cliente con un CRÉDITO HIPOTECARIO activo?",
        "tipo": str, "mapeo": "yes_no",
        "opciones_humanas": "sí, no",
        "ayuda": "Por favor responda únicamente 'sí' o 'no'."
    },
    {
        "clave": "contact",
        "pregunta": "2. ¿Por qué MEDIO DE CONTACTO se le comunicó?",
        "tipo": str, "mapeo": "contact",
        "opciones_humanas": "celular, teléfono, desconocido",
        "ayuda": "Opciones válidas: celular, teléfono o desconocido."
    },
    {
        "clave": "previous",
        "pregunta": "3. ¿Número de CONTACTOS PREVIOS antes de esta campaña?",
        "tipo": int, "minimo": 0, "maximo": 100,
        "ayuda": "Ingrese un número entero (ejemplo: 0 para cliente nuevo, 1, 2, etc.)."
    },
    {
        "clave": "pdays",
        "pregunta": "4. ¿DÍAS transcurridos desde el último contacto previo?",
        "tipo": int, "minimo": -1, "maximo": 999,
        "ayuda": "Ingrese la cantidad de días transcurridos."
    },
    {
        "clave": "poutcome",
        "pregunta": "5. ¿Cuál fue el RESULTADO de la campaña anterior?",
        "tipo": str, "mapeo": "poutcome",
        "opciones_humanas": "éxito, fallo, desconocido, otro",
        "ayuda": "Opciones válidas: éxito, fallo, desconocido u otro."
    },
    {
        "clave": "education",
        "pregunta": "6. ¿Cuál es su NIVEL EDUCATIVO?",
        "tipo": str, "mapeo": "education",
        "opciones_humanas": "superior, secundaria, primaria, desconocido",
        "ayuda": "Opciones válidas: superior, secundaria, primaria o desconocido."
    },
    {
        "clave": "balance",
        "pregunta": "7. ¿Cuál es su SALDO PROMEDIO ANUAL en cuenta (€/$)?",
        "tipo": float, "minimo": -8100.0, "maximo": 100000.0,
        "ayuda": "Ingrese un monto numérico válido (ejemplo: 1500)."
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
    gestionando la resolución lógica automática y comandos globales.
    """
    if config_slot["clave"] == "pdays" and datos_acumulados.get("previous", 0) == 0:
        console.print("\n[dim]ℹ️ Como indicó 0 contactos previos, se asigna automáticamente 'Sin contacto previo' (-1).[/dim]")
        return -1, "OK"

    if config_slot["clave"] == "poutcome" and datos_acumulados.get("previous", 0) == 0:
        console.print("[dim]ℹ️ Asignando resultado 'desconocido' por ser cliente nuevo.[/dim]")
        return "unknown", "OK"

    while True:
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
                console.print(f"[bold yellow]⚠️ Entrada no comprendida.[/bold yellow] {config_slot['ayuda']}")
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

def construir_payload_completo(datos_7_vars: dict) -> dict:
    """
    Construye el vector de características completo para la API incorporando 
    los valores recolectados en los slots y completando los parámetros neutros restantes.
    """
    return {
        "housing": datos_7_vars.get("housing", "no"),
        "contact": datos_7_vars.get("contact", "cellular"),
        "previous": datos_7_vars.get("previous", 0),
        "pdays": datos_7_vars.get("pdays", -1),
        "poutcome": datos_7_vars.get("poutcome", "unknown"),
        "education": datos_7_vars.get("education", "secondary"),
        "balance": datos_7_vars.get("balance", 0.0),
        "age": 40,
        "job": "management",
        "marital": "married",
        "default": "no",
        "loan": "no",
        "day": 15,
        "month": "may",
        "campaign": 1,
        "duration": 250.0
    }

def interpretar_resultado_no_tecnico(data_api: dict):
    """
    Procesa la respuesta JSON de la API y muestra una recomendación comercial 
    comprensible y orientada a usuarios no técnicos.
    """
    probabilidad = data_api.get("probabilidad", data_api.get("probabilidad_suscripcion", 0.0))
    if probabilidad <= 1.0:
        probabilidad = round(probabilidad * 100, 2)
    
    etiqueta = data_api.get("prediccion", data_api.get("etiqueta", "no"))

    console.print("\n" + "=" * 60, style="bold green" if (etiqueta in [1, "yes"] or probabilidad >= 50.0) else "bold yellow")

    if etiqueta in [1, "yes"] or probabilidad >= 50.0:
        mensaje_claro = (
            f"[bold green]✨ ¡ALTA PROBABILIDAD DE SUSCRIPCIÓN![/bold green]\n\n"
            f"El cliente tiene un [bold]{probabilidad}% de probabilidad[/bold] de aceptar el producto.\n"
            f"💡 [bold]Sugerencia:[/bold] Presentar la tasa promocional e incentivar el cierre de inmediato."
        )
        estilo_borde = "green"
    else:
        mensaje_claro = (
            f"[bold yellow]⚠️ BAJA PROBABILIDAD DE ACEPTACIÓN[/bold yellow]\n\n"
            f"El cliente tiene un [bold]{probabilidad}% de probabilidad[/bold].\n"
            f"💡 [bold]Sugerencia:[/bold] Ofrecer productos alternativos con liquidez inmediata o no insistir."
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

    except requests.exceptions.ConnectionError:
        console.print("\n[bold red]❌ Error de conexión con la API en http://127.0.0.1:8000/predecir.[/bold red]")
        derivar_a_humano("Servicio no disponible.")

if __name__ == "__main__":
    ejecutar_chatbot()