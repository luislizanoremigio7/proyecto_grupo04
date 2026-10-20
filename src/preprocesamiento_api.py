"""Módulo de validación de esquemas y enumeraciones para los datos de entrada de la API.

Define las estructuras de datos tipadas mediante Pydantic y los catálogos
de opciones (Enumeraciones) basados estrictamente en el dataset Bank Marketing.
"""

from pydantic import BaseModel, Field
from enum import Enum


class JobEnum(str, Enum):
    """Enumeración con los tipos de ocupación laboral contemplados en el dataset."""
    admin = "admin."
    blue_collar = "blue-collar"
    entrepreneur = "entrepreneur"
    housemaid = "housemaid"
    management = "management"
    retired = "retired"
    self_employed = "self-employed"
    services = "services"
    student = "student"
    technician = "technician"
    unemployed = "unemployed"
    unknown = "unknown"


class MaritalEnum(str, Enum):
    """Enumeración con los estados civiles posibles."""
    divorced = "divorced"
    married = "married"
    single = "single"


class EducationEnum(str, Enum):
    """Enumeración con los niveles educativos alcanzados."""
    primary = "primary"
    secondary = "secondary"
    tertiary = "tertiary"
    unknown = "unknown"


class YesNoEnum(str, Enum):
    """Enumeración binaria estándar para afirmaciones."""
    yes = "yes"
    no = "no"


class ContactEnum(str, Enum):
    """Enumeración con los medios de comunicación de contacto."""
    cellular = "cellular"
    telephone = "telephone"
    unknown = "unknown"


class MonthEnum(str, Enum):
    """Enumeración con los meses del año."""
    jan = "jan"
    feb = "feb"
    mar = "mar"
    apr = "apr"
    may = "may"
    jun = "jun"
    jul = "jul"
    aug = "aug"
    sep = "sep"
    oct = "oct"
    nov = "nov"
    dec = "dec"


class POutcomeEnum(str, Enum):
    """Enumeración con los posibles resultados de campañas de marketing anteriores."""
    failure = "failure"
    other = "other"
    success = "success"
    unknown = "unknown"


class ClienteEntrada(BaseModel):
    """Esquema de validación estricta para los atributos de entrada del cliente.

    Excluye por completo la variable 'duration' para evitar la fuga de datos,
    reuniendo exactamente 40 características de entrada procesables.
    """
    age: int = Field(..., ge=18, le=100, description="Edad del cliente", example=41)
    job: JobEnum = Field(..., description="Tipo de trabajo del cliente")
    marital: MaritalEnum = Field(..., description="Estado civil")
    education: EducationEnum = Field(..., description="Nivel educativo")
    default: YesNoEnum = Field(..., description="¿Tiene crédito en morosidad?")
    balance: float = Field(..., description="Saldo promedio anual en euros", example=250.0)
    housing: YesNoEnum = Field(..., description="¿Tiene préstamo hipotecario?")
    loan: YesNoEnum = Field(..., description="¿Tiene préstamo personal?")
    contact: ContactEnum = Field(..., description="Tipo de comunicación de contacto")
    day: int = Field(..., ge=1, le=31, description="Último día de contacto del mes", example=15)
    month: MonthEnum = Field(..., description="Mes del último contacto")
    campaign: int = Field(..., ge=1, description="Número de contactos realizados en esta campaña", example=2)
    pdays: int = Field(..., description="Días transcurridos desde la campaña anterior (-1 si no hubo)", example=-1)
    previous: int = Field(..., description="Número de contactos previos", example=0)
    poutcome: POutcomeEnum = Field(..., description="Resultado de la campaña anterior")