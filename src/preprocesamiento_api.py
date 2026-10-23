"""Módulo de validación de esquemas y enumeraciones para los datos de entrada de la API.

Define las estructuras de datos tipadas mediante Pydantic y los catálogos
de opciones (Enumeraciones) basados estrictamente en el dataset Bank Marketing.
"""

from pydantic import BaseModel, ConfigDict, Field
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

    Recibe solo las 5 variables elegidas con el ranking de correlación. Cualquier
    otro campo (por ejemplo 'duration', que filtra el futuro) se rechaza.
    """
    model_config = ConfigDict(extra="forbid")

    poutcome: POutcomeEnum = Field(..., description="Resultado de la campaña anterior")
    month: MonthEnum = Field(..., description="Mes en que se realizará la llamada")
    contact: ContactEnum = Field(..., description="Tipo de comunicación de contacto")
    housing: YesNoEnum = Field(..., description="¿Tiene préstamo hipotecario?")
    job: JobEnum = Field(..., description="Tipo de trabajo del cliente")
