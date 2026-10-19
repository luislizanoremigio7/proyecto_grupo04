from pydantic import BaseModel, Field
from enum import Enum

# Definición de enumeraciones con las opciones reales del dataset Bank Marketing
class JobEnum(str, Enum):
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
    divorced = "divorced"
    married = "married"
    single = "single"

class EducationEnum(str, Enum):
    primary = "primary"
    secondary = "secondary"
    tertiary = "tertiary"
    unknown = "unknown"

class YesNoEnum(str, Enum):
    yes = "yes"
    no = "no"

class ContactEnum(str, Enum):
    cellular = "cellular"
    telephone = "telephone"
    unknown = "unknown"

class MonthEnum(str, Enum):
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
    failure = "failure"
    other = "other"
    success = "success"
    unknown = "unknown"


class ClienteEntrada(BaseModel):
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