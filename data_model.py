from pydantic import BaseModel

class Water(BaseModel):
    Hardness: float
    Solids: float
    Chloramines: float
    Conductivity: float
    Organic_carbon: float
    Turbidity: float
    ph: float
    Sulfate: float
    Trihalomethanes: float

