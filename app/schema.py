# app/schemas.py

from pydantic import BaseModel, Field
from typing import Literal, Optional

class HouseFeatures(BaseModel):
    """
    Schema for the input features of the house price prediction model.
    This reflects the raw input format expected by the API.
    """
    area: float = Field(..., gt=0, description="Area of the house in sqft (must be positive)")
    bedrooms: Optional[float] = Field(None, ge=1, description="Number of bedrooms (minimum 1, can be null)")
    bathrooms: float = Field(..., ge=1, description="Number of bathrooms (minimum 1)")
    stories: float = Field(..., ge=1, description="Number of stories (minimum 1)")
    mainroad: Literal['yes', 'no'] = Field(..., description="Whether the house is on a main road")
    guestroom: Literal['yes', 'no'] = Field(..., description="Whether the house has a guest room")
    basement: Literal['yes', 'no'] = Field(..., description="Whether the house has a basement")
    hotwaterheating: Literal['yes', 'no'] = Field(..., description="Whether the house has hot water heating")
    airconditioning: Literal['yes', 'no'] = Field(..., description="Whether the house has air conditioning")
    parking: int = Field(..., ge=0, le=3, description="Number of parking spots (0 to 3)")
    prefarea: Literal['yes', 'no'] = Field(..., description="Whether the house is in a preferred area")
    furnishingstatus: Literal['furnished', 'semi-furnished', 'unfurnished'] = Field(..., description="Furnishing status of the house")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "area": 7420.0,
                    "bedrooms": 4.0,
                    "bathrooms": 2.0,
                    "stories": 3.0,
                    "mainroad": "yes",
                    "guestroom": "no",
                    "basement": "no",
                    "hotwaterheating": "no",
                    "airconditioning": "yes",
                    "parking": 2,
                    "prefarea": "yes",
                    "furnishingstatus": "furnished"
                },
                {
                    "area": 3000.0,
                    "bedrooms": 2.0,
                    "bathrooms": 1.0,
                    "stories": 1.0,
                    "mainroad": "no",
                    "guestroom": "no",
                    "basement": "no",
                    "hotwaterheating": "no",
                    "airconditioning": "no",
                    "parking": 0,
                    "prefarea": "no",
                    "furnishingstatus": "unfurnished"
                }
            ]
        }
    }

class PredictionResponse(BaseModel):
    """Schema for the prediction output."""
    predicted_price: float = Field(..., description="The predicted house price")
    model_version: str = Field(..., description="Version of the model used for prediction")