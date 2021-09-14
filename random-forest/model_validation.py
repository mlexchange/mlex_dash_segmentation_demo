from enum import Enum
from pydantic import BaseModel, Field, validator
from typing import Optional


class TrainingParameters(BaseModel):
    n_estimators: int = Field(description='number of trees')
    oob_score: Optional[bool] = Field(default=False, description="output out-of-bag score") 
    max_depth: int = Field(description="maximum depth for growing tree")


class Metadata(BaseModel):
    oob: float = Field(description='out-of-bag error')


class TestingParameters(BaseModel):
    show_progress: Optional[int] = Field(default=1, description="number of iterations to progress report")
