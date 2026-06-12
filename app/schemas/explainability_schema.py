from pydantic import BaseModel, Field


class AutoSklearnTrainParams(BaseModel):
    time_left_for_this_task: int = Field(default=120, ge=30, le=1800)
    per_run_time_limit: int = Field(default=30, ge=10, le=300)


class AutoSklearnExplainParams(BaseModel):
    top_n: int = Field(default=5, ge=1, le=10)