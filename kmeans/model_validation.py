from pydantic import BaseModel, Field

class TrainingParameters(BaseModel):
    n_clusters: int = Field(description='Number of custers')
    init: str = Field(default='k-means++', description='Method of initialization: k-means++, random')
    n_init: int = Field(default=10, description='Number of time the k-means algorithm will be run with different centroid seed')
    max_iter: int = Field(default=300, description='Maximum number of iterations of the k-means algorithm for a single run')
    tol: float = Field(default=1e-4, description='Relative tolerance')
    random_state: int = Field(default=None, description='Use an int to make the randomness deterministic')
    algorithm: str = Field(default='auto', description='K-means algorithm to use: auto, full, elkan')
