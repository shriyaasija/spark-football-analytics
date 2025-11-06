from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import ClassVar

class Settings(BaseSettings):
    APP_NAME: ClassVar[str] = "SPARK API"
    APP_VERSION: ClassVar[str] = "1.0.0"
    APP_DESCRIPTION: ClassVar[str] = "Sports Performance Analysis and Ranking Kit"
    DEBUG: bool = True

    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "spark_db"
    POSTGRES_USER: str = "spark_user"  # add this (you used it in DATABASE_URL)
    POSTGRES_PASSWORD: str = "spark_password_2024"

    API_V1_PREFIX: str = "/api/v1"

    APP_ENV: str = "development"
    SECRET_KEY: str = "your-secret-key-change-in-production"

    # API config
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # JWT Authentication
    SECRET_KEY: str = "your-secret-key-change-in-production-use-openssl-rand-hex-32"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours

    @property
    def DATABASE_URL(self):
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )
    
    class Config:
        env_file = "../.env"
        case_sensitive = True

@lru_cache
def get_settings():
    return Settings()