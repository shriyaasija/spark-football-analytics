from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from src.core.config import get_settings
from src.core.database import get_db, engine, Base
from src.api.routes import teams
from src.schemas.schemas import HealthCheck

settings = get_settings()

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware (allows frontend to call API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(teams.router, prefix=settings.API_V1_PREFIX)


@app.get("/", response_model=HealthCheck)
def root():
    """Root endpoint - API health check"""
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "database": "connected"
    }


@app.get("/health", response_model=HealthCheck)
def health_check(db: Session = Depends(get_db)):
    """Detailed health check with database connection test"""
    try:
        # Test database connection
        db.execute("SELECT 1")
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    return {
        "status": "healthy" if db_status == "connected" else "unhealthy",
        "version": settings.APP_VERSION,
        "database": db_status
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # Auto-reload on code changes
    )