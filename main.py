from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn

# Import routers
from routers import auth, fields, vi_analysis, utils, images, tunnel

# Import database
from database import Base, engine

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    print("Starting Grovi API...")
    
    # Create database tables
    try:
        Base.metadata.create_all(bind=engine)
        print("Database tables created successfully")
    except Exception as e:
        print(f"Error creating database tables: {e}")
    
    # Initialize Google Earth Engine (optional)
    try:
        from gee_service import gee_service
        print("Google Earth Engine service initialized")
    except Exception as e:
        print(f"Info: Google Earth Engine not available - real satellite data will not be accessible: {e}")
    
    yield
    
    # Shutdown
    print("Shutting down Grovi API...")

# Create FastAPI application
app = FastAPI(
    title="Grovi - Crop Monitoring API",
    description="API for crop monitoring using satellite data and vegetation indices",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://localhost:5173", 
        "http://127.0.0.1:3000", 
        "http://127.0.0.1:5173",
        # Current tunnel URLs
        "https://flour-memo-regards-legend.trycloudflare.com",
        "https://sf-polyphonic-scanners-maintenance.trycloudflare.com",
        # Previous tunnels (kept for reference)
        "https://postage-msie-skilled-insulin.trycloudflare.com",
        "https://visitors-hc-assign-glasses.trycloudflare.com",
    ],
    # Accept any trycloudflare.com subdomain during development
    allow_origin_regex=r"^https://[a-zA-Z0-9-]+\.trycloudflare\.com$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router)
app.include_router(fields.router)
app.include_router(vi_analysis.router)
app.include_router(vi_analysis.vi_router)  # Add compatibility router
app.include_router(images.router)
app.include_router(utils.router)
app.include_router(tunnel.router)

@app.get("/")
def read_root():
    """Root endpoint"""
    return {
        "message": "Welcome to Grovi - Crop Monitoring API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc"
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Grovi API is running successfully"
    }

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    print(f"Global exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "message": "เกิดข้อผิดพลาดภายในระบบ"
        }
    )

if __name__ == "__main__":
    print("🚀 Starting Grovi API server...")
    try:
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        import traceback
        traceback.print_exc()
