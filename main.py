"""FastAPI application entry point for the AI Research Analyzer.

Configures CORS middleware, registers API routes, and provides a
``uvicorn.run`` invocation for local development.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import Settings

settings = Settings()

app = FastAPI(
    title="AI Research Analyzer",
    description="RAG Chatbot with AI Evaluation Engine for Research Papers",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
