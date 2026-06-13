"""
Server Entry Point — Run the FastAPI application.

Usage:
    python run_server.py
"""

import uvicorn


def main() -> None:
    """Start the uvicorn server."""
    print("=" * 60)
    print("  Customer Churn Prediction API Server")
    print("=" * 60)
    print()
    print("  📖 Swagger UI:  http://localhost:8000/docs")
    print("  📖 ReDoc:       http://localhost:8000/redoc")
    print("  🔗 API Root:    http://localhost:8000/")
    print()
    print("─" * 60)

    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
