import os
import argparse
import uvicorn
import sys

def main():
    parser = argparse.ArgumentParser(description="Run FastAPI with MLflow model selection.")
    parser.add_argument("--experiment_id", required=False, help="MLflow experiment ID (optional, will use best model if not provided)")
    parser.add_argument("--run_id", required=False, help="MLflow run ID (optional, will use best model if not provided)")
    parser.add_argument("--host", default="0.0.0.0", help="Host for the API server")
    parser.add_argument("--port", type=int, default=8000, help="Port for the API server")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    # Ensure project root is in sys.path for module resolution
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Set environment variables only if provided
    if args.experiment_id:
        os.environ["EXPERIMENT_ID"] = args.experiment_id
    if args.run_id:
        os.environ["RUN_ID"] = args.run_id

    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )

if __name__ == "__main__":
    main()
