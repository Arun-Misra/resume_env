# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Resume Env Environment.

This module creates an HTTP server that exposes the ResumeEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    # Try package-relative imports first (when run via 'uv run' or as part of a package)
    from ..models import ResumeAction, ResumeObservation
    from .resume_env_environment import ResumeEnvironment
except (ImportError, ValueError):
    try:
        # Try direct imports (when run as 'python -m server.app' from root)
        from models import ResumeAction, ResumeObservation
        from server.resume_env_environment import ResumeEnvironment
    except ImportError:
        # Fallback for other contexts
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from models import ResumeAction, ResumeObservation
        from server.resume_env_environment import ResumeEnvironment


# Create the app with web interface and README integration
app = create_app(
    ResumeEnvironment,
    ResumeAction,
    ResumeObservation,
    env_name="resume_env",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)


def main():
    """Entry point for the environment server."""
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Resume Env Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
