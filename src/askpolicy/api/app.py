"""Security-only ASGI application.

With JWT_SECRET, JWT_ISSUER, and JWT_AUDIENCE set in the process environment,
run from the repository root:

    uv run --locked --no-sync uvicorn askpolicy.api.app:create_app --factory --app-dir src --no-access-log

Disable Uvicorn's access log to keep raw query strings out of logs.
"""

from fastapi import Depends, FastAPI, Response

from ..auth.context import AuthenticatedContext, make_auth_dependency
from ..auth.jwt import JWTConfig, JWTVerifier
from ..config import load_jwt_config


def create_app(*, jwt_config: JWTConfig | None = None) -> FastAPI:
    """Validate security configuration before constructing any routes.

    The zero-argument Uvicorn entry point uses the process environment.
    Trusted application code can inject a validated JWTConfig explicitly.
    """
    config = load_jwt_config() if jwt_config is None else jwt_config
    verifier = JWTVerifier(config)
    require_context = make_auth_dependency(verifier)
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, redirect_slashes=False)

    @app.get("/auth/verify")
    async def verify(
        response: Response, context: AuthenticatedContext = Depends(require_context),
    ) -> dict[str, bool | str]:
        response.headers["Cache-Control"] = "no-store"
        return {"authenticated": True, "active_role": context.active_role}

    return app
