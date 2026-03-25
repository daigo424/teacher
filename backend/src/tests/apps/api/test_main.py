from fastapi.routing import APIRoute

from apps.api.main import app


def test_app_metadata_and_registered_routes():
    routes = {(route.path, next(iter(route.methods))) for route in app.routes if isinstance(route, APIRoute)}

    assert app.title == "Teacher API"
    assert ("/health", "GET") in routes
    assert ("/ask", "POST") in routes
