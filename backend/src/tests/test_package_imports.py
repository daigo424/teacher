import importlib


def test_package_modules_import():
    module_names = [
        "apps.api",
        "apps.api.routers",
        "apps.api.schemas",
        "apps.ingest",
        "apps.wikipedia_to_markdown",
        "packages.core.config",
        "packages.core.db",
        "packages.core.db.models",
        "packages.core.schemas",
        "packages.core.services",
    ]

    for module_name in module_names:
        assert importlib.import_module(module_name).__name__ == module_name
