"""Command-line adapter for the local-first experiment engine."""

from importlib import import_module

__all__ = [
    'build_parser',
    'main',
]


def __getattr__(name: str):
    if name in __all__:
        cli_main = import_module('ironflow_exp.engine.cli.main')
        return getattr(cli_main, name)

    raise AttributeError(name)
