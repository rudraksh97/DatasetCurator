"""Query handlers package.

This package provides query handlers following the Strategy pattern,
allowing new query types to be added without modifying existing code.
"""
from services.queries.registry import QueryHandlerRegistry, get_query_registry
from services.queries.base import BaseQueryHandler

__all__ = [
    "QueryHandlerRegistry",
    "get_query_registry",
    "BaseQueryHandler",
]
