"""Query handler registry.

This module provides a registry for query handlers, implementing the
Strategy pattern with centralized handler lookup.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Optional

import pandas as pd

from protocols import QueryResult
from services.queries.base import BaseQueryHandler
from services.queries.handlers import (
    CalculateRatioHandler,
    FindColumnsHandler,
    GetRandomValueHandler,
    GetRowCountHandler,
    GetRowHandler,
    GetStatisticsHandler,
    GetValueHandler,
    GroupByHandler,
    ListColumnsHandler,
    SearchRowsHandler,
)


class QueryHandlerRegistry:
    """Registry for query handlers.
    
    Maintains a collection of query handlers and routes queries
    to the appropriate handler based on function name.
    
    This implements the Strategy pattern - handlers are strategies
    and the registry is the context that selects the right strategy.
    
    Attributes:
        _handlers: List of registered handlers.
    """
    
    def __init__(self):
        """Initialize the registry with default handlers."""
        self._handlers: List[BaseQueryHandler] = []
        self._register_default_handlers()
    
    def _register_default_handlers(self) -> None:
        """Register the default set of query handlers."""
        handlers = [
            FindColumnsHandler(),
            SearchRowsHandler(),
            GetRowHandler(),
            GetValueHandler(),
            CalculateRatioHandler(),
            GetStatisticsHandler(),
            GroupByHandler(),
            GetRandomValueHandler(),
            ListColumnsHandler(),
            GetRowCountHandler(),
        ]
        for handler in handlers:
            self.register(handler)
    
    def register(self, handler: BaseQueryHandler) -> None:
        """Register a query handler.
        
        Args:
            handler: Handler instance to register.
        """
        self._handlers.append(handler)
    
    def get_handler(self, function_name: str) -> Optional[BaseQueryHandler]:
        """Get a handler for the given function name.
        
        Args:
            function_name: Name of the function to handle.
        
        Returns:
            Handler instance or None if not found.
        """
        for handler in self._handlers:
            if handler.can_handle(function_name):
                return handler
        return None
    
    def execute(
        self,
        df: pd.DataFrame,
        function_name: str,
        arguments: Dict[str, Any],
    ) -> QueryResult:
        """Execute a query using the appropriate handler.
        
        Args:
            df: DataFrame to query.
            function_name: Name of the function to execute.
            arguments: Query parameters.
        
        Returns:
            Query result dictionary.
        """
        handler = self.get_handler(function_name)
        
        if handler:
            try:
                return handler.execute(df, arguments)
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        return {"success": False, "error": f"Unknown function: {function_name}"}
    
    @property
    def supported_functions(self) -> List[str]:
        """Get list of all supported function names.
        
        Returns:
            List of function name strings.
        """
        functions = []
        for handler in self._handlers:
            functions.extend(handler.supported_functions)
        return functions


@lru_cache(maxsize=1)
def get_query_registry() -> QueryHandlerRegistry:
    """Get the singleton query handler registry.
    
    Returns:
        Configured QueryHandlerRegistry instance.
    """
    return QueryHandlerRegistry()
