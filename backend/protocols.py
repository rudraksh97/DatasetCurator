"""Abstract protocols (interfaces) for dependency inversion.

This module defines abstract interfaces that decouple components from their
concrete implementations, enabling:
- Easy testing with mock implementations
- Swapping implementations without changing clients
- Clear contracts between components

Note: We use typing.Protocol for structural subtyping (duck typing).
"""
from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Protocol, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


# Type aliases
Message = Dict[str, Any]
QueryResult = Dict[str, Any]


class LLMClient(Protocol):
    """Protocol for LLM API clients.
    
    Implementations should handle the specifics of communicating with
    different LLM providers (OpenAI, OpenRouter, Anthropic, etc.).
    """
    
    @abstractmethod
    async def complete(
        self,
        messages: Sequence[Message],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """Send a chat completion request.
        
        Args:
            messages: Conversation history.
            model: Model identifier (uses default if None).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
        
        Returns:
            The model's response text.
        """
        ...
    
    @abstractmethod
    async def complete_with_tools(
        self,
        messages: Sequence[Message],
        tools: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Any:
        """Send a chat completion request with function calling.
        
        Args:
            messages: Conversation history.
            tools: Tool definitions for function calling.
            model: Model identifier (uses default if None).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
        
        Returns:
            The API response object with potential tool calls.
        """
        ...


class IntentClassifier(Protocol):
    """Protocol for classifying user intent."""
    
    @abstractmethod
    async def classify(
        self,
        message: str,
        has_data: bool = False,
        columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Classify the intent of a user message.
        
        Args:
            message: User's input message.
            has_data: Whether user has loaded data.
            columns: Available column names if data is loaded.
        
        Returns:
            Dictionary with 'intent', 'params', and 'explanation' keys.
        """
        ...


class ExecutionPlanner(Protocol):
    """Protocol for planning data transformations."""
    
    @abstractmethod
    async def create_plan(
        self,
        user_message: str,
        columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Create an execution plan from user request.
        
        Args:
            user_message: User's transformation request.
            columns: Available column names.
        
        Returns:
            Dictionary with 'is_multi_step', 'steps', and optional 'error' keys.
        """
        ...


class QueryHandler(Protocol):
    """Protocol for data query handlers.
    
    Each implementation handles a specific type of query (count, statistics,
    search, etc.) following the Strategy pattern.
    """
    
    @property
    @abstractmethod
    def query_type(self) -> str:
        """The type of query this handler supports."""
        ...
    
    @abstractmethod
    def can_handle(self, function_name: str) -> bool:
        """Check if this handler can process the given function.
        
        Args:
            function_name: Name of the function to handle.
        
        Returns:
            True if this handler supports the function.
        """
        ...
    
    @abstractmethod
    def execute(
        self,
        df: "pd.DataFrame",
        arguments: Dict[str, Any],
    ) -> QueryResult:
        """Execute the query on the DataFrame.
        
        Args:
            df: DataFrame to query.
            arguments: Query parameters.
        
        Returns:
            Dictionary with 'success' key and results or 'error'.
        """
        ...


class DataOperationExecutor(Protocol):
    """Protocol for executing data operations."""
    
    @abstractmethod
    def execute(
        self,
        operation: str,
        params: Dict[str, Any],
    ) -> tuple[bool, str]:
        """Execute a data operation.
        
        Args:
            operation: Operation name.
            params: Operation parameters.
        
        Returns:
            Tuple of (success, message).
        """
        ...
    
    @abstractmethod
    def get_result(self) -> "pd.DataFrame":
        """Get the resulting DataFrame after operations."""
        ...


class DatasetRepository(Protocol):
    """Protocol for dataset persistence operations.
    
    Implements the Repository pattern for data access.
    """
    
    @abstractmethod
    async def get(self, dataset_id: str) -> Any:
        """Get a dataset state by ID.
        
        Args:
            dataset_id: Unique dataset identifier.
        
        Returns:
            The dataset state object.
        
        Raises:
            ValueError: If dataset not found.
        """
        ...
    
    @abstractmethod
    async def save(self, state: Any) -> Any:
        """Save or update a dataset state.
        
        Args:
            state: Dataset state to save.
        
        Returns:
            The saved state.
        """
        ...


class EmbeddingService(Protocol):
    """Protocol for embedding generation and search."""
    
    @abstractmethod
    async def generate(self, text: str) -> List[float]:
        """Generate embedding for a single text.
        
        Args:
            text: Text to embed.
        
        Returns:
            Embedding vector as list of floats.
        """
        ...
    
    @abstractmethod
    async def generate_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed.
        
        Returns:
            List of embedding vectors.
        """
        ...
    
    @abstractmethod
    async def search(
        self,
        dataset_id: str,
        query: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search for similar content.
        
        Args:
            dataset_id: Dataset to search in.
            query: Search query text.
            limit: Maximum results to return.
        
        Returns:
            List of matches with similarity scores.
        """
        ...


class DataLoader(Protocol):
    """Protocol for loading and processing data files."""
    
    @abstractmethod
    def load(
        self,
        file_path: Any,
        max_rows: Optional[int] = None,
        sample: bool = False,
    ) -> "pd.DataFrame":
        """Load a DataFrame from file.
        
        Args:
            file_path: Path to the data file.
            max_rows: Maximum rows to load.
            sample: Whether to use random sampling for large files.
        
        Returns:
            Loaded DataFrame.
        """
        ...
    
    @abstractmethod
    def is_large(self, file_path: Any) -> tuple[bool, Optional[int]]:
        """Check if a file is considered large.
        
        Args:
            file_path: Path to check.
        
        Returns:
            Tuple of (is_large, estimated_row_count).
        """
        ...
