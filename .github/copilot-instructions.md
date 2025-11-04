# GitHub Copilot Instructions for Python Projects

## Agent Instructions

- Follow the project structure and organization as outlined in the Project Overview.
- Adhere to the coding standards and best practices specified in the Code Style & Formatting section.
- Implement error handling and logging as described in the Error Handling and Logging Standards sections.
- Provide education during the implementation of this project by adding comments and docstrings to explain complex logic particularly around any mathematical concepts and algorithms to educate the user.

## Project Overview

This is a modern Python application following industry best practices for maintainable, scalable, and robust software development.

## Code Style & Formatting

### Python Standards

- Follow PEP 8 style guidelines strictly
- Use `black` for code formatting with line length of 88 characters
- Use `isort` for import sorting
- Use `flake8` or `ruff` for linting
- Type hints are required for all function signatures and class attributes
- Use `mypy` for static type checking

### Code Organization

```python
# Import order (using isort):
# 1. Standard library imports
# 2. Third-party imports
# 3. Local application imports

import os
from typing import Optional, Dict, List

import requests
import pandas as pd

from .models import User
from .utils import helper_function
```

## Architecture Patterns

### Project Structure

```
project/
├── src/
│   └── project_name/
│       ├── __init__.py
│       ├── main.py
│       ├── models/
│       ├── services/
│       ├── utils/
│       └── config/
├── tests/
├── docs/
├── requirements/
│   ├── base.txt
│   ├── dev.txt
│   └── prod.txt
├── pyproject.toml
├── README.md
└── .env.example
```

### Design Principles

- **Single Responsibility Principle**: Each class/function should have one reason to change
- **Dependency Injection**: Use dependency injection for testability
- **Configuration Management**: Use environment variables and configuration files
- **Error Handling**: Implement comprehensive error handling with custom exceptions
- **Logging**: Use structured logging with appropriate log levels

## Code Quality Standards

### Function Design

```python
from typing import Optional, List, Dict
import logging

logger = logging.getLogger(__name__)

def process_data(
    data: List[Dict[str, Any]],
    filter_criteria: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Process and filter data according to specified criteria.

    Args:
        data: List of data dictionaries to process
        filter_criteria: Optional filtering parameters

    Returns:
        Filtered and processed data

    Raises:
        ValidationError: If data format is invalid
        ProcessingError: If processing fails
    """
    if not data:
        logger.warning("Empty data provided for processing")
        return []

    try:
        # Implementation here
        result = []
        for item in data:
            if _meets_criteria(item, filter_criteria):
                result.append(_transform_item(item))
        return result
    except Exception as e:
        logger.error(f"Data processing failed: {e}")
        raise ProcessingError(f"Failed to process data: {e}") from e
```

### Class Design

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

@dataclass
class UserModel:
    """User data model with validation."""
    id: int
    name: str
    email: str

    def __post_init__(self) -> None:
        if not self.email or "@" not in self.email:
            raise ValueError("Invalid email format")

class DataProcessor(Protocol):
    """Protocol for data processing implementations."""

    def process(self, data: Any) -> Any:
        """Process the given data."""
        ...

class BaseService(ABC):
    """Abstract base class for services."""

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    @abstractmethod
    def execute(self) -> Any:
        """Execute the service logic."""
        pass
```

## Testing Standards

### Test Structure

- Use `pytest` as the testing framework
- Maintain >90% code coverage
- Write unit tests, integration tests, and end-to-end tests
- Use `pytest-mock` for mocking
- Use `pytest-cov` for coverage reporting

### Test Examples

```python
import pytest
from unittest.mock import Mock, patch
from src.services.user_service import UserService
from src.models.user import User

class TestUserService:
    """Test suite for UserService."""

    @pytest.fixture
    def mock_repository(self):
        return Mock()

    @pytest.fixture
    def user_service(self, mock_repository):
        return UserService(repository=mock_repository)

    def test_create_user_success(self, user_service, mock_repository):
        # Arrange
        user_data = {"name": "John Doe", "email": "john@example.com"}
        expected_user = User(id=1, **user_data)
        mock_repository.save.return_value = expected_user

        # Act
        result = user_service.create_user(user_data)

        # Assert
        assert result == expected_user
        mock_repository.save.assert_called_once()

    def test_create_user_invalid_email_raises_error(self, user_service):
        # Arrange
        user_data = {"name": "John Doe", "email": "invalid-email"}

        # Act & Assert
        with pytest.raises(ValueError, match="Invalid email format"):
            user_service.create_user(user_data)
```

## Error Handling

### Custom Exceptions

```python
class ProjectBaseException(Exception):
    """Base exception for project-specific errors."""
    pass

class ValidationError(ProjectBaseException):
    """Raised when data validation fails."""
    pass

class ProcessingError(ProjectBaseException):
    """Raised when data processing fails."""
    pass

class ConfigurationError(ProjectBaseException):
    """Raised when configuration is invalid."""
    pass
```

### Error Handling Patterns

```python
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def safe_operation(data: Any) -> Optional[Any]:
    """Perform operation with comprehensive error handling."""
    try:
        # Validate input
        if not _validate_data(data):
            raise ValidationError("Invalid input data")

        # Process data
        result = _process_data(data)
        logger.info("Operation completed successfully")
        return result

    except ValidationError as e:
        logger.error(f"Validation failed: {e}")
        raise  # Re-raise validation errors
    except ProcessingError as e:
        logger.error(f"Processing failed: {e}")
        return None  # Return None for processing errors
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        raise ProcessingError(f"Operation failed: {e}") from e
```

## Configuration Management

### Environment-based Configuration

```python
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class Settings:
    """Application settings loaded from environment variables."""

    # Database
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///app.db")
    database_pool_size: int = int(os.getenv("DATABASE_POOL_SIZE", "10"))

    # API
    api_host: str = os.getenv("API_HOST", "localhost")
    api_port: int = int(os.getenv("API_PORT", "8000"))

    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    # External Services
    external_api_key: Optional[str] = os.getenv("EXTERNAL_API_KEY")

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not self.external_api_key:
            raise ConfigurationError("EXTERNAL_API_KEY is required")

# Global settings instance
settings = Settings()
```

## Logging Standards

### Structured Logging Setup

```python
import logging
import json
from typing import Any, Dict

class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)

def setup_logging(log_level: str = "INFO") -> None:
    """Configure application logging."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("app.log"),
        ]
    )

    # Apply JSON formatter to all handlers
    formatter = JSONFormatter()
    for handler in logging.root.handlers:
        handler.setFormatter(formatter)
```

## Performance & Security

### Performance Guidelines

- Use `asyncio` for I/O-bound operations
- Implement caching strategies (Redis, in-memory)
- Use database connection pooling
- Profile code with `cProfile` for performance bottlenecks
- Use lazy loading and pagination for large datasets

### Security Best Practices

```python
import secrets
import hashlib
from cryptography.fernet import Fernet

class SecurityUtils:
    """Security utility functions."""

    @staticmethod
    def generate_secure_token(length: int = 32) -> str:
        """Generate a cryptographically secure random token."""
        return secrets.token_urlsafe(length)

    @staticmethod
    def hash_password(password: str, salt: bytes) -> str:
        """Hash password with salt using SHA-256."""
        return hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000).hex()

    @staticmethod
    def encrypt_sensitive_data(data: str, key: bytes) -> str:
        """Encrypt sensitive data using Fernet encryption."""
        f = Fernet(key)
        return f.encrypt(data.encode()).decode()
```

## Development Workflow

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3.11

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.3.0
    hooks:
      - id: mypy
```

### Documentation Standards

- Use docstrings for all public functions, classes, and modules
- Follow Google or NumPy docstring conventions
- Include type hints in docstrings
- Maintain up-to-date README.md with setup and usage instructions
- Use Sphinx or MkDocs for API documentation

## Dependencies Management

### Requirements Structure

```
# requirements/base.txt
requests>=2.31.0
pydantic>=2.0.0
sqlalchemy>=2.0.0

# requirements/dev.txt
-r base.txt
pytest>=7.4.0
pytest-cov>=4.1.0
black>=23.3.0
isort>=5.12.0
mypy>=1.3.0
pre-commit>=3.3.0

# requirements/prod.txt
-r base.txt
gunicorn>=21.2.0
```

## Copilot-Specific Instructions

When generating code:

1. Always include proper type hints
2. Add comprehensive docstrings
3. Implement error handling with custom exceptions
4. Include logging statements for important operations
5. Write corresponding unit tests
6. Follow the established project structure
7. Use dependency injection patterns
8. Validate inputs and handle edge cases
9. Optimize for readability and maintainability
10. Consider security implications

When refactoring existing code:

1. Maintain backward compatibility when possible
2. Update tests to reflect changes
3. Update documentation
4. Consider performance implications
5. Ensure type safety with mypy

## Additional Tools & Libraries

### Recommended Libraries

- **Web Frameworks**: FastAPI, Flask
- **Database**: SQLAlchemy, Alembic
- **Validation**: Pydantic
- **Testing**: pytest, pytest-mock, pytest-cov
- **Async**: asyncio, aiohttp
- **CLI**: Click, Typer
- **Environment**: python-dotenv
- **Monitoring**: structlog, sentry-sdk

### Development Tools

- **Code Quality**: ruff, black, isort, mypy
- **Documentation**: Sphinx, mkdocs
- **Containerization**: Docker, docker-compose
- **CI/CD**: GitHub Actions
- **Package Management**: pip-tools, poetry
