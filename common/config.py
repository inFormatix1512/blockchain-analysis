"""
Configuration management for the Blockchain Analysis project.

Centralizes all configuration values with environment variable support
and sensible defaults for local development.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class DatabaseConfig:
    """PostgreSQL database configuration."""
    host: str = field(default_factory=lambda: os.environ.get('PGHOST', 'localhost'))
    port: int = field(default_factory=lambda: int(os.environ.get('PGPORT', '5432')))
    user: str = field(default_factory=lambda: os.environ.get('PGUSER', 'postgres'))
    password: str = field(default_factory=lambda: os.environ.get('PGPASSWORD', 'postgres'))
    database: str = field(default_factory=lambda: os.environ.get('PGDATABASE', 'blockchain'))
    
    @property
    def connection_string(self) -> str:
        """Returns a PostgreSQL connection string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    def to_dict(self) -> dict:
        """Returns connection parameters as dictionary for psycopg2."""
        return {
            'host': self.host,
            'port': self.port,
            'user': self.user,
            'password': self.password,
            'dbname': self.database
        }


@dataclass(frozen=True)
class RPCConfig:
    """Bitcoin Core RPC configuration."""
    user: str = field(default_factory=lambda: os.environ.get('RPC_USER', 'yourrpcuser'))
    password: str = field(default_factory=lambda: os.environ.get('RPC_PASSWORD', 'yourrpcpassword'))
    host: str = field(default_factory=lambda: os.environ.get('RPC_HOST', 'localhost'))
    port: int = field(default_factory=lambda: int(os.environ.get('RPC_PORT', '8332')))
    timeout: int = field(default_factory=lambda: int(os.environ.get('RPC_TIMEOUT') or '120'))
    max_retries: int = field(default_factory=lambda: int(os.environ.get('RPC_MAX_RETRIES', '3')))
    retry_delay: int = field(default_factory=lambda: int(os.environ.get('RPC_RETRY_DELAY', '5')))
    
    @property
    def url(self) -> str:
        """Returns the RPC endpoint URL."""
        return f"http://{self.host}:{self.port}"
    
    @property
    def auth(self) -> tuple:
        """Returns authentication tuple for requests."""
        return (self.user, self.password)


@dataclass(frozen=True)
class IngestConfig:
    """Block ingestion configuration."""
    max_blocks_per_run: int = field(
        default_factory=lambda: int(os.environ.get('MAX_BLOCKS_PER_RUN', '10'))
    )
    batch_size: int = field(
        default_factory=lambda: int(os.environ.get('BATCH_SIZE', '100'))
    )
    enable_mempool: bool = field(
        default_factory=lambda: os.environ.get('ENABLE_MEMPOOL', 'true').lower() == 'true'
    )


@dataclass(frozen=True)
class MLConfig:
    """Machine Learning configuration."""
    model_dir: str = field(
        default_factory=lambda: os.environ.get(
            'MODEL_DIR', 
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ml', 'models')
        )
    )
    random_state: int = 42
    test_size: float = 0.2
    contamination: float = 0.05
    n_clusters: int = 5


class Config:
    """
    Main configuration class providing access to all configuration sections.
    
    Usage:
        config = Config()
        db_params = config.database.to_dict()
        rpc_url = config.rpc.url
    """
    _instance: Optional['Config'] = None
    
    def __new__(cls) -> 'Config':
        """Singleton pattern for configuration."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self) -> None:
        """Initialize configuration sections."""
        self.database = DatabaseConfig()
        self.rpc = RPCConfig()
        self.ingest = IngestConfig()
        self.ml = MLConfig()
    
    @classmethod
    def reset(cls) -> None:
        """Reset singleton instance (useful for testing)."""
        cls._instance = None


# Convenience function for quick access
def get_config() -> Config:
    """Returns the singleton Config instance."""
    return Config()
