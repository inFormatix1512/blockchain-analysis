"""
Bitcoin Core RPC client for the Blockchain Analysis project.

Provides a robust RPC client with retry logic, error handling,
and common Bitcoin operations.
"""

import logging
import time
from typing import Any, Dict, List, Optional

import requests
from requests.exceptions import RequestException

from .config import Config

logger = logging.getLogger(__name__)


class RPCError(Exception):
    """Custom exception for RPC operations."""
    
    def __init__(self, message: str, code: Optional[int] = None):
        super().__init__(message)
        self.code = code


class BitcoinRPC:
    """
    Bitcoin Core JSON-RPC client with retry logic.
    
    Usage:
        rpc = BitcoinRPC()
        
        # Get blockchain info
        info = rpc.call('getblockchaininfo')
        
        # Get block by height
        block_hash = rpc.call('getblockhash', [height])
        block = rpc.call('getblock', [block_hash, 2])
        
        # With context manager for better resource management
        with BitcoinRPC() as rpc:
            info = rpc.get_blockchain_info()
    """
    
    def __init__(self):
        """Initialize RPC client with configuration."""
        self._config = Config().rpc
        self._session: Optional[requests.Session] = None
        self._request_id = 0
    
    @property
    def session(self) -> requests.Session:
        """Lazy initialization of requests session."""
        if self._session is None:
            self._session = requests.Session()
            self._session.auth = self._config.auth
            self._session.headers.update({'Content-Type': 'application/json'})
        return self._session
    
    def _get_request_id(self) -> str:
        """Generate unique request ID."""
        self._request_id += 1
        return f"blockchain-analysis-{self._request_id}"
    
    def call(
        self, 
        method: str, 
        params: Optional[List] = None,
        max_retries: Optional[int] = None,
        retry_delay: Optional[int] = None
    ) -> Any:
        """
        Execute an RPC call with retry logic.
        
        Args:
            method: RPC method name (e.g., 'getblockchaininfo').
            params: List of parameters for the method.
            max_retries: Override default max retries.
            retry_delay: Override default retry delay.
            
        Returns:
            The result from the RPC call.
            
        Raises:
            RPCError: If the RPC call fails after all retries.
        """
        max_retries = max_retries or self._config.max_retries
        retry_delay = retry_delay or self._config.retry_delay
        
        payload = {
            'jsonrpc': '2.0',
            'id': self._get_request_id(),
            'method': method,
            'params': params or [],
        }
        
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                response = self.session.post(
                    self._config.url,
                    json=payload,
                    timeout=self._config.timeout,
                )
                response.raise_for_status()
                
                result = response.json()
                
                if result.get('error') is not None:
                    error = result['error']
                    raise RPCError(
                        f"RPC error: {error.get('message', str(error))}",
                        code=error.get('code')
                    )
                
                return result['result']
                
            except RequestException as exc:
                last_exception = exc
                logger.warning(
                    "RPC call '%s' failed (attempt %d/%d): %s",
                    method, attempt + 1, max_retries, exc
                )
            except RPCError:
                raise
            
            if attempt < max_retries - 1:
                sleep_time = retry_delay * (2 ** attempt)
                logger.debug("Retrying in %d seconds...", sleep_time)
                time.sleep(sleep_time)
        
        raise RPCError(
            f"RPC call '{method}' failed after {max_retries} attempts: {last_exception}"
        )
    
    def close(self) -> None:
        """Close the requests session."""
        if self._session is not None:
            self._session.close()
            self._session = None
    
    def __enter__(self) -> 'BitcoinRPC':
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
    
    # Convenience methods for common operations
    
    def get_blockchain_info(self) -> Dict:
        """Get blockchain information."""
        return self.call('getblockchaininfo')
    
    def get_block_count(self) -> int:
        """Get current block height."""
        return self.call('getblockcount')
    
    def get_block_hash(self, height: int) -> str:
        """Get block hash by height."""
        return self.call('getblockhash', [height])
    
    def get_block(self, block_hash: str, verbosity: int = 2) -> Dict:
        """
        Get block data by hash.
        
        Args:
            block_hash: The block hash.
            verbosity: 0=hex, 1=json, 2=json with tx details.
        """
        return self.call('getblock', [block_hash, verbosity])
    
    def get_mempool_info(self) -> Dict:
        """Get mempool information."""
        return self.call('getmempoolinfo')
    
    def get_raw_mempool(self, verbose: bool = True) -> Dict:
        """Get raw mempool contents."""
        return self.call('getrawmempool', [verbose])
    
    def is_syncing(self) -> bool:
        """Check if node is in initial block download."""
        try:
            info = self.get_blockchain_info()
            return info.get('initialblockdownload', True)
        except RPCError as exc:
            logger.warning("Failed to check IBD status: %s", exc)
            return True


# Convenience function for quick RPC calls
def rpc_call(method: str, params: Optional[List] = None) -> Any:
    """
    Execute a single RPC call.
    
    This is a convenience function that creates a BitcoinRPC instance,
    executes the call, and returns the result.
    
    For multiple calls, it's more efficient to use BitcoinRPC directly.
    
    Args:
        method: RPC method name.
        params: Method parameters.
        
    Returns:
        The result from the RPC call.
    """
    with BitcoinRPC() as rpc:
        return rpc.call(method, params)
