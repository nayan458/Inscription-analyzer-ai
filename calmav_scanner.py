"""
ClamAV Scanner Utility Module

This module provides reusable functions for scanning files with ClamAV.
Can be used across multiple services for virus/malware detection.

Requirements:
    - ClamAV daemon (clamd) running
    - pip install clamd
"""

import clamd
import asyncio
import tempfile
import os
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from enum import Enum
import logging

# Configure logging
logger = logging.getLogger(__name__)


class ScanStatus(Enum):
    """Enum for scan result statuses"""
    CLEAN = "clean"
    INFECTED = "infected"
    ERROR = "error"
    UNKNOWN = "unknown"


class ClamAVScanner:
    """
    ClamAV Scanner wrapper class
    
    Provides methods to connect to ClamAV daemon and scan files.
    Supports both Unix socket and TCP connections.
    """
    
    def __init__(self, connection_type: str = "unix", host: str = "localhost", port: int = 3310):
        """
        Initialize ClamAV Scanner
        
        Args:
            connection_type: "unix" for Unix socket or "tcp" for TCP connection
            host: Hostname for TCP connection (default: localhost)
            port: Port for TCP connection (default: 3310)
        """
        self.connection_type = connection_type
        self.host = host
        self.port = port
        self._client = None
    
    def get_client(self) -> Optional[clamd.ClamdUnixSocket]:
        """
        Get or create ClamAV client connection
        
        Returns:
            ClamAV client instance or None if connection fails
        """
        try:
            if self.connection_type == "unix":
                client = clamd.ClamdUnixSocket()
            else:
                client = clamd.ClamdNetworkSocket(self.host, self.port)
            
            # Test connection
            client.ping()
            self._client = client
            return client
        except Exception as e:
            logger.error(f"ClamAV connection failed: {e}")
            return None
    
    def is_available(self) -> bool:
        """
        Check if ClamAV service is available
        
        Returns:
            True if ClamAV is available, False otherwise
        """
        client = self.get_client()
        return client is not None
    
    def get_version(self) -> Optional[str]:
        """
        Get ClamAV version information
        
        Returns:
            Version string or None if unavailable
        """
        try:
            client = self.get_client()
            if client:
                return client.version()
            return None
        except Exception as e:
            logger.error(f"Failed to get ClamAV version: {e}")
            return None
    
    def scan_file(self, file_path: str) -> Dict:
        """
        Scan a file for malware
        
        Args:
            file_path: Path to the file to scan
            
        Returns:
            Dictionary containing scan results
        """
        try:
            client = self.get_client()
            if not client:
                return {"error": "ClamAV service unavailable"}
            
            if not os.path.exists(file_path):
                return {"error": f"File not found: {file_path}"}
            
            scan_result = client.scan(file_path)
            return scan_result
        except Exception as e:
            logger.error(f"Scan failed for {file_path}: {e}")
            return {"error": f"Scan failed: {str(e)}"}
    
    def scan_stream(self, file_stream: bytes) -> Dict:
        """
        Scan a file stream for malware
        
        Args:
            file_stream: Bytes content of the file
            
        Returns:
            Dictionary containing scan results
        """
        try:
            client = self.get_client()
            if not client:
                return {"error": "ClamAV service unavailable"}
            
            scan_result = client.instream(file_stream)
            return scan_result
        except Exception as e:
            logger.error(f"Stream scan failed: {e}")
            return {"error": f"Stream scan failed: {str(e)}"}


class ScanResult:
    """
    Formatted scan result class
    """
    
    def __init__(self, filename: str, status: ScanStatus, message: str, 
                 threat_name: Optional[str] = None, file_size: Optional[int] = None):
        self.filename = filename
        self.status = status
        self.message = message
        self.threat_name = threat_name
        self.file_size = file_size
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format"""
        result = {
            "filename": self.filename,
            "status": self.status.value,
            "message": self.message
        }
        if self.threat_name:
            result["threat_name"] = self.threat_name
        if self.file_size is not None:
            result["file_size"] = self.file_size
        return result
    
    def is_safe(self) -> bool:
        """Check if file is safe (clean)"""
        return self.status == ScanStatus.CLEAN


# ============================================================================
# Utility Functions
# ============================================================================

def format_scan_result(clamav_result: Dict, filename: str, 
                       file_size: Optional[int] = None) -> ScanResult:
    """
    Format raw ClamAV result into a user-friendly ScanResult object
    
    Args:
        clamav_result: Raw result from ClamAV scan
        filename: Name of the scanned file
        file_size: Size of the file in bytes (optional)
        
    Returns:
        ScanResult object
    """
    if "error" in clamav_result:
        return ScanResult(
            filename=filename,
            status=ScanStatus.ERROR,
            message=clamav_result["error"],
            file_size=file_size
        )
    
    # Parse ClamAV result
    for file_path, result in clamav_result.items():
        status_code, details = result
        
        if status_code == "OK":
            return ScanResult(
                filename=filename,
                status=ScanStatus.CLEAN,
                message="No threats detected",
                file_size=file_size
            )
        elif status_code == "FOUND":
            return ScanResult(
                filename=filename,
                status=ScanStatus.INFECTED,
                message=f"Threat detected: {details}",
                threat_name=details,
                file_size=file_size
            )
        else:
            return ScanResult(
                filename=filename,
                status=ScanStatus.UNKNOWN,
                message=f"Unexpected result: {result}",
                file_size=file_size
            )
    
    # Fallback if no results
    return ScanResult(
        filename=filename,
        status=ScanStatus.UNKNOWN,
        message="No scan results returned",
        file_size=file_size
    )


async def save_upload_file_temp(file_content: bytes, filename: str) -> str:
    """
    Save uploaded file content to a temporary file
    
    Args:
        file_content: Bytes content of the file
        filename: Original filename (used to preserve extension)
        
    Returns:
        Path to temporary file
        
    Raises:
        Exception if file save fails
    """
    try:
        suffix = Path(filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_content)
            return tmp.name
    except Exception as e:
        logger.error(f"Failed to save temporary file: {e}")
        raise Exception(f"File save failed: {str(e)}")


def cleanup_temp_file(file_path: str) -> None:
    """
    Clean up temporary file
    
    Args:
        file_path: Path to the temporary file to delete
    """
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
            logger.debug(f"Cleaned up temporary file: {file_path}")
    except Exception as e:
        logger.warning(f"Cleanup failed for {file_path}: {e}")


async def scan_file_async(scanner: ClamAVScanner, file_path: str) -> Dict:
    """
    Asynchronously scan a file
    
    Args:
        scanner: ClamAVScanner instance
        file_path: Path to file to scan
        
    Returns:
        Dictionary containing scan results
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, scanner.scan_file, file_path)


async def scan_uploaded_file(
    scanner: ClamAVScanner,
    file_content: bytes,
    filename: str,
    cleanup: bool = True
) -> ScanResult:
    """
    Scan an uploaded file (high-level convenience function)
    
    Args:
        scanner: ClamAVScanner instance
        file_content: File content as bytes
        filename: Original filename
        cleanup: Whether to clean up temporary file after scanning
        
    Returns:
        ScanResult object with scan details
    """
    temp_file_path = None
    try:
        # Save to temporary file
        temp_file_path = await save_upload_file_temp(file_content, filename)
        
        # Scan the file
        raw_result = await scan_file_async(scanner, temp_file_path)
        
        # Format result
        result = format_scan_result(
            raw_result, 
            filename, 
            file_size=len(file_content)
        )
        
        return result
        
    finally:
        # Cleanup temporary file
        if cleanup and temp_file_path:
            cleanup_temp_file(temp_file_path)


async def scan_multiple_files(
    scanner: ClamAVScanner,
    files: List[Tuple[bytes, str]],
    cleanup: bool = True
) -> Tuple[List[ScanResult], Dict]:
    """
    Scan multiple files and return results with summary
    
    Args:
        scanner: ClamAVScanner instance
        files: List of tuples (file_content, filename)
        cleanup: Whether to clean up temporary files
        
    Returns:
        Tuple of (list of ScanResults, summary dict)
    """
    results = []
    
    for file_content, filename in files:
        result = await scan_uploaded_file(scanner, file_content, filename, cleanup)
        results.append(result)
    
    # Generate summary
    summary = {
        "total_files": len(results),
        "clean": sum(1 for r in results if r.status == ScanStatus.CLEAN),
        "infected": sum(1 for r in results if r.status == ScanStatus.INFECTED),
        "errors": sum(1 for r in results if r.status == ScanStatus.ERROR),
        "unknown": sum(1 for r in results if r.status == ScanStatus.UNKNOWN)
    }
    
    return results, summary


# ============================================================================
# Quick-start helper function
# ============================================================================

def create_scanner(connection_type: str = "unix", 
                   host: str = "localhost", 
                   port: int = 3310) -> ClamAVScanner:
    """
    Factory function to create a ClamAVScanner instance
    
    Args:
        connection_type: "unix" or "tcp"
        host: Hostname for TCP connection
        port: Port for TCP connection
        
    Returns:
        ClamAVScanner instance
    """
    return ClamAVScanner(connection_type, host, port)