"""Direct Sarvam Vision API client (REST) - bypasses MCP for reliability."""
import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Optional

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

SARVAM_BASE_URL = "https://api.sarvam.ai"
DOC_JOB_BASE = "/doc-digitization/job/v1"
DOC_JOB_UPLOAD = f"{DOC_JOB_BASE}/upload-files"

MAX_POLL_ATTEMPTS = 60
POLL_INTERVAL_SECONDS = 3


class SarvamVisionClient:
    """Direct REST client for Sarvam Vision Document Intelligence.
    
    Note: Output download (ZIP file with extracted text) requires a SAS token
    which is not exposed via direct API endpoints. The MCP server handles
    SAS token generation automatically. For full text extraction, use
    MCPFallback.vision_extract() which routes through the MCP server.
    """

    def __init__(self, api_key: Optional[str] = None, base_url: str = SARVAM_BASE_URL):
        self.api_key = api_key or settings.sarvam_api_key
        self.base_url = base_url
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={"api-subscription-key": self.api_key},
                timeout=300.0,
            )
        return self._client

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _mime_type(self, path: Path) -> str:
        suffix = path.suffix.lower().lstrip(".")
        return {
            "pdf": "application/pdf",
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "webp": "image/webp",
            "tiff": "image/tiff",
            "zip": "application/zip",
        }.get(suffix, "application/octet-stream")

    async def submit_job(
        self,
        file_path: str,
        output_format: str = "md",
        language_code: str = "hi-IN",
    ) -> dict:
        """
        Submit a document for Vision OCR processing.
        
        Returns job details including job_id for polling.
        Output download requires MCP server (SAS token generation).
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        client = await self._get_client()

        # Step 1: Create job
        logger.info("Creating Document Intelligence job...")
        create_body = {
            "job_parameters": {
                "language": language_code if language_code != "unknown" else "hi-IN",
                "output_format": output_format,
            }
        }
        create_resp = await client.post(DOC_JOB_BASE, json=create_body)
        create_resp.raise_for_status()
        create_data = create_resp.json()
        job_id = create_data["job_id"]
        logger.info(f"Created job: {job_id}")

        # Step 2: Get upload URLs
        logger.info(f"Getting upload URL for job {job_id}...")
        upload_req = {"job_id": job_id, "files": [path.name]}
        upload_resp = await client.post(DOC_JOB_UPLOAD, json=upload_req)
        upload_resp.raise_for_status()
        upload_data = upload_resp.json()

        upload_urls = upload_data.get("upload_urls", {})
        if not upload_urls:
            raise RuntimeError(f"No upload URLs returned for job {job_id}")

        # Step 3: Upload file
        logger.info("Uploading document...")
        file_details = next(iter(upload_urls.values()))
        presigned_url = file_details["file_url"]
        file_metadata = file_details.get("file_metadata") or {}

        extra_headers = {str(k): str(v) for k, v in file_metadata.items()}
        extra_headers.setdefault("x-ms-blob-type", "BlockBlob")
        mime_type = await self._mime_type(path)
        
        with path.open("rb") as fh:
            file_content = fh.read()
        
        upload_client = httpx.AsyncClient(timeout=300.0)
        try:
            blob_resp = await upload_client.put(
                presigned_url,
                content=file_content,
                headers={"Content-Type": mime_type, **extra_headers},
            )
            blob_resp.raise_for_status()
        finally:
            await upload_client.aclose()

        # Step 4: Start job
        logger.info("Starting processing...")
        start_resp = await client.post(f"{DOC_JOB_BASE}/{job_id}/start", json={})
        start_resp.raise_for_status()

        return {
            "job_id": job_id,
            "job_state": "Pending",
            "storage_container_type": create_data.get("storage_container_type"),
            "output_format": output_format,
            "language_code": language_code,
        }

    async def poll_status(self, job_id: str) -> dict:
        """Poll job status until completion."""
        client = await self._get_client()
        
        for attempt in range(MAX_POLL_ATTEMPTS):
            status_resp = await client.get(f"{DOC_JOB_BASE}/{job_id}/status")
            status_resp.raise_for_status()
            status_data = status_resp.json()
            job_state = status_data.get("job_state", "")
            
            if job_state in {"Completed", "PartiallyCompleted", "Failed"}:
                return status_data
            
            if (attempt + 1) % 5 == 0:
                logger.info(f"Polling... attempt {attempt + 1}/{MAX_POLL_ATTEMPTS}")
            
            await asyncio.sleep(POLL_INTERVAL_SECONDS)
        
        # Timeout
        return {
            "job_id": job_id,
            "job_state": "timeout",
            "error": f"Job did not complete within {MAX_POLL_ATTEMPTS * POLL_INTERVAL_SECONDS}s",
        }

    async def get_job_result(self, job_id: str) -> dict:
        """
        Get job result. Note: Output download (ZIP with extracted text) 
        requires MCP server for SAS token generation.
        
        Returns job details including job_details with processing info.
        For full text extraction, use MCPFallback.vision_extract().
        """
        status_data = await self.poll_status(job_id)
        
        # Add note about output download
        status_data["_note"] = (
            "Output ZIP download requires MCP server for SAS token generation. "
            "Use MCPFallback.vision_extract() for full text extraction."
        )
        
        return status_data


async def vision_submit_job(
    file_path: str,
    output_format: str = "md",
    language_code: str = "hi-IN",
) -> dict:
    """Convenience function to submit a Vision OCR job."""
    client = SarvamVisionClient()
    try:
        return await client.submit_job(file_path, output_format, language_code)
    finally:
        await client.close()


async def vision_poll_status(job_id: str) -> dict:
    """Convenience function to poll Vision job status."""
    client = SarvamVisionClient()
    try:
        return await client.poll_status(job_id)
    finally:
        await client.close()


async def vision_get_result(job_id: str) -> dict:
    """Convenience function to get Vision job result."""
    client = SarvamVisionClient()
    try:
        return await client.get_job_result(job_id)
    finally:
        await client.close()