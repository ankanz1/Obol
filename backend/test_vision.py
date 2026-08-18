#!/usr/bin/env python3
"""Test Sarvam Vision job submission and polling via direct REST API."""
import asyncio
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

def create_test_image():
    img = Image.new('RGB', (800, 600), color='white')
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.load_default()
    except:
        font = None
    
    text = """Sarvam AI Vision Test Document
    
This is a test document for Sarvam Vision OCR extraction.
It contains multiple lines of text in English.

नमस्ते! यह एक हिंदी टेस्ट है।
नमस्ते दुनिया!

Key Information:
- Company: Sarvam AI
- Product: Vision Document Intelligence
- Languages: 23 Indian languages
- Features: Table preservation, multi-page support

Contact: support@sarvam.ai
Website: https://sarvam.ai
"""
    draw.text((50, 50), text, fill='black', font=font)
    
    path = Path("/tmp/sarvam_test_doc.png")
    img.save(path)
    return str(path)


async def test_vision():
    test_file = create_test_image()
    print(f"Created test file: {test_file}")
    
    from app.mcp.vision_direct import vision_submit_job, vision_poll_status
    
    try:
        # Submit job
        print("Submitting job...")
        job = await vision_submit_job(test_file, "md", "hi-IN")
        print(f"Job submitted: {job}")
        job_id = job["job_id"]
        
        # Poll status
        print("Polling status...")
        result = await vision_poll_status(job_id)
        print("Job completed:")
        for k, v in result.items():
            if k != "job_details":
                print(f"  {k}: {v}")
            else:
                print(f"  {k}: {len(v)} items")
                for detail in v:
                    print(f"    {detail}")
        
        print("\nNote: Output download requires MCP server for SAS token generation.")
        print("Use MCPFallback.vision_extract() for full text extraction.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(test_file):
            os.unlink(test_file)


if __name__ == "__main__":
    asyncio.run(test_vision())
