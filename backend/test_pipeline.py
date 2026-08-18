#!/usr/bin/env python3
"""Test full pipeline with document ingestion."""
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


async def test_pipeline():
    test_file = create_test_image()
    print(f"Created test file: {test_file}")
    
    from app.pipeline.graph import run_pipeline
    
    try:
        # Run pipeline with file_path (document ingestion mode)
        result = await run_pipeline(
            audio_bytes=b"",  # Empty audio - text-only mode
            language="hi",
            session_id="test_vision_123",
            transcript=None,  # Not used in document mode
            file_path=test_file,  # Document file for Vision OCR
        )
        
        print("Pipeline result:")
        print(f"  extracted_text: {result.get('extracted_text', '')[:200]}...")
        print(f"  vision_job_id: {result.get('vision_job_id')}")
        print(f"  vision_output_url: {result.get('vision_output_url')}")
        print(f"  ingested_chunks: {result.get('ingested_chunks')}")
        print(f"  ingestion_language: {result.get('ingestion_language')}")
        print(f"  vision_error: {result.get('vision_error')}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(test_file):
            os.unlink(test_file)


if __name__ == "__main__":
    asyncio.run(test_pipeline())
