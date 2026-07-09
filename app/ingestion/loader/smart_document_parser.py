
import os
import re

from loguru import logger
from mistralai import Mistral
from mistralai.models import FileChunk
from langchain_core.documents import Document
from app.config import settings

# Prompt used to generate image descriptions via VLM
# The VLM classifies + summarizes in a single call.
# Non-informative images (logos, covers, decorative) get "SKIP" → filtered out.
_IMAGE_SUMMARY_PROMPT = (
    "Analyze this image extracted from a document.\n\n"
    "FIRST, classify it. Respond with exactly SKIP (nothing else) if the image is:\n"
    "- A cover image\n"
    "- A company logo\n"
    "- A page that only shows company name, tagline, or contact info\n\n"
    "other than that make brief summary about the image if contains real, substantive data (actual charts with readable "
    "values, diagrams with technical content, tables with data, or informative "
    "illustrations), then describe it — include data values, labels, "
    "trends, and any visible text. Be concise but thorough. Respond in plain text only."
)

class SmartDocumentParser:
    """
    Multimodal document parser using Mistral OCR 4.
    Pipeline: Upload file → OCR extraction → Process pages (+ VLM image summaries)
    Output:   list[Document] ready for chunking
    """
    def __init__(self, summarize_images: bool = True) -> None:
        self._client = Mistral(api_key=settings.MISTRAL_API_KEY)
        self._summarize_images = summarize_images
        logger.info("SmartDocumentParser initialized | summarize_images={}", summarize_images)

    def parse(self, file_path: str) -> list[Document]:
        """Parse a document into a list of LangChain Documents (one per page)."""

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at: {file_path}")
        
        logger.info("Parsing document: {}", file_path)

        # 1. Upload file to Mistral
        file_id = self._upload_file(file_path)

        # 2. Run OCR
        ocr_response = self._run_ocr(file_id)
        total_pages = len(ocr_response.pages)
        logger.info("OCR complete — {} pages extracted", total_pages)

         # 3. Process each page into a LangChain Document
        documents = []
        for page in ocr_response.pages:
            doc = self._process_page(page, file_path, total_pages)
            documents.append(doc)
        logger.info("Done — {} Documents produced", len(documents))
        return documents
    
    def _upload_file(self, file_path: str) -> str:
        """Upload a file to Mistral and return the file_id."""
        with open(file_path, "rb") as f:
            uploaded = self._client.files.upload(
                file={
                    "file_name": os.path.basename(file_path),
                    "content": f,
                },
                purpose="ocr",
            )
        logger.debug("File uploaded — file_id={}", uploaded.id)
        return uploaded.id
    
    def _run_ocr(self, file_id: str):
        """Run Mistral OCR 4 on an uploaded file."""
        return self._client.ocr.process(
            model="mistral-ocr-4-0",
            document=FileChunk(file_id=file_id),
            include_image_base64=True,
            table_format="markdown",
        )
    
    def _process_page(self, page, file_path: str, total_pages: int) -> Document:
        """Convert a single OCR page into a LangChain Document."""
        page_number = page.index + 1
        markdown = page.markdown or ""
        # Extract images from this page
        images = self._extract_images(page)
        # If VLM summarization is enabled, summarize each image
        # and inject the summary into the markdown
        if self._summarize_images and images:
            markdown = self._inject_image_summaries(markdown, images, page_number)
        return Document(
            page_content=markdown,
            metadata={
                "source": file_path,
                "page_number": page_number,
                "total_pages": total_pages,
                "images": images,    # list of {"id": ..., "image_base64": ..., "summary": ...}
            },
        )

         # ---- Private: Image Handling ----
    def _extract_images(self, page) -> list[dict]:
        """Extract images from an OCR page as simple dicts."""
        images = []
        for img in getattr(page, "images", []) or []:
            image_base64 = getattr(img, "image_base64", None)
            # SDK returns UNSET sentinel for missing values — normalize to None
            if not isinstance(image_base64, str):
                image_base64 = None
            images.append({
                "id": img.id,
                "image_base64": image_base64,
                "summary": "",
            })
        return images
    
    def _inject_image_summaries(
        self, markdown: str, images: list[dict], page_number: int
    ) -> str:
        """Summarize each image via VLM and replace its markdown placeholder."""
        for image in images:
            if not image["image_base64"]:
                continue
            summary = self._summarize_image(image["image_base64"])
            image["summary"] = summary
            if summary:
                # Replace ![img_id](base64_url) with [Image: summary]
                pattern = re.compile(
                    rf"!\[{re.escape(image['id'])}\]\([^)]*\)", re.DOTALL
                )
                markdown = pattern.sub(f"[Image: {summary}]", markdown)
                logger.debug("Page {} — image {} summarized", page_number, image["id"])
        return markdown

    def _summarize_image(self, image_base64: str) -> str:
        """Call Mistral Pixtral VLM to describe an image. Returns '' on failure."""
        try:
            # OCR may return base64 with or without the data URL prefix
            if image_base64.startswith("data:image"):
                image_url = image_base64
            else:
                mime = "image/jpeg" if image_base64.startswith("/9j/") else "image/png"
                image_url = f"data:{mime};base64,{image_base64}"
            
            response = self._client.chat.complete(
                model="mistral-small-latest",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": _IMAGE_SUMMARY_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": image_url,
                            },
                        ],
                    }
                ],
                max_tokens=300,
            )
            result = response.choices[0].message.content.strip()

            # VLM classified this image as non-informative
            if result.upper().startswith("SKIP"):
                logger.debug("Image classified as non-informative — skipped")
                return ""

            return result
        except Exception as e:
            logger.warning("VLM summarization failed: {}", e)
            return ""
