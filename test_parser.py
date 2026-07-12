import os
from loguru import logger
from app.ingestion.loader.smart_document_parser import SmartDocumentParser
from app.ingestion.chunking.chunker import Chunker

def main():
    # Target test file
    file_path = os.path.join("data", "EFESO_E2E_Inflation_Management.pdf")
    
    if not os.path.exists(file_path):
        logger.error(f"Test file not found at {file_path}")
        return

    try:
        # Step 1: Parse
        parser = SmartDocumentParser(summarize_images=True)
        docs = parser.parse(file_path)
        logger.info(f"Parsed into {len(docs)} page Documents")

        # Save parsed output as markdown for inspection
        full_markdown = "\n\n---new page---\n\n".join(doc.page_content for doc in docs)
        with open("parsed_output.md", "w", encoding="utf-8") as f:
            f.write(full_markdown)
        logger.info("Saved parsed markdown to: parsed_output.md")

        # Step 2: Chunk
        chunker = Chunker()
        chunks = chunker.chunk(docs)

        # Print each chunk
        print("\n" + "="*80)
        print(f"CHUNKS — {len(chunks)} total from {len(docs)} pages")
        print("="*80)

        for i, chunk in enumerate(chunks):
            page = chunk.metadata.get("page_number", "?")
            h1 = chunk.metadata.get("header1", "")
            h2 = chunk.metadata.get("header2", "")
            h3 = chunk.metadata.get("header3", "")
            header_path = " > ".join(filter(None, [h1, h2, h3]))
            content_len = len(chunk.page_content)

            print(f"\n--- Chunk[{i}] | Page {page} | {content_len} chars ---")
            if header_path:
                print(f"[Headers]: {header_path}")
            print(f"[Content]:\n{chunk.page_content[:300]}{'...' if content_len > 300 else ''}")
            print("-"*60)

        print("\n" + "="*80)

    except Exception as e:
        logger.error(f"Failed: {e}")

if __name__ == "__main__":
    main()

