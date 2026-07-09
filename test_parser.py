import os
from loguru import logger
from app.ingestion.loader.smart_document_parser import SmartDocumentParser

def main():
    # Target test file
    file_path = os.path.join("data", "EFESO_E2E_Inflation_Management.pdf")
    
    if not os.path.exists(file_path):
        logger.error(f"Test file not found at {file_path}. Please place your EFESO_Big_Data.pdf in the data directory.")
        return


    logger.info(f"Parsing document: {file_path}")
    try:
        parser = SmartDocumentParser(summarize_images=True)
        result = parser.parse(file_path)
        
        # Combine all page content into full markdown
        full_markdown = "\n\n---new page---\n\n".join(doc.page_content for doc in result)
        
        # Save output markdown
        output_path = "parsed_output.md"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_markdown)
            
        logger.info(f"Successfully parsed document! ({len(result)} pages)")
        
        # Inspect what one Document looks like (this is what the chunker receives)
        print("\n" + "="*80)
        print("DOCUMENT INSPECTION — What the chunker will receive")
        print("="*80)
        
        for i, doc in enumerate(result):
            page = doc.metadata["page_number"]
            num_images = len(doc.metadata.get("images", []))
            content_len = len(doc.page_content)
            print(f"\n--- Document[{i}] | Page {page} | {content_len} chars | {num_images} images ---")
            print(f"\n[page_content]:\n{doc.page_content}")
            print(f"\n[metadata keys]: {list(doc.metadata.keys())}")
            print(f"  source: {doc.metadata['source']}")
            print(f"  page_number: {doc.metadata['page_number']}")
            print(f"  total_pages: {doc.metadata['total_pages']}")
            if doc.metadata.get("images"):
                for img in doc.metadata["images"]:
                    summary = img.get("summary", "")
                    has_b64 = "yes" if img.get("image_base64") else "no"
                    print(f"  image: id={img['id']} | has_base64={has_b64} | summary={summary[:100]}{'...' if len(summary) > 100 else ''}")
        
        print("\n" + "="*80)
        
    except Exception as e:
        logger.error(f"Failed to parse document: {e}")

if __name__ == "__main__":
    main()
