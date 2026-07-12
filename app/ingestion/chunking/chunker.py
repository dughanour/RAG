import os
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from loguru import logger

class Chunker:
    """
    Structure-aware document chunker for multimodal RAG.
    Pipeline (3 passes):
        1. Split markdown by headers (#, ##, ###)  ← THIS STEP
        2. Size guard — further split large sections (next step)
        3. Prepend contextual headers to each chunk (next step)
    Input:  list[Document] from SmartDocumentParser (one per page)
    Output: list[Document] chunked and enriched, ready for embedding
    """
    def __init__(
        self,
        chunk_size=800,
        chunk_overlap= 150,
        headers_to_split_on: list[tuple[str,str]] | None = None
        ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.headers_to_split_on = headers_to_split_on

        if headers_to_split_on is None:
            self.headers_to_split_on = [
                ("#", "header1"),
                ("##", "header2"),
                ("###", "header3"),
            ]
        
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            strip_headers = False  # Keep headers in the content — we want them in the chunk text
        )

        logger.info(
            "Chunker initialized | chunk_size={} | overlap={}",
            chunk_size, chunk_overlap,
        )
    
    def chunk(self, documents:list[Document]) -> list[Document]:
        """
        Chunk a list of page-level Documents into smaller, retrieval-ready Documents.
        Args:
            documents: Output from SmartDocumentParser.parse() — one Document per page.
        Returns:
            List of chunked Documents with preserved metadata.
        """

        all_chunks: list[Document] = []

        for doc in documents:
            page_chunks = self.split_by_headers(doc)
            all_chunks.extend(page_chunks)
        
        logger.info(
            "Chunking complete — {} pages → {} chunks",
            len(documents), len(all_chunks),
        )
        return all_chunks
    
    # ---- Pass 1: Split by Markdown Headers ----
    def split_by_headers(self, doc: Document) -> list[Document]:
        """
        Split a single page Document by its markdown headers (#, ##, ###).
        Each resulting chunk inherits the original document's metadata
        (source, page_number, images, etc.) plus the header hierarchy
        from the split.
        """

        header_splits = self.markdown_splitter.split_text(doc.page_content)

        chunks: list[Document] = []
        for split in header_splits:
           # Merge the original page metadata with the header metadata from the split
           merged_metadata = {**doc.metadata, **split.metadata}

           chunks.append(
            Document(
            page_content = split.page_content,
            metadata=merged_metadata,
           )
           )
        
        # If the splitter produced nothing (no headers found), keep the original doc
        if not chunks:
            chunks.append(doc)
        
        return chunks

            
