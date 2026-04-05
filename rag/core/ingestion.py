import glob
import os
from typing import List

import fitz  # PyMuPDF


class PDFIngester:
    """
    Ingests PDF files from a directory and extracts their text content.

    Uses PyMuPDF (fitz) for fast, accurate text extraction.
    The extracted text can then be passed to any chunker
    (e.g. ``DocumentChunker`` or ``RecursiveChunker``).

    Args:
        strip_whitespace: If True (default), strip leading/trailing whitespace
                          from each page's text before joining.
        page_separator:   String inserted between pages. Default is a double
                          newline so paragraph-aware chunkers can split on it.
    """

    def __init__(
        self,
        strip_whitespace: bool = True,
        page_separator: str = "\n\n",
    ):
        self.strip_whitespace = strip_whitespace
        self.page_separator = page_separator

    # ── Public API ────────────────────────────────────────────────────────────

    def extract_text(self, pdf_path: str) -> str:
        """Extracts and returns all text from a single PDF file."""
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc = fitz.open(pdf_path)
        pages: List[str] = []
        for page in doc:
            text = page.get_text()
            if self.strip_whitespace:
                text = text.strip()
            if text:
                pages.append(text)
        doc.close()

        return self.page_separator.join(pages)

    def ingest_directory(self, directory_path: str) -> List[str]:
        """
        Extracts text from every PDF in *directory_path*.

        Returns:
            A list where each item is the full extracted text of one document.
            Pass each item to your chunker's ``chunk_text`` / ``split_text``
            method, or use ``ingest_and_chunk`` for a one-shot workflow.
        """
        if not os.path.isdir(directory_path):
            raise ValueError(f"Invalid directory: {directory_path}")

        pdf_paths = sorted(glob.glob(os.path.join(directory_path, "*.pdf")))
        return [self.extract_text(p) for p in pdf_paths]

    def ingest_and_chunk(self, directory_path: str, chunker) -> List[str]:
        """
        Convenience method: ingest all PDFs in *directory_path* and split them
        with *chunker*.

        *chunker* must expose a ``chunk_text(text: str) -> List[str]`` method
        (``DocumentChunker`` and ``RecursiveChunker`` both do).

        Returns:
            A flat list of text chunks across all documents.
        """
        all_chunks: List[str] = []
        for text in self.ingest_directory(directory_path):
            all_chunks.extend(chunker.chunk_text(text))
        return all_chunks
