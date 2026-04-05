import glob
import os
from typing import List

import fitz


class PDFIngester:
    def __init__(self, strip_whitespace: bool = True, page_separator: str = "\n\n"):
        self.strip_whitespace = strip_whitespace
        self.page_separator = page_separator

    def extract_text(self, pdf_path: str) -> str:
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
        if not os.path.isdir(directory_path):
            raise ValueError(f"Invalid directory: {directory_path}")
        pdf_paths = sorted(glob.glob(os.path.join(directory_path, "*.pdf")))
        return [self.extract_text(p) for p in pdf_paths]

    def ingest_and_chunk(self, directory_path: str, chunker) -> List[str]:
        all_chunks: List[str] = []
        for text in self.ingest_directory(directory_path):
            all_chunks.extend(chunker.chunk_text(text))
        return all_chunks
