import os
import glob
from typing import List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentChunker:
    def __init__(self, chunk_size: int = 150, overlap: int = 20):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def process_directory(self, directory_path: str) -> List[str]:
        if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
            raise ValueError(f"Invalid directory path: {directory_path}")

        all_chunks = []
        for file_path in glob.glob(os.path.join(directory_path, "*.txt")):
            with open(file_path, "r", encoding="utf-8") as f:
                all_chunks.extend(self.chunk_text(f.read()))
        return all_chunks

    def chunk_text(self, text: str) -> List[str]:
        words = text.split()
        if not words:
            return []
        chunks = []
        i = 0
        while i < len(words):
            chunks.append(" ".join(words[i:i + self.chunk_size]))
            i += self.chunk_size - self.overlap
        return chunks


class RecursiveChunker:
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None,
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators or ["\n\n", "\n", " ", ""],
            length_function=len,
            is_separator_regex=False,
        )

    def process_directory(self, directory_path: str) -> List[str]:
        if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
            raise ValueError(f"Invalid directory path: {directory_path}")

        all_chunks: List[str] = []
        for file_path in glob.glob(os.path.join(directory_path, "*.txt")):
            with open(file_path, "r", encoding="utf-8") as f:
                all_chunks.extend(self.splitter.split_text(f.read()))
        return all_chunks

    def chunk_text(self, text: str) -> List[str]:
        return self.splitter.split_text(text)
