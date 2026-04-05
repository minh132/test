import os
import glob
from typing import List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentChunker:
    """
    Reads text files from a directory and splits them into smaller chunks.
    Uses a simple word-based sliding-window strategy.
    """

    def __init__(self, chunk_size: int = 150, overlap: int = 20):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def process_directory(self, directory_path: str) -> List[str]:
        """Reads all .txt files in a directory and returns a list of text chunks."""
        if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
            raise ValueError(f"Invalid directory path: {directory_path}")

        file_paths = glob.glob(os.path.join(directory_path, "*.txt"))
        all_chunks = []

        for file_path in file_paths:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                all_chunks.extend(self._chunk_text(text))

        return all_chunks

    def _chunk_text(self, text: str) -> List[str]:
        """Splits a single text string into overlapping chunks based on words."""
        words = text.split()
        chunks = []

        if not words:
            return chunks

        i = 0
        while i < len(words):
            chunk_words = words[i : i + self.chunk_size]
            chunks.append(" ".join(chunk_words))
            i += self.chunk_size - self.overlap

        return chunks

    def chunk_text(self, text: str) -> List[str]:
        """Public alias for _chunk_text. Same interface as RecursiveChunker."""
        return self._chunk_text(text)


class RecursiveChunker:
    """
    Reads text files from a directory and splits them using LangChain's
    RecursiveCharacterTextSplitter.

    The splitter tries each separator in order (paragraph → newline → space →
    single character) so that natural text boundaries are preferred.

    Args:
        chunk_size:  Maximum number of *characters* per chunk (default 500).
        chunk_overlap: Number of *characters* that consecutive chunks share (default 50).
        separators:  Custom list of separators. Leave as None to use LangChain defaults
                     (["\\n\\n", "\\n", " ", ""]).
    """

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
        """Reads all .txt files in a directory and returns a list of text chunks."""
        if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
            raise ValueError(f"Invalid directory path: {directory_path}")

        file_paths = glob.glob(os.path.join(directory_path, "*.txt"))
        all_chunks: List[str] = []

        for file_path in file_paths:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            all_chunks.extend(self.splitter.split_text(text))

        return all_chunks

    def chunk_text(self, text: str) -> List[str]:
        """Splits a single string into chunks. Useful for ad-hoc usage."""
        return self.splitter.split_text(text)
