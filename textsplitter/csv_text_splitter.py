from langchain.text_splitter import TextSplitter
from typing import List, Any

class CSVTextSplitter(TextSplitter):
    """Implementation of splitting text that looks at characters."""

    def __init__(self, line_separator: str = "\n", quotechar: str ='"', **kwargs: Any):
        """Create a new CSVTextSplitter."""
        super().__init__(**kwargs)
        self._line_separator = line_separator
        self._quotechar = quotechar

    def split_text(self, text: str) -> List[str]:
        """Split incoming text and return chunks."""
        # First we naively split the large input into a bunch of smaller ones.
        return text.split(self._line_separator)
