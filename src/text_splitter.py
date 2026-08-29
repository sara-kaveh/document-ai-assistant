import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class SectionAwareSplitter:
    """Splits documents into chunks while preserving section metadata."""

    def __init__(
        self,
        chunk_size=800,
        chunk_overlap=150
    ):
        self.chunk_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def split_documents(self, documents):
        """Split documents by section and create overlapping text chunks."""

        all_chunks = []
        current_section = "Unknown"

        for document in documents:
            text = document.page_content
            lines = text.splitlines()

            current_text = []

            for line in lines:
                line = line.strip()

                if not line:
                    continue

                # Detect numbered section headings such as "1 Introduction".
                if self._is_section_heading(line):

                    # Split the content accumulated under the previous section.
                    if current_text:
                        section_text = "\n".join(current_text)

                        chunks = self.chunk_splitter.split_text(
                            section_text
                        )

                        for chunk in chunks:
                            metadata = document.metadata.copy()
                            metadata["section"] = current_section

                            all_chunks.append(
                                Document(
                                    page_content=chunk,
                                    metadata=metadata
                                )
                            )

                    # Store the new heading for subsequent chunks.
                    current_section = line
                    current_text = []

                else:
                    current_text.append(line)

            # Process any content remaining after the last section heading.
            if current_text:
                section_text = "\n".join(current_text)

                chunks = self.chunk_splitter.split_text(
                    section_text
                )

                for chunk in chunks:
                    metadata = document.metadata.copy()
                    metadata["section"] = current_section

                    all_chunks.append(
                        Document(
                            page_content=chunk,
                            metadata=metadata
                        )
                    )

        return all_chunks

    def _is_section_heading(self, line):
        """Return True when a line matches a numbered section heading."""

        pattern = r"^\d+(\.\d+)*\s+.+"

        return bool(re.match(pattern, line))
