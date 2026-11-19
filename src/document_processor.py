
from pathlib import Path
import json
import re

import fitz  # PyMuPDF


# Project directories
POLICY_DIR = Path("documents/policies")
OUTPUT_DIR = Path("artifacts/vector_index")
OUTPUT_FILE = OUTPUT_DIR / "chunks.json"

# Chunk configuration
CHUNK_WORDS = 180
OVERLAP_WORDS = 40


def clean_text(text: str) -> str:
    """Clean extracted PDF text."""
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_into_chunks(text: str):
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []

    start = 0

    while start < len(words):
        end = min(start + CHUNK_WORDS, len(words))

        chunk = " ".join(words[start:end]).strip()

        if chunk:
            chunks.append(chunk)

        if end == len(words):
            break

        start = end - OVERLAP_WORDS

    return chunks


def process_pdfs():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(POLICY_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            "No PDF files found in documents/policies/"
        )

    all_chunks = []
    chunk_number = 1
    extracted_pages = 0

    for pdf_path in pdf_files:

        print(f"\nReading PDF: {pdf_path.name}")

        try:
            with fitz.open(pdf_path) as pdf:

                print("Total pages:", len(pdf))

                for page_number, page in enumerate(pdf, start=1):

                    text = clean_text(page.get_text("text"))

                    if not text:
                        print(
                            f"Warning: No extractable text "
                            f"on page {page_number}"
                        )
                        continue

                    extracted_pages += 1

                    page_chunks = split_into_chunks(text)

                    for chunk_text in page_chunks:

                        all_chunks.append({
                            "chunk_id": f"CH{chunk_number:05d}",
                            "text": chunk_text,
                            "source": pdf_path.name,
                            "page": page_number
                        })

                        chunk_number += 1

        except Exception as error:
            raise RuntimeError(
                f"Could not process {pdf_path.name}: {error}"
            ) from error

    if not all_chunks:
        raise ValueError(
            "No text was extracted from the PDFs. "
            "Check whether the document is scanned."
        )

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(
            all_chunks,
            file,
            ensure_ascii=False,
            indent=2
        )

    print("\n========== EXTRACTION SUMMARY ==========")
    print("PDF files processed:", len(pdf_files))
    print("Pages with extracted text:", extracted_pages)
    print("Total text chunks:", len(all_chunks))
    print("Output file:", OUTPUT_FILE)
    print("========================================")

    print("\nFirst extracted passage:")
    print(all_chunks[0]["text"][:500])


if __name__ == "__main__":
    process_pdfs()
