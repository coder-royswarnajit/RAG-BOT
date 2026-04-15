import PyPDF2
import re


def load_documents(pdf_path):
    docs = []

    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)

        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()

            if not page_text:
                continue

            
            text = re.sub(r'\n+', '\n', page_text)
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'Page \d+', '', text)

            docs.append({
                "text": text.strip(),
                "source": pdf_path,
                "page": i + 1,
                "doc_id": f"{pdf_path}_page_{i+1}"
            })

    return docs