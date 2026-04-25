# Research Folder Extraction Script

## What is this?

This is a Python script designed to extract content from files stored in a Google Drive folder named **"Research"** and convert them into a structured JSON file.

---

## What does it do?

The script performs the following steps:

1. Connects to Google Drive
2. Finds the "Research" folder
3. Reads all supported files inside the folder
4. Extracts:
   - Text content
   - Tables
5. Converts tables into **Markdown format**
6. Stores everything in a structured JSON file
7. Uploads the JSON file back to the same folder

---

## Supported File Types

- PDF (.pdf)
- Word Documents (.docx)
- Google Docs

---

## Output

The script generates:

research_context.json

This file contains:
- File metadata (name, ID, link)
- Extracted text
- Tables (as rows + Markdown)

---

## Technologies Used

### Google Drive API
- Used to read, download, and upload files

### PyMuPDF (fitz)
- Extracts text from PDF files
- Fast and efficient

### pdfplumber
- Extracts tables from PDFs

### python-docx
- Reads Word documents (.docx)
- Extracts text and tables

### Google Colab Authentication
- Allows access to your Google Drive

---

## Table Handling

Tables are converted into Markdown format for better readability and compatibility with LLMs.

Example:

| Metric | Value |
|---|---|
| Accuracy | 92% |

---

## Purpose

This script is used as a data extraction layer for:

- Preparing data for LLMs
- Building vector databases
- Creating structured knowledge bases

---

## Notes

- Tables are preserved in both raw and Markdown format
- Scanned PDFs may not extract perfectly (OCR not included)
- Old .doc files are not supported

