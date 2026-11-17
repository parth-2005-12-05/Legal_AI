"""Document ingestion (design: "Document ingestion and OCR").

    sanitiser.py   real file type by magic bytes, size / page caps, active-content
                   detection in PDFs, optional ClamAV scan
    text_layer.py  born-digital shortcut: use a PDF's own text layer and skip OCR
    ocr.py         OCR engine adapters (Tesseract optional) and the ROVER-style
                   word vote that combines two engines
"""
