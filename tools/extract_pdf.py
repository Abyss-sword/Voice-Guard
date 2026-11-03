import sys
from pypdf import PdfReader

r = PdfReader(sys.argv[1])
print("pages:", len(r.pages))
for i, pg in enumerate(r.pages, 1):
    print(f"\n\n=========== PAGE {i} ===========")
    print(pg.extract_text())
