from pdfminer.high_level import extract_text
p = r"C:\Users\keerthana\OneDrive\Desktop\PolicyLens\backend\media\documents\2025\10\28\resume_0.1.pdf"
print("Reading:", p)
try:
    txt = extract_text(p) or ""
    print("LEN:", len(txt))
    print("\n---- FIRST 800 CHARS ----\n")
    print(txt[:800].replace("\n", "\\n"))
    counts = {ord(c): txt.count(c) for c in set(txt) if ord(c) < 32 or ord(c) > 127}
    items = list(counts.items())[:10]
    print("\n---- NON-ASCII/CONTROL SAMPLE (ord,count) ----")
    print(items)
except Exception as e:
    print("ERROR:", e)
