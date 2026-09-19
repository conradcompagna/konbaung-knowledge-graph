import fitz
base = r"C:\Users\conra\Documents\primary texts\ocr_out"
files = [
    ("ocr_konbaungsetvol2.pdf", "vol2"),
    ("ocr_konbaungsetvol3.pdf", "vol3"),
]
for f, name in files:
    d = fitz.open(base + "\\" + f)
    print(name, len(d))
