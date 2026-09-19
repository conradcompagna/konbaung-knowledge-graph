import os
import base64
import json
import time
from pathlib import Path
import requests

def load_api_key():
    env_path = Path(__file__).resolve().parent.parent / ".env"
    for line in env_path.read_text().splitlines():
        if line.startswith("GOOGLE_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("GOOGLE_API_KEY not found in .env")

def ocr_image(image_path: Path, api_key: str, language_hint: str = "my"):
    url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
    content = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    body = {
        "requests": [
            {
                "image": {"content": content},
                "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
                "imageContext": {"languageHints": [language_hint]},
            }
        ]
    }
    resp = requests.post(url, json=body, timeout=60)
    resp.raise_for_status()
    return resp.json()

def run(book_id: str, max_pages: int = None, force: bool = False, language_hint: str = "my"):
    base = Path(__file__).resolve().parent.parent / "data"
    pages_dir = base / "rendered_pages" / book_id
    json_dir = base / "ocr_json" / book_id
    json_dir.mkdir(parents=True, exist_ok=True)

    api_key = load_api_key()
    pages = sorted(pages_dir.glob("page_*.png"))
    if max_pages:
        pages = pages[:max_pages]

    for page_path in pages:
        out_path = json_dir / (page_path.stem + ".json")
        if out_path.exists() and not force:
            print(f"skip (exists): {out_path}")
            continue

        for attempt in range(4):
            try:
                result = ocr_image(page_path, api_key, language_hint)
                break
            except requests.HTTPError as e:
                if attempt == 3:
                    raise
                wait = 2 ** attempt
                print(f"error on {page_path}, retrying in {wait}s: {e}")
                time.sleep(wait)

        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        text = result.get("responses", [{}])[0].get("fullTextAnnotation", {}).get("text", "")
        print(f"OCR'd {page_path.name}: {len(text)} chars")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--book-id", required=True)
    ap.add_argument("--max-pages", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--language-hint", default="my")
    args = ap.parse_args()
    run(args.book_id, args.max_pages, args.force, args.language_hint)
