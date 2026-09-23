"""Validate report links and save checksums of the deliverable outputs."""

from pathlib import Path
from html.parser import HTMLParser
import re, hashlib, json

O = Path(__file__).resolve().parents[1]


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.ids = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "a" and a.get("href"):
            self.links.append(a["href"])
        if tag == "img":
            self.links.append(a["src"])
        if "id" in a:
            self.ids.append(a["id"])


q = Links()
q.feed((O / "index.html").read_text(encoding="utf-8"))
r = Links()
r.feed((O / "source_passages.html").read_text(encoding="utf-8"))
for link in q.links + r.links:
    if "://" in link:
        continue
    name, _, anchor = link.partition("#")
    assert (O / name).exists(), link
    if name == "source_passages.html" and anchor:
        assert anchor in r.ids, link
assert len(q.ids) == len(set(q.ids))
js = re.search(r"<script>([\s\S]*?)</script>", (O / "index.html").read_text(encoding="utf-8"))[1]
(O / "scripts/report_ui.js").write_text(js, encoding="utf-8")
checks = []
for folder in ["scripts", "results", "figures"]:
    for f in sorted((O / folder).glob("*")):
        if f.is_file():
            checks.append(
                hashlib.sha256(f.read_bytes()).hexdigest() + "  " + f.relative_to(O).as_posix()
            )
for pattern in ["*.md", "*.html", "*.ps1"]:
    for f in sorted(O.glob(pattern)):
        checks.append(hashlib.sha256(f.read_bytes()).hexdigest() + "  " + f.name)
(O / "OUTPUT_CHECKSUMS.sha256").write_text("\n".join(checks) + "\n")
print("Report targets and source anchors verified; output checksums:", len(checks))
