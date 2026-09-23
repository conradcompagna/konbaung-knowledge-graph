"""Recover only CRC-verified complete local members of a truncated nested ZIP."""

from pathlib import Path
import zipfile, struct, zlib, json, hashlib, sys

sys.stdout.reconfigure(encoding="utf-8")
OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parent
target = OUT / "recovered_source"
target.mkdir(exist_ok=True)
archive = next(ROOT.glob("*FINAL_Statistical*.zip"))
with zipfile.ZipFile(archive) as z:
    name = next(n for n in z.namelist() if "DIGHUM_WEBGPT" in n)
    nested_name = name
    raw = z.read(name)
pos = 0
records = []
failure = None
while raw[pos : pos + 4] == b"PK\x03\x04":
    start = pos
    _, ver, flag, method, tm, dt, crc, csize, usize, nlen, xlen = struct.unpack_from(
        "<4s5H3I2H", raw, pos
    )
    name = raw[pos + 30 : pos + 30 + nlen].decode("utf-8")
    pos += 30 + nlen + xlen
    if flag & 8:
        failure = f"Data descriptor at {name}; not recovered"
        break
    if pos + csize > len(raw):
        failure = f"Truncated member {name}: requires {csize} bytes, only {len(raw) - pos} remain"
        break
    compressed = raw[pos : pos + csize]
    pos += csize
    if method == 8:
        dec = zlib.decompressobj(-15)
        body = dec.decompress(compressed) + dec.flush()
        assert dec.eof and not dec.unused_data, name
    elif method == 0:
        body = compressed
    else:
        raise ValueError((name, method))
    assert len(body) == usize and zlib.crc32(body) & 0xFFFFFFFF == crc, name
    dest = (target / name).resolve()
    assert dest.is_relative_to(target.resolve())
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(body)
    r = {
        "member": name,
        "size": usize,
        "sha256": hashlib.sha256(body).hexdigest(),
        "crc_verified": True,
        "offset": start,
    }
    records.append(r)
    print(name, usize, flush=True)
manifest = target / "CHECKSUMS.sha256"
if manifest.exists():
    expected = {}
    for line in manifest.read_text().splitlines():
        h, n = line.split(maxsplit=1)
        expected[n.lstrip("*")] = h
    for r in records:
        r["manifest_sha256_matches"] = (
            expected.get(r["member"]) == r["sha256"] if r["member"] in expected else None
        )
        assert r["manifest_sha256_matches"] is not False, r["member"]
summary = {
    "container": str(archive.name),
    "nested_member": nested_name,
    "complete_members": records,
    "failure": failure,
    "bytes_total": len(raw),
    "stopped_at": pos,
}
(OUT / "source_recovery.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("RECOVERY", len(records), failure, flush=True)
