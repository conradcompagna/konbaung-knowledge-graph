"""Exercise the production-critical routes without binding a network port."""

from app import app


paths = [
    "/chronicles/vol1/47",
    "/knowledge-graph/vol1/47",
    "/api/v1/health",
    "/api/graph/stats",
    "/chronicle-assets/chronicles.css",
]

client = app.test_client()
results = [(path, client.get(path).status_code) for path in paths]
results.append(
    (
        "/api/chronicles/segment",
        client.post(
            "/api/chronicles/segment",
            json={"volumeId": "vol1", "pageNumber": 47},
        ).status_code,
    )
)
print(results)

if any(status != 200 for _, status in results):
    raise SystemExit(1)
