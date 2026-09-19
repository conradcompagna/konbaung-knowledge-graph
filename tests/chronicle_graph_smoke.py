#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path

from playwright.sync_api import sync_playwright


BASE_URL = os.environ.get("READER_BASE_URL", "http://127.0.0.1:5077")
ARTIFACT = (
    Path(__file__).resolve().parents[1]
    / "konbaung_reader_app"
    / "test-artifacts"
    / "reader-knowledge-graph.png"
)
CORPUS_ARTIFACT = ARTIFACT.with_name("reader-full-corpus-graph.png")


def main() -> None:
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1600, "height": 1000})
        errors: list[str] = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on(
            "response",
            lambda response: (
                errors.append(f"{response.status} {response.url}")
                if response.status >= 400
                and not response.url.endswith("/favicon.ico")
                else None
            ),
        )
        try:
            page.goto(f"{BASE_URL}/chronicles/vol1/47", wait_until="networkidle")
            page.locator("#graph-toggle").click()
            page.locator("#graph-workspace").wait_for(state="visible")
            page.wait_for_function(
                "() => document.querySelector('#graph-counts').textContent"
                ".includes('20 nodes')"
            )
            initial = page.evaluate(
                """() => ({
                  readerHidden: document.querySelector('#workspace').hidden,
                  graphVisible: !document.querySelector('#graph-workspace').hidden,
                  counts: document.querySelector('#graph-counts').textContent,
                  canvasWidth: document.querySelector('#graph-canvas .sigma-mouse').width,
                  canvasHeight: document.querySelector('#graph-canvas .sigma-mouse').height,
                  canvases: Array.from(
                    document.querySelectorAll('#graph-canvas canvas')
                  ).map(canvas => canvas.className)
                })"""
            )
            if (
                not initial["readerHidden"]
                or not initial["graphVisible"]
                or initial["canvasWidth"] <= 0
                or initial["canvasHeight"] <= 0
                or "sigma-collisionLabels" not in initial["canvases"]
                or "sigma-atlasRegions" in initial["canvases"]
            ):
                raise AssertionError(f"Graph workspace did not initialize: {initial}")

            page.locator("#graph-scope").select_option("vol1")
            page.wait_for_function(
                "() => document.querySelector('#graph-counts').textContent"
                ".includes('7,566 nodes · 8,096 relations · 8,147 claims')"
            )
            page.locator("#graph-scope").select_option("corpus")
            page.wait_for_function(
                "() => document.querySelector('#graph-counts').textContent"
                ".includes('23,890 nodes · 26,867 relations · 27,129 claims')"
            )
            atlas_canvases = page.locator("#graph-canvas canvas").evaluate_all(
                "(canvases) => canvases.map(canvas => canvas.className)"
            )
            if (
                "sigma-atlasRegions" not in atlas_canvases
                or "sigma-collisionLabels" in atlas_canvases
            ):
                raise AssertionError(
                    f"Corpus did not use the isolated atlas renderer: {atlas_canvases}"
                )
            for _ in range(5):
                page.locator("#graph-zoom-in").click()
            page.locator("#graph-zoom-out").click()
            page.locator("#graph-zoom-in").click()
            page.wait_for_timeout(300)
            page.screenshot(path=str(CORPUS_ARTIFACT), full_page=True)
            corpus = {
                "heading": page.locator("#graph-heading").inner_text(),
                "counts": page.locator("#graph-counts").inner_text(),
                "scope": page.locator("#graph-scope").input_value(),
            }
            page.locator("#graph-scope").select_option("page")
            page.wait_for_function(
                "() => document.querySelector('#graph-counts').textContent"
                ".includes('20 nodes · 16 relations · 16 claims')"
            )

            page.locator("details.graph-panel").first.evaluate(
                "(panel) => { panel.open = true; }"
            )
            page.locator("#graph-kind").select_option("entity")
            page.locator("#graph-search").fill("Alaungpaya")
            page.locator("#graph-search-button").click()
            page.locator(".graph-result").first.wait_for()
            page.locator(".graph-result", has_text="Alaungpaya").first.click()
            page.wait_for_function(
                "() => document.querySelector('#graph-heading').textContent === 'Alaungpaya'"
            )
            page.locator("#graph-sort").select_option("gemini")
            page.wait_for_function(
                "() => Array.from(document.querySelectorAll('.graph-result-meta'))"
                ".some(item => item.textContent.includes('% Gemini'))"
            )
            semantic = page.locator(".graph-result-meta").first.inner_text()

            page.locator("#graph-close").click()
            page.locator('[data-triple-variant="v3"]').click()
            page.locator(
                '[data-triple-variant="v3"][aria-pressed="true"]'
            ).wait_for()
            first_triple = page.locator("#triple-list [data-triple-id]").first
            first_triple.click()
            page.locator("#graph-toggle").click()
            page.wait_for_function(
                "() => document.querySelector('#graph-counts').textContent"
                ".includes('2 nodes · 1 relations · 1 claims')"
            )
            page.locator(".graph-page-link").first.wait_for()
            claim = {
                "counts": page.locator("#graph-counts").inner_text(),
                "sourceLink": page.locator(".graph-page-link").first.inner_text(),
                "burmese": page.locator(".graph-claim-my").first.inner_text(),
                "english": page.locator(".graph-claim-en").first.inner_text(),
            }
            if not claim["burmese"] or not claim["english"]:
                raise AssertionError(f"Claim provenance is incomplete: {claim}")
            if errors:
                raise AssertionError(" | ".join(errors))
            page.screenshot(path=str(ARTIFACT), full_page=True)
            print(
                json.dumps(
                    {
                        "initial": initial,
                        "corpus": corpus,
                        "semanticResult": semantic,
                        "claim": claim,
                    },
                    ensure_ascii=True,
                    indent=2,
                )
            )
        finally:
            browser.close()


if __name__ == "__main__":
    main()
