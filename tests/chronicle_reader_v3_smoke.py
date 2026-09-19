#!/usr/bin/env python3
"""Browser smoke test for the ungrounded V3 reader interaction."""

from __future__ import annotations

import json
import os
from pathlib import Path

from playwright.sync_api import sync_playwright


BASE_URL = os.environ.get("READER_BASE_URL", "http://127.0.0.1:5077")
ARTIFACT_DIR = (
    Path(__file__).resolve().parents[1] / "konbaung_reader_app" / "test-artifacts"
)


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        browser_errors: list[str] = []
        page.on("pageerror", lambda error: browser_errors.append(str(error)))
        page.on(
            "console",
            lambda message: (
                browser_errors.append(message.text)
                if message.type == "error"
                else None
            ),
        )
        try:
            page.goto(f"{BASE_URL}/chronicles/vol1/47", wait_until="networkidle")
            page.locator('[data-triple-variant="v3"]').click()
            page.locator(
                '[data-triple-variant="v3"][aria-pressed="true"]'
            ).wait_for()
            page.wait_for_function(
                "() => document.querySelectorAll("
                "'#triple-list [data-triple-id]').length === 16"
            )

            state = page.evaluate(
                """async () => {
                  const v2 = await fetch('/api/chronicles/page/vol1/47?triples=v2').then(r => r.json());
                  const v3 = await fetch('/api/chronicles/page/vol1/47?triples=v3').then(r => r.json());
                  const unavailable = await fetch('/api/chronicles/page/vol2/132?triples=v3').then(r => r.json());
                  const sentenceFields = payload => payload.sentences.map(sentence => ({
                    id: sentence.id,
                    text: sentence.text,
                    translation: sentence.translation,
                    fragments: sentence.fragments
                  }));
                  return {
                    canonicalTextEqual: v2.canonicalText === v3.canonicalText,
                    sentencesEqual: JSON.stringify(sentenceFields(v2)) === JSON.stringify(sentenceFields(v3)),
                    v3Annotations: v3.annotations.length,
                    v3HasAnySpan: v3.annotations.some(annotation =>
                      [annotation.subject, annotation.relation, annotation.object, annotation.evidence].some(endpoint =>
                        Number.isInteger(endpoint.startUtf16)
                        || Number.isInteger(endpoint.endUtf16)
                        || (endpoint.fragments || []).length)),
                    unavailableStatus: unavailable.diagnostics.v3Status,
                    unavailableAnnotations: unavailable.annotations.length,
                    unavailableDecisionCount: unavailable.sentences.filter(
                      sentence => sentence.tripleDecision === 'unavailable').length
                  };
                }"""
            )
            if not state["canonicalTextEqual"] or not state["sentencesEqual"]:
                raise AssertionError(f"V3 changed text or translations: {state}")
            if state["v3Annotations"] != 16 or state["v3HasAnySpan"]:
                raise AssertionError(f"V3 grounding is wrong: {state}")
            if (
                state["unavailableStatus"] != "partial"
                or state["unavailableAnnotations"] != 2
                or state["unavailableDecisionCount"] != 45
            ):
                raise AssertionError(f"Unresolved V3 page is not explicit: {state}")

            if page.locator("#annotation-toggle").is_disabled():
                raise AssertionError("Sentence highlighting toggle is disabled in V3")
            if page.locator("#canonical-text [data-memberships]").count():
                raise AssertionError("V3 rendered memberships into canonical text")
            for selector in (
                ".annotation-subject",
                ".annotation-predicate",
                ".annotation-object",
                ".annotation-evidence",
            ):
                if page.locator(f"#canonical-text {selector}").count():
                    raise AssertionError(f"V3 rendered highlight class {selector}")
            if not page.locator("#triple-list .entity-analytical").count():
                raise AssertionError("V3 analytical endpoints were not rendered")
            if "ungrounded" in page.locator("#triple-list").inner_text().lower():
                raise AssertionError("V3 cards still repeat the Ungrounded label")

            page.locator("#inspector-pane").evaluate(
                "element => { element.scrollTop = 0; }"
            )
            page.locator(
                '#canonical-text [data-sentence-id="vol1_s000003"]'
            ).first.click()
            page.wait_for_timeout(150)
            navigation = page.evaluate(
                """() => {
                  const pane = document.querySelector('#inspector-pane');
                  const group = document.querySelector(
                    '#triple-list [data-sentence-id="vol1_s000003"]');
                  const paneRect = pane.getBoundingClientRect();
                  const groupRect = group.getBoundingClientRect();
                  return {
                    scrollTop: pane.scrollTop,
                    active: group.classList.contains('active-sentence'),
                    visible: groupRect.bottom > paneRect.top && groupRect.top < paneRect.bottom,
                    relationPath: document.querySelector('#relation-path').getAttribute('d'),
                    textMemberships: document.querySelectorAll(
                      '#canonical-text [data-memberships]').length,
                    sentenceHighlights: document.querySelectorAll(
                      '#canonical-text .sentence-active').length
                  };
                }"""
            )
            if (
                not navigation["active"]
                or not navigation["visible"]
                or navigation["scrollTop"] <= 0
            ):
                raise AssertionError(
                    f"V3 sentence did not navigate to its group: {navigation}"
                )
            if (
                navigation["relationPath"]
                or navigation["textMemberships"]
                or navigation["sentenceHighlights"] <= 0
            ):
                raise AssertionError(
                    f"V3 sentence navigation highlight is wrong: {navigation}"
                )

            page.locator("#triple-list [data-triple-id]").last.click()
            page.wait_for_timeout(100)
            if page.locator("#canonical-text [data-memberships]").count():
                raise AssertionError("Clicking a V3 triple created memberships")
            if page.locator("#relation-path").get_attribute("d"):
                raise AssertionError("Clicking a V3 triple drew a connector")

            page.screenshot(
                path=str(
                    ARTIFACT_DIR
                    / "reader-v3-ungrounded-sentence-navigation.png"
                ),
                full_page=True,
            )

            page.locator('[data-triple-variant="v2"]').click()
            page.locator(
                '[data-triple-variant="v2"][aria-pressed="true"]'
            ).wait_for()
            page.wait_for_function(
                "() => document.querySelectorAll("
                "'#triple-list [data-triple-id]').length === 9"
            )
            if page.locator("#annotation-toggle").is_disabled():
                raise AssertionError("Grounded annotation toggle stayed disabled in V2")
            if not page.locator("#canonical-text [data-memberships]").count():
                raise AssertionError("V2 grounding disappeared after switching from V3")

            page.locator('[data-triple-variant="v3"]').click()
            page.locator(
                '[data-triple-variant="v3"][aria-pressed="true"]'
            ).wait_for()
            page.wait_for_function(
                "() => document.querySelectorAll("
                "'#triple-list [data-triple-id]').length === 16"
            )
            if page.locator("#canonical-text [data-memberships]").count():
                raise AssertionError("Switching back to V3 retained V2 memberships")
            if browser_errors:
                raise AssertionError(f"Browser errors: {' | '.join(browser_errors)}")
            print(
                json.dumps(
                    {"state": state, "sentenceNavigation": navigation},
                    indent=2,
                )
            )
        finally:
            browser.close()


if __name__ == "__main__":
    main()
