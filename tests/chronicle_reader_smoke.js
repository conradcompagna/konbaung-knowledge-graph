const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

const BASE_URL = process.env.READER_BASE_URL || 'http://127.0.0.1:5077';
const ARTIFACT_DIR = path.join(__dirname, '..', 'konbaung_reader_app', 'test-artifacts');

function sameBox(before, after, label) {
  for (const field of ['x', 'y', 'width', 'height']) {
    if (before[field] !== after[field]) {
      throw new Error(`Hover changed ${label}.${field}: ${before[field]} -> ${after[field]}`);
    }
  }
}

async function segmentPage(page, volumeId, pageNumber) {
  return page.evaluate(async ({ volumeId, pageNumber }) => {
    const response = await fetch('/api/chronicles/segment', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ volumeId, pageNumber })
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `Segmentation failed: ${response.status}`);
    return payload;
  }, { volumeId, pageNumber });
}

async function main() {
  fs.mkdirSync(ARTIFACT_DIR, { recursive: true });
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  let translatePosts = 0;
  page.on('request', (request) => {
    if (request.method() === 'POST' && request.url().endsWith('/api/chronicles/translate')) translatePosts += 1;
  });

  try {
    await page.goto(`${BASE_URL}/chronicles/vol1/47`, { waitUntil: 'networkidle' });
    await page.waitForFunction(() => document.querySelector('#canonical-text')?.textContent.length > 100);
    await page.locator('#canonical-text.text-fitted').waitFor();

    const heading = await page.locator('#page-heading').textContent();
    if (!heading.includes('source page 47')) throw new Error(`Unexpected heading: ${heading}`);
    if (await page.locator('#inspector-pane > .inspector-section').count() !== 2) {
      throw new Error('Inspector must contain only summary and triples');
    }
    const tripleOrder = await page.evaluate(async () => {
      const payload = await fetch('/api/chronicles/page/vol1/47').then((response) => response.json());
      const expected = payload.annotations
        .filter((annotation) => annotation.evidence.status === 'resolved')
        .sort((a, b) => a.evidence.startUtf16 - b.evidence.startUtf16
          || a.evidence.endUtf16 - b.evidence.endUtf16
          || a.id.localeCompare(b.id))
        .map((annotation) => annotation.id);
      const rendered = Array.from(document.querySelectorAll('#triple-list [data-triple-id]'))
        .map((row) => row.dataset.tripleId);
      return { expected, rendered };
    });
    if (JSON.stringify(tripleOrder.expected) !== JSON.stringify(tripleOrder.rendered)) {
      throw new Error('Triple list is not ordered by canonical evidence position');
    }
    const evidenceNavigationTripleId = await page.evaluate(async () => {
      const payload = await fetch('/api/chronicles/page/vol1/47').then((response) => response.json());
      return payload.annotations.find((annotation) =>
        annotation.evidence.status === 'resolved'
        && (annotation.subject.inferred || annotation.object.inferred)
      )?.id || null;
    });
    if (evidenceNavigationTripleId) {
      await page.locator(`[data-triple-id="${evidenceNavigationTripleId}"]`).click();
      await page.waitForTimeout(100);
      const evidencePosition = await page.evaluate((annotationId) => {
        const memberships = (element) => (element.dataset.memberships || '').split(',').filter(Boolean).map((entry) => {
          const split = entry.lastIndexOf(':');
          return { annotationId: entry.slice(0, split), role: entry.slice(split + 1) };
        });
        const evidence = Array.from(document.querySelectorAll('#canonical-text [data-memberships]')).find((item) =>
          memberships(item).some((membership) =>
            membership.annotationId === annotationId && membership.role === 'evidence'));
        const pane = document.querySelector('#text-scroll');
        if (!evidence || !pane) return null;
        const evidenceRect = evidence.getBoundingClientRect();
        const paneRect = pane.getBoundingClientRect();
        return { top: evidenceRect.top, bottom: evidenceRect.bottom, paneTop: paneRect.top, paneBottom: paneRect.bottom };
      }, evidenceNavigationTripleId);
      if (!evidencePosition
        || evidencePosition.bottom < evidencePosition.paneTop
        || evidencePosition.top > evidencePosition.paneBottom) {
        throw new Error('Clicking an inferred-endpoint triple did not reveal its evidence span');
      }
      await page.keyboard.press('Escape');
      await page.evaluate(() => { document.querySelector('#text-scroll').scrollTop = 0; });
    }
    for (const removedId of ['#relation-section', '#dictionary-card', '#translation-section', '#diagnostic-summary']) {
      if (await page.locator(removedId).count()) throw new Error(`Removed inspector UI returned: ${removedId}`);
    }

    const layout = await page.evaluate(() => {
      const text = document.querySelector('#text-scroll');
      const inspector = document.querySelector('#inspector-pane');
      const canonical = document.querySelector('#canonical-text');
      return {
        text: { client: text.clientHeight, scroll: text.scrollHeight, horizontal: text.scrollWidth - text.clientWidth },
        inspector: { client: inspector.clientHeight, scroll: inspector.scrollHeight },
        whiteSpace: getComputedStyle(canonical).whiteSpace,
        fontSize: getComputedStyle(canonical).fontSize
      };
    });
    if (layout.text.scroll <= layout.text.client) throw new Error('Chronicle pane is not vertically scrollable');
    if (layout.inspector.scroll <= layout.inspector.client) throw new Error('Summary/triple pane is not vertically scrollable');
    if (layout.text.horizontal > 1) throw new Error(`Chronicle pane has ${layout.text.horizontal}px lateral overflow`);
    if (layout.whiteSpace !== 'pre') throw new Error(`Source lines are not preserved: white-space=${layout.whiteSpace}`);

    const annotated = page.locator('.annotation-subject').first();
    const before = await page.evaluate(() => ({
      text: document.querySelector('#canonical-text').getBoundingClientRect().toJSON(),
      fragment: document.querySelector('.annotation-subject').getBoundingClientRect().toJSON()
    }));
    await annotated.hover();
    await page.locator('#hover-popup:not([hidden])').waitFor();
    const after = await page.evaluate(() => ({
      text: document.querySelector('#canonical-text').getBoundingClientRect().toJSON(),
      fragment: document.querySelector('.annotation-subject').getBoundingClientRect().toJSON()
    }));
    sameBox(before.text, after.text, 'text');
    sameBox(before.fragment, after.fragment, 'fragment');

    const collision = await page.evaluate(() => {
      const popup = document.querySelector('#hover-popup').getBoundingClientRect();
      const target = document.querySelector('.annotation-subject').getBoundingClientRect();
      const area = (a, b) => Math.max(0, Math.min(a.right, b.right) - Math.max(a.left, b.left))
        * Math.max(0, Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top));
      const annotations = Array.from(document.querySelectorAll('.annotation-active-subject, .annotation-active-object'))
        .flatMap((fragment) => Array.from(fragment.getClientRects()));
      const path = document.querySelector('#relation-path');
      const svgRect = document.querySelector('#relation-overlay').getBoundingClientRect();
      const relationHits = path && path.hasAttribute('d')
        ? Array.from({ length: 41 }, (_, index) => path.getPointAtLength(path.getTotalLength() * index / 40))
            .filter((point) => popup.left <= svgRect.left + point.x && svgRect.left + point.x <= popup.right
              && popup.top <= svgRect.top + point.y && svgRect.top + point.y <= popup.bottom).length
        : 0;
      return { target: area(popup, target), annotations: annotations.reduce((sum, rect) => sum + area(popup, rect), 0), relationHits };
    });
    if (collision.target !== 0 || collision.annotations !== 0 || collision.relationHits !== 0) {
      throw new Error(`Popup collision: ${JSON.stringify(collision)}`);
    }
    if (!await page.locator('.annotation-active-evidence').count()) throw new Error('Hovered triple did not mark its evidence block');
    if (!await page.locator('#hover-popup .entity-burmese').count()) throw new Error('Hover popup omitted Burmese entity spans');
    if (!await page.locator('#relation-path').getAttribute('d')) throw new Error('Hovered entities are not visibly linked');
    await page.screenshot({ path: path.join(ARTIFACT_DIR, 'reader-clean-hover.png') });

    await page.locator('#text-scroll').evaluate((element) => { element.scrollTop = element.scrollHeight; });
    const firstTriple = page.locator('[data-triple-id]').first();
    const firstTripleId = await firstTriple.getAttribute('data-triple-id');
    await firstTriple.click();
    await page.waitForTimeout(100);
    const snapped = await page.evaluate((annotationId) => {
      const pane = document.querySelector('#text-scroll').getBoundingClientRect();
      return Array.from(document.querySelectorAll('[data-memberships]')).some((fragment) => {
        if (!fragment.dataset.memberships.includes(annotationId)) return false;
        const rect = fragment.getBoundingClientRect();
        return rect.bottom > pane.top && rect.top < pane.bottom && rect.right > pane.left && rect.left < pane.right;
      });
    }, firstTripleId);
    if (!snapped) throw new Error('Clicking a sidebar triple did not snap its evidence into view');
    if (!await firstTriple.evaluate((element) => element.classList.contains('active'))) throw new Error('Clicked triple was not pinned');

    await page.locator('#next-page').click();
    await page.waitForURL('**/chronicles/vol1/48');
    await page.waitForFunction(() => document.querySelector('#translate-button')?.textContent === 'Glosses ready');
    if (!await page.locator('#translate-button').isDisabled()) throw new Error('Cached page gloss button must be disabled');
    if (translatePosts !== 0) throw new Error('Loading a permanently cached page issued a Gemini POST');

    const cached = await page.evaluate(async () => {
      const response = await fetch('/api/chronicles/glosses/vol1/48');
      const payload = await response.json();
      return { status: response.status, cached: payload.cached, tokenCount: payload.tokens.length };
    });
    if (cached.status !== 200 || !cached.cached || !cached.tokenCount) throw new Error(`Cache-only endpoint failed: ${JSON.stringify(cached)}`);

    const page48 = await segmentPage(page, 'vol1', 48);
    const konbaungIndex = page48.dictionary.findIndex((entry) => entry.head === 'ကုန်းဘောင်');
    if (konbaungIndex < 0) throw new Error('Could not find Konbaung dictionary token');
    await page.locator(`[data-token-index="${konbaungIndex}"]`).first().hover();
    const hoverGloss = await page.locator('#hover-popup .hover-gloss').textContent();
    const popupText = await page.locator('#hover-popup').textContent();
    if (hoverGloss.trim() !== 'Konbaung') throw new Error(`Unexpected cached gloss: ${hoverGloss}`);
    if (/dictionary|gemini|wiki|mmd|pali/i.test(popupText)) throw new Error(`Popup contains redundant labels/source names: ${popupText}`);
    if (popupText.split('ကုန်းဘောင်').length - 1 !== 1) throw new Error('Popup repeats the headword');
    await page.screenshot({ path: path.join(ARTIFACT_DIR, 'reader-clean-gloss-popup.png') });

    await page.locator('#next-page').click();
    await page.waitForURL('**/chronicles/vol1/49');
    await page.waitForFunction(() => document.querySelector('#page-heading')?.textContent.includes('source page 49'));
    await page.locator('#canonical-text.text-fitted').waitFor();
    const page49 = await segmentPage(page, 'vol1', 49);
    const crossWhitespace = page49.tokens.filter((token) => token.text.trim() && /\s/.test(token.text));
    if (crossWhitespace.length) throw new Error(`Tokens crossed whitespace: ${JSON.stringify(crossWhitespace.slice(0, 3))}`);
    const multiPosIndex = page49.dictionary.findIndex((entry) => entry.partsOfSpeech.length > 1 && entry.definitions.length);
    if (multiPosIndex < 0) throw new Error('No multiple-POS dictionary test token found');
    await page.locator(`[data-token-index="${multiPosIndex}"]`).first().hover();
    const shownPos = await page.locator('#hover-popup .hover-pos').textContent();
    if (!shownPos.includes(',')) throw new Error(`Multiple POS values were not displayed: ${shownPos}`);
    if (!await page.locator('#hover-popup .hover-senses li').count()) throw new Error('Definitions were not displayed');

    const inferredTripleId = 'vol1-p0049-a004';
    const inferredRowText = (await page.locator(`[data-triple-id="${inferredTripleId}"]`).innerText()).toLowerCase();
    for (const required of ['evidence', 'subject', 'object', 'inferred']) {
      if (!inferredRowText.includes(required)) throw new Error(`Sidebar triple omitted ${required}`);
    }
    await page.locator(`[data-memberships*="${inferredTripleId}:object"]`).hover();
    await page.locator('#hover-popup:not([hidden])').waitFor();
    if (await page.locator('#hover-popup .hover-triple').count() !== 1) throw new Error('Inline hover mixed multiple triples');
    if (await page.locator('.annotation-active-evidence').count() === 0) throw new Error('Evidence block was not marked');
    if (await page.locator('.triple-row.active').getAttribute('data-triple-id') !== inferredTripleId) {
      throw new Error('Hovered inferred endpoint activated the wrong triple');
    }
    if (!await page.locator('#hover-popup .inferred-label').count()) throw new Error('Inferred role was not labeled');

    await page.setViewportSize({ width: 390, height: 844 });
    await page.reload({ waitUntil: 'networkidle' });
    await page.waitForFunction(() => document.querySelector('#canonical-text')?.textContent.length > 100);
    await page.locator('#canonical-text.text-fitted').waitFor();
    await page.locator('#close-summary').click();
    const mobileOverflow = await page.evaluate(() => ({
      page: document.documentElement.scrollWidth - window.innerWidth,
      reader: document.querySelector('#text-scroll').scrollWidth - document.querySelector('#text-scroll').clientWidth
    }));
    if (mobileOverflow.page > 1 || mobileOverflow.reader > 1) throw new Error(`Mobile lateral overflow: ${JSON.stringify(mobileOverflow)}`);
    await page.screenshot({ path: path.join(ARTIFACT_DIR, 'reader-clean-mobile.png') });

    process.stdout.write(JSON.stringify({ heading, layout, collision, cached, hoverGloss, shownPos, page49Tokens: page49.tokens.length }, null, 2));
  } finally {
    await browser.close();
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
