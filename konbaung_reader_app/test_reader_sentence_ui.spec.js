const { test, expect } = require('@playwright/test');

test('sentence translations and triples stay page based', async ({ page }) => {
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));

  await page.goto('http://127.0.0.1:5077/chronicles/vol1/47');
  await expect(page.locator('#page-heading')).toHaveText('Volume I, source page 47');
  await expect(page.locator('.sentence-group')).toHaveCount(3);
  await expect(page.locator('.sentence-evidence .evidence-translation')).toHaveCount(3);
  await expect(page.locator('.triple-row')).toHaveCount(16);
  await expect(page.locator('#triple-variant-switch')).toHaveCount(0);
  await expect(page.locator('#annotation-toggle')).toHaveCount(0);
  await expect(page.locator('#translate-button')).toHaveCount(0);
  await expect(page.locator('#annotation-count')).toContainText('annotations');
  await expect(page.locator('#chronicle-app')).not.toContainText(/[ÂÃâ]/);
  await expect(
    page.locator('.sentence-group[data-sentence-id="vol1_s000003"]'),
  ).toHaveCount(1);

  const dictionaryTokenIndex = await page.evaluate(async () => {
    const response = await fetch('/api/chronicles/segment', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ volumeId: 'vol1', pageNumber: 47 }),
    });
    const segmentation = await response.json();
    return Object.entries(segmentation.dictionary || {}).find(([, entry]) => entry)?.[0] || null;
  });
  expect(dictionaryTokenIndex).not.toBeNull();
  const dictionaryToken = page.locator(
    `.token-fragment[data-token-index="${dictionaryTokenIndex}"]`,
  ).first();
  await expect(dictionaryToken).toBeVisible();
  await dictionaryToken.hover();
  await expect(page.locator('#hover-popup .hover-dictionary')).toBeVisible();
  await expect(page.locator('#hover-popup .hover-triple')).toHaveCount(0);
  await expect(page.locator('#hover-popup')).toHaveCSS('overflow-y', 'visible');

  await page.goto('http://127.0.0.1:5077/chronicles/vol1/48');
  await expect(page.locator('#page-heading')).toHaveText('Volume I, source page 48');
  await expect(
    page.locator('.sentence-group[data-sentence-id="vol1_s000003"]'),
  ).toHaveCount(1);

  await page.goto('http://127.0.0.1:5077/chronicles/vol1/49');
  const lastTriple = page.locator('.triple-row').last();
  await lastTriple.scrollIntoViewIfNeeded();
  await lastTriple.click();
  await expect(lastTriple).toHaveClass(/active/);
  expect(errors).toEqual([]);
});
