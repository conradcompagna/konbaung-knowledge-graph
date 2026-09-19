const { test, expect } = require('@playwright/test');

const baseUrl = process.env.GRAPH_TEST_URL || 'http://127.0.0.1:5077';

test('the knowledge graph contains the API entry point and compact documentation', async ({ page }) => {
  await page.goto(`${baseUrl}/chronicles/vol1/47`);
  await expect(page.locator('.toolbar a[href="/api/v1/docs"]')).toHaveCount(0);

  await page.locator('#graph-toggle').click();
  await expect(page).toHaveURL(`${baseUrl}/knowledge-graph/vol1/47`);
  await expect(page.locator('.graph-header-actions .graph-api-link')).toBeVisible();

  await page.goto(`${baseUrl}/api/v1/docs`);
  await expect(page.locator('.site-nav-brand')).toHaveCount(0);
  await expect(page.locator('.site-nav-links a')).toHaveCount(4);
  await expect(page.locator('.site-nav-links a[href="/reader"]')).toHaveText('Neural Reader');
  await expect(page.locator('.site-nav-links a[href="/knowledge-graph"]')).toHaveText('Knowledge Graph');
  await expect(page.locator('.site-nav-links a[href="/api/v1/docs"]')).toHaveAttribute('aria-current', 'page');
  await expect(page.locator('.fact a[href="/api/v1/openapi.json"]')).toHaveText('JSON · OpenAPI 3.1');
  await expect(page.getByRole('heading', { name: 'Knowledge Graph API' })).toBeVisible();
  await expect(page.locator('.endpoint')).toHaveCount(6);
  await expect(page.getByRole('heading', { name: 'Query rules' })).toBeVisible();
  await expect(page.locator('.badges')).toHaveCount(0);
  await expect(page.locator('body')).not.toContainText('Triple filters');
});

test('the dedicated knowledge graph route opens the canonical graph frontend', async ({ page }) => {
  await page.goto(`${baseUrl}/knowledge-graph/vol1/47`);
  await expect(page.locator('#graph-workspace')).toBeVisible();
  await expect(page.locator('[data-site-section="knowledge-graph"]')).toHaveAttribute('aria-current', 'page');

  await page.locator('#graph-close').click();
  await expect(page).toHaveURL(`${baseUrl}/chronicles/vol1/47`);
  await expect(page.locator('[data-site-section="chronicles"]')).toHaveAttribute('aria-current', 'page');
});
