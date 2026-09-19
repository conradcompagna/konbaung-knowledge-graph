const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

const baseUrl = (process.env.GRAPH_TEST_URL || 'https://burmeseneuralreader.com').replace(/\/$/, '');
const readerBaseUrl = (process.env.READER_TEST_URL || baseUrl).replace(/\/$/, '');
const outputRoot = path.join(__dirname, '..', 'artifacts', 'live_review');
const expectedNavigation = ['Neural Reader', 'Chronicle Reader', 'Knowledge Graph', 'API'];

async function verifyPage(page, route, checks, screenshotName, routeBaseUrl = baseUrl) {
  const failures = [];
  const onResponse = (response) => {
    if (response.url().startsWith(routeBaseUrl) && response.status() >= 400) {
      failures.push(`${response.status()} ${response.url()}`);
    }
  };
  page.on('response', onResponse);
  const response = await page.goto(`${routeBaseUrl}${route}`, { waitUntil: 'domcontentloaded' });
  if (!response || !response.ok()) throw new Error(`${route} returned ${response && response.status()}`);
  for (const check of checks) await page.locator(check).waitFor({ state: 'visible' });
  if (route.startsWith('/knowledge-graph')) {
    await page.locator('#graph-canvas canvas').first().waitFor({ state: 'visible' });
    await page.waitForFunction(() => document.querySelector('#graph-counts')?.textContent.trim());
    await page.locator('#graph-loading').waitFor({ state: 'hidden' });
    await page.waitForTimeout(500);
  }
  await page.screenshot({ path: path.join(outputRoot, screenshotName), fullPage: false });
  const navigation = await page.evaluate(() => {
    const nav = document.querySelector('.site-nav');
    const links = [...document.querySelectorAll('.site-nav-links a')];
    const active = links.find((link) => link.getAttribute('aria-current') === 'page');
    const navStyle = getComputedStyle(nav);
    const linkStyle = getComputedStyle(links[0]);
    return {
      labels: links.map((link) => link.textContent.trim()),
      brandCount: document.querySelectorAll('.site-nav-brand, .site-nav > .brand').length,
      height: Math.round(nav.getBoundingClientRect().height),
      background: navStyle.backgroundColor,
      border: navStyle.borderBottomColor,
      gap: getComputedStyle(document.querySelector('.site-nav-links')).gap,
      linkFont: linkStyle.fontFamily,
      linkFontSize: linkStyle.fontSize,
      linkPadding: linkStyle.padding,
      linkRadius: linkStyle.borderRadius,
      activeBackground: getComputedStyle(active).backgroundColor,
    };
  });
  page.off('response', onResponse);
  if (failures.length) throw new Error(`${route} had failed subrequests: ${failures.join(', ')}`);
  if (JSON.stringify(navigation.labels) !== JSON.stringify(expectedNavigation)) {
    throw new Error(`${route} has a different top-level link set: ${navigation.labels.join(', ')}`);
  }
  if (navigation.brandCount !== 0) throw new Error(`${route} still renders a corner brand`);
  return { url: page.url(), navigation };
}

async function main() {
  fs.mkdirSync(outputRoot, { recursive: true });
  const browser = await chromium.launch({ channel: 'chrome' });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const pageErrors = [];
  page.on('pageerror', (error) => pageErrors.push(error.message));

  const results = [];
  results.push(await verifyPage(page, '/reader', [
    '.site-nav a[href="/reader"][aria-current="page"]',
    '#reader-app',
  ], 'reader.png', readerBaseUrl));
  results.push(await verifyPage(page, '/chronicles', [
    '[data-site-section="chronicles"][aria-current="page"]',
    '#canonical-text',
  ], 'chronicles.png'));
  results.push(await verifyPage(page, '/knowledge-graph', [
    '[data-site-section="knowledge-graph"][aria-current="page"]',
    '#graph-workspace',
  ], 'knowledge-graph.png'));
  results.push(await verifyPage(page, '/api/v1/docs', [
    '.site-nav-links a[href="/api/v1/docs"][aria-current="page"]',
    'h1',
  ], 'api-docs.png'));

  const mobilePage = await browser.newPage({ viewport: { width: 390, height: 844 } });
  mobilePage.on('pageerror', (error) => pageErrors.push(error.message));
  const mobileResults = [];
  mobileResults.push(await verifyPage(mobilePage, '/reader', [
    '.site-nav a[href="/reader"][aria-current="page"]',
    '#reader-app',
  ], 'mobile-reader.png', readerBaseUrl));
  mobileResults.push(await verifyPage(mobilePage, '/chronicles', [
    '[data-site-section="chronicles"][aria-current="page"]',
    '#canonical-text',
  ], 'mobile-chronicles.png'));
  mobileResults.push(await verifyPage(mobilePage, '/knowledge-graph', [
    '[data-site-section="knowledge-graph"][aria-current="page"]',
    '#graph-workspace',
  ], 'mobile-knowledge-graph.png'));
  mobileResults.push(await verifyPage(mobilePage, '/api/v1/docs', [
    '.site-nav-links a[href="/api/v1/docs"][aria-current="page"]',
    'h1',
  ], 'mobile-api-docs.png'));

  const referenceNavigation = JSON.stringify(results[0].navigation);
  for (const result of results.slice(1)) {
    if (JSON.stringify(result.navigation) !== referenceNavigation) {
      throw new Error(`Navigation styling differs on ${result.url}`);
    }
  }
  const mobileReferenceNavigation = JSON.stringify(mobileResults[0].navigation);
  for (const result of mobileResults.slice(1)) {
    if (JSON.stringify(result.navigation) !== mobileReferenceNavigation) {
      throw new Error(`Mobile navigation styling differs on ${result.url}: ${JSON.stringify(result.navigation)} versus ${mobileReferenceNavigation}`);
    }
  }

  await browser.close();
  if (pageErrors.length) throw new Error(`Browser page errors: ${pageErrors.join(', ')}`);
  console.log([...results, ...mobileResults].map((result) => result.url));
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
