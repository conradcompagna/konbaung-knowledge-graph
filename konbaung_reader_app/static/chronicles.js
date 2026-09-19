(function () {
  'use strict';

  const initial = window.__CHRONICLE_INITIAL__ || { volumeId: 'vol1', pageNumber: 47, graphOpen: false };
  const state = {
    index: null,
    volumeId: initial.volumeId,
    pageNumber: Number(initial.pageNumber),
    page: null,
    segmentation: null,
    loadSequence: 0,
    hoveredAnnotationId: null,
    pinnedAnnotationId: null,
    pinnedSentenceId: null,
    prefetchedPages: new Map(),
    graphOpen: Boolean(initial.graphOpen)
  };

  const el = {
    workspace: document.getElementById('workspace'),
    inspector: document.getElementById('inspector-pane'),
    message: document.getElementById('app-message'),
    volume: document.getElementById('volume-select'),
    previous: document.getElementById('previous-page'),
    next: document.getElementById('next-page'),
    pageInput: document.getElementById('page-input'),
    pagePosition: document.getElementById('page-position'),
    graphToggle: document.getElementById('graph-toggle'),
    heading: document.getElementById('page-heading'),
    annotationCount: document.getElementById('annotation-count'),
    textScroll: document.getElementById('text-scroll'),
    text: document.getElementById('canonical-text'),
    overlay: document.getElementById('relation-overlay'),
    relationPath: document.getElementById('relation-path'),
    summaryText: document.getElementById('summary-text'),
    tripleList: document.getElementById('triple-list'),
    hoverPopup: document.getElementById('hover-popup')
  };

  function escapeHtml(value) {
    return String(value == null ? '' : value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  function setMessage(message) { el.message.textContent = message || ''; }

  function currentVolume() {
    return state.index.volumes.find((item) => item.id === state.volumeId);
  }

  // The visible reader/graph state determines the stable page URL shown to
  // the user, so browser history and the top-level navigation stay in unison.
  function currentPagePath() {
    const section = state.graphOpen ? 'knowledge-graph' : 'chronicles';
    return `/${section}/${state.volumeId}/${state.pageNumber}`;
  }

  // The active top-level link reflects the same graph visibility flag used
  // for history URLs, rather than retaining the state from the first render.
  function syncSiteNavigation() {
    const activeSection = state.graphOpen ? 'knowledge-graph' : 'chronicles';
    document.querySelectorAll('[data-site-section]').forEach((link) => {
      const active = link.dataset.siteSection === activeSection;
      link.classList.toggle('active', active);
      if (active) link.setAttribute('aria-current', 'page');
      else link.removeAttribute('aria-current');
    });
  }

  // The page endpoint defaults to the canonical V3 annotation corpus.
  function canonicalPageUrl(volumeId, pageNumber) {
    return `/api/chronicles/page/${volumeId}/${pageNumber}`;
  }

  function pagePosition() {
    const volume = currentVolume();
    return volume ? volume.availablePages.indexOf(state.pageNumber) : -1;
  }

  function annotationById(id) {
    return state.page && state.page.annotations.find((item) => item.id === id);
  }

  function activeAnnotationId() {
    return state.pinnedAnnotationId || state.hoveredAnnotationId;
  }

  function updateNavigation() {
    const volume = currentVolume();
    const position = pagePosition();
    if (!volume) return;
    el.volume.value = state.volumeId;
    el.pageInput.value = String(state.pageNumber);
    el.pageInput.min = String(volume.firstPage);
    el.pageInput.max = String(volume.lastPage);
    el.pagePosition.textContent = `${position + 1} / ${volume.pageCount}`;
    el.previous.disabled = position <= 0;
    el.next.disabled = position < 0 || position >= volume.availablePages.length - 1;
  }

  async function fetchJson(url, options) {
    const response = await fetch(url, options);
    const payload = await response.json().catch(() => ({}));
    if (!response.ok || payload.ok === false) throw new Error(payload.error || `Request failed: ${response.status}`);
    return payload;
  }

  function clearInteraction() {
    state.hoveredAnnotationId = null;
    state.pinnedAnnotationId = null;
    state.pinnedSentenceId = null;
    el.relationPath.removeAttribute('d');
    el.hoverPopup.hidden = true;
  }

  function populateVolumeSelect() {
    el.volume.innerHTML = state.index.volumes.map((volume) =>
      `<option value="${escapeHtml(volume.id)}">${escapeHtml(volume.label)}</option>`
    ).join('');
  }

  async function initialize() {
    syncSiteNavigation();
    // A direct graph URL must reveal the graph immediately; reader segmentation
    // can take several seconds on a cold production worker and runs underneath it.
    if (state.graphOpen) el.graphToggle.click();
    try {
      state.index = await fetchJson('/api/chronicles/index');
      populateVolumeSelect();
      const validVolume = state.index.volumes.some((item) => item.id === state.volumeId);
      if (!validVolume) state.volumeId = state.index.volumes[0].id;
      const volume = currentVolume();
      if (!volume.availablePages.includes(state.pageNumber)) state.pageNumber = volume.availablePages[0];
      await loadPage(false);
    } catch (error) {
      setMessage(error.message);
    }
  }

  async function loadPage(pushHistory) {
    const sequence = ++state.loadSequence;
    clearInteraction();
    setMessage('');
    el.text.textContent = '';
    el.text.classList.remove('text-fitted');
    updateNavigation();
    if (pushHistory) {
      history.pushState(
        { volumeId: state.volumeId, pageNumber: state.pageNumber, graphOpen: state.graphOpen },
        '',
        currentPagePath()
      );
    }

    try {
      const pagePromise = fetchJson(canonicalPageUrl(state.volumeId, state.pageNumber));
      const segmentPromise = fetchJson('/api/chronicles/segment', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ volumeId: state.volumeId, pageNumber: state.pageNumber })
      });
      const [page, segmentation] = await Promise.all([pagePromise, segmentPromise]);
      if (sequence !== state.loadSequence) return;
      state.page = page;
      state.segmentation = segmentation;
      renderPage();
      prefetchAdjacentPages();
    } catch (error) {
      if (sequence !== state.loadSequence) return;
      setMessage(error.message);
    }
  }

  function endpointBoundaries(annotation, boundaries) {
    for (const endpoint of [annotation.subject, annotation.relation, annotation.object, annotation.evidence]) {
      const ranges = Array.isArray(endpoint.fragments) && endpoint.fragments.length
        ? endpoint.fragments
        : [endpoint];
      for (const range of ranges) {
        if (Number.isInteger(range.startUtf16) && Number.isInteger(range.endUtf16)) {
          boundaries.add(range.startUtf16);
          boundaries.add(range.endUtf16);
        }
      }
    }
  }

  function sentenceRanges(sentence) {
    return Array.isArray(sentence.fragments) && sentence.fragments.length
      ? sentence.fragments
      : [sentence];
  }

  function sentenceIdForRange(start, end) {
    for (const sentence of state.page.sentences || []) {
      if (sentenceRanges(sentence).some((range) =>
        Number.isInteger(range.startUtf16) && Number.isInteger(range.endUtf16)
        && start >= range.startUtf16 && end <= range.endUtf16 && start < range.endUtf16)) {
        return sentence.id;
      }
    }
    return null;
  }

  function membershipForRange(start, end) {
    const memberships = [];
    for (const annotation of state.page.annotations) {
      for (const role of ['subject', 'predicate', 'object', 'evidence']) {
        const endpoint = role === 'predicate' ? annotation.relation : annotation[role];
        const ranges = Array.isArray(endpoint.fragments) && endpoint.fragments.length
          ? endpoint.fragments
          : [endpoint];
        if (ranges.some((range) => Number.isInteger(range.startUtf16) && Number.isInteger(range.endUtf16)
            && start >= range.startUtf16 && end <= range.endUtf16 && start < range.endUtf16)) {
          memberships.push({ annotationId: annotation.id, role });
        }
      }
    }
    return memberships;
  }

  function tokenForRange(start, end) {
    return state.segmentation.tokens.find((token) => start >= token.startUtf16 && end <= token.endUtf16) || null;
  }

  function renderPage() {
    const page = state.page;
    const boundaries = new Set([0, page.diagnostics.canonicalUtf16Length]);
    for (const token of state.segmentation.tokens) {
      boundaries.add(token.startUtf16);
      boundaries.add(token.endUtf16);
    }
    for (const sentence of page.sentences || []) {
      for (const range of sentenceRanges(sentence)) {
        if (Number.isInteger(range.startUtf16) && Number.isInteger(range.endUtf16)) {
          boundaries.add(range.startUtf16);
          boundaries.add(range.endUtf16);
        }
      }
    }
    for (const annotation of page.annotations) endpointBoundaries(annotation, boundaries);
    const ordered = Array.from(boundaries).filter((value) => Number.isInteger(value)).sort((a, b) => a - b);
    const fragment = document.createDocumentFragment();
    let reconstructed = '';

    for (let i = 0; i < ordered.length - 1; i += 1) {
      const start = ordered[i];
      const end = ordered[i + 1];
      if (end <= start) continue;
      const text = page.canonicalText.slice(start, end);
      reconstructed += text;
      const span = document.createElement('span');
      span.className = 'text-fragment';
      span.textContent = text;
      span.dataset.start = String(start);
      span.dataset.end = String(end);
      const sentenceId = sentenceIdForRange(start, end);
      if (sentenceId) {
        span.dataset.sentenceId = sentenceId;
        span.classList.add('sentence-navigation-fragment');
      }
      const token = tokenForRange(start, end);
      if (token && !/^\s+$/.test(token.text)) {
        span.classList.add('token-fragment');
        span.dataset.tokenIndex = String(token.index);
      }
      const memberships = membershipForRange(start, end);
      if (memberships.length) {
        span.dataset.memberships = memberships.map((item) => `${item.annotationId}:${item.role}`).join(',');
        if (memberships.some((item) => item.role === 'subject')) span.classList.add('annotation-subject');
        if (memberships.some((item) => item.role === 'predicate')) span.classList.add('annotation-predicate');
        if (memberships.some((item) => item.role === 'object')) span.classList.add('annotation-object');
        if (memberships.some((item) => item.role === 'evidence')) span.classList.add('annotation-evidence');
        if (memberships.some((item) => {
          const annotation = annotationById(item.annotationId);
          return (item.role === 'subject' || item.role === 'object') && annotation && annotation[item.role].inferred;
        })) span.classList.add('annotation-inferred');
      }
      fragment.appendChild(span);
    }

    if (reconstructed !== page.canonicalText) {
      throw new Error('Rendered fragments do not reconstruct the canonical page string');
    }
    el.text.innerHTML = '';
    el.text.appendChild(fragment);
    const volume = currentVolume();
    el.heading.textContent = `${volume.label}, source page ${page.pageNumber}`;
    el.annotationCount.textContent = `${page.diagnostics.annotationCount} annotations · ${page.diagnostics.translatedSentenceCount} translated sentences`;
    el.summaryText.textContent = page.summary || 'No summary is available for this page.';
    renderTripleList();
    updateNavigation();
    requestAnimationFrame(() => {
      fitCanonicalText();
      syncRelationOverlaySize();
      drawRelationConnector();
      if (!document.fonts || !document.fonts.ready) el.text.classList.add('text-fitted');
    });
    if (document.fonts && document.fonts.ready) {
      document.fonts.ready.then(() => requestAnimationFrame(() => {
        fitCanonicalText();
        syncRelationOverlaySize();
        drawRelationConnector();
        el.text.classList.add('text-fitted');
      }));
    }
  }

  function fitCanonicalText() {
    el.text.style.fontSize = '';
    const styles = getComputedStyle(el.textScroll);
    const available = el.textScroll.clientWidth
      - parseFloat(styles.paddingLeft)
      - parseFloat(styles.paddingRight);
    const baseSize = parseFloat(getComputedStyle(el.text).fontSize);
    const naturalWidth = el.text.scrollWidth;
    if (!available || !naturalWidth || naturalWidth <= available) return;
    const fittedSize = Math.max(6, Math.floor(baseSize * available / naturalWidth * 10) / 10);
    el.text.style.fontSize = `${fittedSize}px`;
  }

  function membershipsFromElement(target) {
    const raw = target && target.dataset ? target.dataset.memberships : '';
    if (!raw) return [];
    return raw.split(',').map((entry) => {
      const split = entry.lastIndexOf(':');
      return { annotationId: entry.slice(0, split), role: entry.slice(split + 1) };
    });
  }

  function metadataHtml(annotation) {
    const labels = { date: 'Date', location: 'Place', quantity: 'Quantity' };
    const rows = Object.entries(annotation.metadata || {}).filter(([, value]) => value && (value.originalText || value.gloss));
    if (!rows.length) return '';
    return `<dl class="triple-metadata">${rows.map(([key, value]) => `
      <div><dt>${labels[key]}</dt><dd>${value.originalText ? `<span lang="my">${escapeHtml(value.originalText)}</span>` : ''}${value.gloss ? `<span>${escapeHtml(value.gloss)}</span>` : ''}</dd></div>`).join('')}</dl>`;
  }

  function endpointStatusHtml(endpoint) {
    const labels = {
      absent: 'Not on this page',
      ambiguous: 'Span ambiguous',
      not_in_sentence: 'No exact span',
      unattached: 'No exact span',
      partial: 'Continues across page'
    };
    const label = endpoint.inferred ? 'Inferred' : labels[endpoint.pagePresence];
    return label ? `<span class="inferred-label">${escapeHtml(label)}</span>` : '';
  }

  function entityHtml(endpoint, role) {
    const status = endpointStatusHtml(endpoint);
    const language = endpoint.language === 'en' ? 'en' : 'my';
    const valueClass = language === 'en' ? 'entity-analytical' : 'entity-burmese';
    return `<div class="triple-entity">
      <span class="entity-role">${role}${status}</span>
      <span class="${valueClass}" lang="${language}">${escapeHtml(endpoint.text)}</span>
      ${endpoint.gloss ? `<span class="entity-gloss">${escapeHtml(endpoint.gloss)}</span>` : ''}
      ${endpoint.type ? `<span class="entity-tag">${escapeHtml(endpoint.type)}</span>` : ''}
    </div>`;
  }

  function relationHtml(relation) {
    const status = endpointStatusHtml(relation);
    const grounding = relation.predicateText || relation.predicateGloss
      ? `<span class="predicate-grounding">
          ${relation.predicateText ? `<span class="predicate-burmese" lang="my">${escapeHtml(relation.predicateText)}</span>` : ''}
          ${relation.predicateGloss ? `<span class="predicate-gloss">${escapeHtml(relation.predicateGloss)}</span>` : ''}
        </span>`
      : '';
    return `<span class="triple-relation">
      <span class="relation-label">${escapeHtml(relation.rawLabel)}</span>
      ${status}
      ${grounding}
    </span>`;
  }

  function sentenceEvidenceHtml(sentence) {
    const translation = sentence.translation
      ? `<span class="evidence-translation">${escapeHtml(sentence.translation)}</span>`
      : '';
    return `
      <span class="evidence-text" lang="my">${escapeHtml(sentence.canonicalText)}</span>
      ${translation}`;
  }

  function sentenceDecisionHtml(sentence) {
    if (sentence.tripleDecision === 'skip') {
      return `<div class="sentence-decision sentence-skip">Skipped · ${escapeHtml(sentence.tripleJustification)}</div>`;
    }
    if (sentence.tripleDecision === 'unavailable') {
      return `<div class="sentence-decision sentence-unavailable">${escapeHtml(sentence.tripleJustification)}</div>`;
    }
    return '';
  }

  function axialCategoryHtml(annotation) {
    const axial = annotation.axial;
    if (!axial) return '';
    if (axial.status !== 'accepted') {
      return `<div class="axial-coding axial-unresolved">
        <span class="axial-heading">Axial categories</span>
        <span>${escapeHtml(`${axial.status}: ${axial.reason || 'No valid assignment is available.'}`)}</span>
      </div>`;
    }
    const category = (role, item, className) => `
      <span class="axial-category ${className}" title="${escapeHtml(item.definition || '')}">
        <span class="axial-role">${role}</span>
        <strong>${escapeHtml(item.tagId)} · ${escapeHtml(item.label)}</strong>
        ${item.provisional ? '<em>Provisional</em>' : ''}
      </span>`;
    return `<div class="axial-coding">
      <span class="axial-heading">Axial categories</span>
      ${category('S', axial.subject, 'axial-entity')}
      ${category('R', axial.relation, 'axial-relation')}
      ${category('O', axial.object, 'axial-entity')}
    </div>`;
  }

  function dictionaryPopupHtml(tokenIndex) {
    const entry = state.segmentation && state.segmentation.dictionary[tokenIndex];
    if (!entry) return '';
    const definitions = (entry.definitions || []).map((definition) => `<li>${escapeHtml(definition)}</li>`).join('');
    const romanizations = (entry.romanizations || []).join(' / ');
    const partsOfSpeech = (entry.partsOfSpeech || []).join(', ');
    return `
      <div class="hover-section hover-dictionary">
        <div class="hover-token-line">
          <div class="hover-token">${escapeHtml(entry.head)}</div>
        </div>
        ${romanizations ? `<div class="hover-transliterations">${escapeHtml(romanizations)}</div>` : ''}
        ${partsOfSpeech ? `<div class="hover-pos">${escapeHtml(partsOfSpeech)}</div>` : ''}
        ${definitions ? `<ol class="hover-senses">${definitions}</ol>` : ''}
      </div>`;
  }

  function positionHoverPopup(target) {
    const targetRect = target.getBoundingClientRect();
    const popupRect = el.hoverPopup.getBoundingClientRect();
    const margin = 12;
    const gap = 10;
    const clamp = (left, top) => ({
      left: Math.max(margin, Math.min(left, window.innerWidth - popupRect.width - margin)),
      top: Math.max(margin, Math.min(top, window.innerHeight - popupRect.height - margin))
    });
    const lefts = [
      targetRect.right + gap,
      targetRect.left - popupRect.width - gap,
      targetRect.left,
      targetRect.right - popupRect.width,
      (targetRect.left + targetRect.right - popupRect.width) / 2
    ];
    const tops = [
      targetRect.top,
      targetRect.bottom + gap,
      targetRect.top - popupRect.height - gap,
      targetRect.bottom - popupRect.height
    ];
    const candidates = [];
    const seenCandidates = new Set();
    for (const left of lefts) {
      for (const top of tops) {
        const candidate = clamp(left, top);
        const key = `${Math.round(candidate.left)}:${Math.round(candidate.top)}`;
        if (!seenCandidates.has(key)) {
          seenCandidates.add(key);
          candidates.push(candidate);
        }
      }
    }
    const maxLeft = Math.max(margin, window.innerWidth - popupRect.width - margin);
    const maxTop = Math.max(margin, window.innerHeight - popupRect.height - margin);
    const gridStep = 24;
    for (let left = margin; left <= maxLeft; left += gridStep) {
      for (let top = margin; top <= maxTop; top += gridStep) {
        const candidate = clamp(left, top);
        const key = `${Math.round(candidate.left)}:${Math.round(candidate.top)}`;
        if (!seenCandidates.has(key)) {
          seenCandidates.add(key);
          candidates.push(candidate);
        }
      }
    }
    for (const candidate of [clamp(maxLeft, margin), clamp(margin, maxTop), clamp(maxLeft, maxTop)]) {
      const key = `${Math.round(candidate.left)}:${Math.round(candidate.top)}`;
      if (!seenCandidates.has(key)) {
        seenCandidates.add(key);
        candidates.push(candidate);
      }
    }
    const activeId = activeAnnotationId();
    const annotationRects = activeId
      ? Array.from(el.text.querySelectorAll('.annotation-subject, .annotation-predicate, .annotation-object'))
          .filter((fragment) => membershipsFromElement(fragment).some((item) =>
            item.annotationId === activeId && (item.role === 'subject' || item.role === 'predicate' || item.role === 'object')))
          .flatMap((fragment) => Array.from(fragment.getClientRects()))
      : [];
    const overlapArea = (candidate, rect) => {
      const width = Math.max(0, Math.min(candidate.left + popupRect.width, rect.right) - Math.max(candidate.left, rect.left));
      const height = Math.max(0, Math.min(candidate.top + popupRect.height, rect.bottom) - Math.max(candidate.top, rect.top));
      return width * height;
    };
    const targetCenter = { x: (targetRect.left + targetRect.right) / 2, y: (targetRect.top + targetRect.bottom) / 2 };
    const relationPoints = [];
    if (el.relationPath.hasAttribute('d')) {
      const length = el.relationPath.getTotalLength();
      const svgRect = el.overlay.getBoundingClientRect();
      for (let index = 0; index <= 40; index += 1) {
        const point = el.relationPath.getPointAtLength(length * index / 40);
        relationPoints.push({ x: svgRect.left + point.x, y: svgRect.top + point.y });
      }
    }
    const scored = candidates.map((candidate) => {
      const popupCenter = { x: candidate.left + popupRect.width / 2, y: candidate.top + popupRect.height / 2 };
      const targetOverlap = overlapArea(candidate, targetRect);
      const annotationOverlap = annotationRects.reduce((total, rect) => total + overlapArea(candidate, rect), 0);
      const relationHits = relationPoints.filter((point) =>
        candidate.left <= point.x && point.x <= candidate.left + popupRect.width
        && candidate.top <= point.y && point.y <= candidate.top + popupRect.height
      ).length;
      const distance = Math.hypot(popupCenter.x - targetCenter.x, popupCenter.y - targetCenter.y);
      return {
        candidate,
        score: targetOverlap * 1000000000 + annotationOverlap * 1000000 + relationHits * 1000000 + distance
      };
    });
    scored.sort((a, b) => a.score - b.score);
    el.hoverPopup.style.left = `${Math.round(scored[0].candidate.left)}px`;
    el.hoverPopup.style.top = `${Math.round(scored[0].candidate.top)}px`;
  }

  // Popups are reserved for dictionary definitions; claim annotations stay in the inspector.
  function renderHoverPopup(target, tokenIndex) {
    const dictionaryHtml = dictionaryPopupHtml(tokenIndex);
    if (!dictionaryHtml) {
      el.hoverPopup.hidden = true;
      return;
    }
    el.hoverPopup.innerHTML = dictionaryHtml;
    el.hoverPopup.style.visibility = 'hidden';
    el.hoverPopup.hidden = false;
    positionHoverPopup(target);
    el.hoverPopup.style.visibility = '';
  }

  function renderTripleList() {
    const earliestEndpointStart = (annotation) => {
      const starts = [annotation.subject, annotation.object]
        .map((endpoint) => endpoint.startUtf16)
        .filter((value) => Number.isInteger(value));
      return starts.length ? Math.min(...starts) : Number.POSITIVE_INFINITY;
    };
    const annotationMap = new Map(state.page.annotations.map((annotation) => [annotation.id, annotation]));
    const sentences = [...(state.page.sentences || [])].sort((a, b) =>
      a.startUtf16 - b.startUtf16 || a.endUtf16 - b.endUtf16 || a.id.localeCompare(b.id)
    );
    el.tripleList.innerHTML = sentences.map((sentence) => {
      const annotations = sentence.tripleIds
        .map((id) => annotationMap.get(id))
        .filter((annotation) => annotation && ['resolved', 'unattached'].includes(annotation.evidence.status))
        .sort((a, b) =>
          earliestEndpointStart(a) - earliestEndpointStart(b)
          || a.id.localeCompare(b.id)
        );
      const rows = annotations.map((annotation) => `
        <button class="triple-row" type="button" data-triple-id="${escapeHtml(annotation.id)}">
          ${entityHtml(annotation.subject, 'Subject')}
          ${relationHtml(annotation.relation)}
          ${entityHtml(annotation.object, 'Object')}
          ${axialCategoryHtml(annotation)}
          ${metadataHtml(annotation)}
        </button>`).join('');
      return `
        <section class="sentence-group" data-sentence-id="${escapeHtml(sentence.id)}">
          <div class="sentence-evidence">${sentenceEvidenceHtml(sentence)}</div>
          ${sentenceDecisionHtml(sentence)}
          ${rows ? `<div class="sentence-triples">${rows}</div>` : ''}
        </section>`;
    }).join('');
  }

  function updateActiveFragments() {
    const activeId = activeAnnotationId();
    const all = el.text.querySelectorAll('[data-memberships]');
    all.forEach((fragment) => {
      const memberships = membershipsFromElement(fragment);
      const activeRoles = new Set(
        activeId
          ? memberships.filter((item) => item.annotationId === activeId).map((item) => item.role)
          : []
      );
      fragment.classList.toggle('annotation-active-evidence', activeRoles.has('evidence'));
      fragment.classList.toggle('annotation-active-subject', activeRoles.has('subject'));
      fragment.classList.toggle('annotation-active-predicate', activeRoles.has('predicate'));
      fragment.classList.toggle('annotation-active-object', activeRoles.has('object'));
      fragment.classList.toggle(
        'annotation-active-inferred',
        Boolean(activeId) && ['subject', 'object'].some((role) =>
          activeRoles.has(role) && annotationById(activeId)[role].inferred)
      );
    });
    el.text.querySelectorAll('[data-sentence-id]').forEach((fragment) => {
      fragment.classList.toggle(
        'sentence-active',
        fragment.dataset.sentenceId === state.pinnedSentenceId
      );
    });
    el.tripleList.querySelectorAll('[data-triple-id]').forEach((row) => {
      row.classList.toggle('active', row.dataset.tripleId === activeId);
    });
    el.tripleList.querySelectorAll('[data-sentence-id]').forEach((group) => {
      group.classList.toggle('active-sentence', group.dataset.sentenceId === state.pinnedSentenceId);
    });
    drawRelationConnector();
  }

  function rectsForAnnotation(annotationId, role) {
    const scrollRect = el.textScroll.getBoundingClientRect();
    const rects = [];
    el.text.querySelectorAll('[data-memberships]').forEach((fragment) => {
      if (!membershipsFromElement(fragment).some((item) => item.annotationId === annotationId && item.role === role)) return;
      for (const rect of fragment.getClientRects()) {
        rects.push({
          left: rect.left - scrollRect.left + el.textScroll.scrollLeft,
          right: rect.right - scrollRect.left + el.textScroll.scrollLeft,
          top: rect.top - scrollRect.top + el.textScroll.scrollTop,
          bottom: rect.bottom - scrollRect.top + el.textScroll.scrollTop
        });
      }
    });
    return rects;
  }

  function center(rect) { return { x: (rect.left + rect.right) / 2, y: (rect.top + rect.bottom) / 2 }; }

  function nearestRectPair(subjectRects, objectRects) {
    let best = null;
    for (const subject of subjectRects) {
      for (const object of objectRects) {
        const a = center(subject);
        const b = center(object);
        const distance = Math.hypot(a.x - b.x, a.y - b.y);
        if (!best || distance < best.distance) best = { subject, object, distance };
      }
    }
    return best;
  }

  function syncRelationOverlaySize() {
    const overlayWidth = el.textScroll.clientWidth;
    const overlayHeight = Math.max(el.textScroll.clientHeight, el.textScroll.scrollHeight);
    el.overlay.style.width = `${overlayWidth}px`;
    el.overlay.style.height = `${overlayHeight}px`;
    el.overlay.setAttribute('viewBox', `0 0 ${overlayWidth} ${overlayHeight}`);
  }

  function drawRelationConnector() {
    const annotationId = activeAnnotationId();
    if (!annotationId) {
      el.relationPath.removeAttribute('d');
      return;
    }
    const pair = nearestRectPair(rectsForAnnotation(annotationId, 'subject'), rectsForAnnotation(annotationId, 'object'));
    if (!pair) {
      el.relationPath.removeAttribute('d');
      return;
    }
    const start = center(pair.subject);
    const end = center(pair.object);
    const direction = end.x >= start.x ? 1 : -1;
    start.x += direction * Math.min(18, Math.abs(pair.subject.right - pair.subject.left) / 2);
    end.x -= direction * Math.min(18, Math.abs(pair.object.right - pair.object.left) / 2);
    const bend = Math.max(36, Math.abs(end.y - start.y) * .35);
    const controlY = Math.min(start.y, end.y) - bend;
    el.relationPath.setAttribute('d', `M ${start.x} ${start.y} C ${start.x} ${controlY}, ${end.x} ${controlY}, ${end.x} ${end.y}`);
  }

  el.text.addEventListener('pointerover', (event) => {
    const target = event.target.closest('.text-fragment');
    if (!target) return;
    const tokenIndex = target.dataset.tokenIndex != null ? Number(target.dataset.tokenIndex) : null;
    renderHoverPopup(target, tokenIndex);
  });

  document.addEventListener('pointerover', (event) => {
    if (event.target.closest('.text-fragment, #hover-popup, [data-triple-id]')) return;
    el.hoverPopup.hidden = true;
  });

  el.text.addEventListener('click', (event) => {
    const target = event.target.closest('.text-fragment');
    if (!target) return;
    const sentenceId = target.dataset.sentenceId;
    if (!sentenceId) return;
    state.pinnedSentenceId = sentenceId;
    state.pinnedAnnotationId = null;
    state.hoveredAnnotationId = null;
    el.hoverPopup.hidden = true;
    updateActiveFragments();
    scrollSentenceIntoView(sentenceId);
  });

  function scrollTripleRowIntoView(annotationId) {
    requestAnimationFrame(() => {
      const row = Array.from(el.tripleList.querySelectorAll('[data-triple-id]'))
        .find((item) => item.dataset.tripleId === annotationId);
      if (!row) return;
      const paneRect = el.inspector.getBoundingClientRect();
      const rowRect = row.getBoundingClientRect();
      el.inspector.scrollTop += rowRect.top - paneRect.top - Math.round(el.inspector.clientHeight * .2);
    });
  }

  function scrollSentenceIntoView(sentenceId) {
    requestAnimationFrame(() => {
      const group = Array.from(el.tripleList.querySelectorAll('[data-sentence-id]'))
        .find((item) => item.dataset.sentenceId === sentenceId);
      if (!group) return;
      const paneRect = el.inspector.getBoundingClientRect();
      const groupRect = group.getBoundingClientRect();
      el.inspector.scrollTop += groupRect.top - paneRect.top - Math.round(el.inspector.clientHeight * .16);
    });
  }

  el.tripleList.addEventListener('pointerover', (event) => {
    const row = event.target.closest('[data-triple-id]');
    if (!row || state.pinnedAnnotationId) return;
    state.hoveredAnnotationId = row.dataset.tripleId;
    updateActiveFragments();
  });

  el.tripleList.addEventListener('pointerleave', () => {
    el.hoverPopup.hidden = true;
    if (state.pinnedAnnotationId) return;
    state.hoveredAnnotationId = null;
    updateActiveFragments();
  });

  el.tripleList.addEventListener('click', (event) => {
    const row = event.target.closest('[data-triple-id]');
    if (!row) return;
    state.pinnedAnnotationId = row.dataset.tripleId;
    state.pinnedSentenceId = row.closest('[data-sentence-id]')?.dataset.sentenceId || null;
    state.hoveredAnnotationId = null;
    updateActiveFragments();
  });

  function navigateTo(volumeId, pageNumber, pushHistory) {
    state.volumeId = volumeId;
    state.pageNumber = Number(pageNumber);
    loadPage(pushHistory);
  }

  function movePage(delta) {
    const volume = currentVolume();
    const position = pagePosition();
    const target = volume.availablePages[position + delta];
    if (target != null) navigateTo(state.volumeId, target, true);
  }

  async function prefetchAdjacentPages() {
    const volume = currentVolume();
    const position = pagePosition();
    for (const pageNumber of [volume.availablePages[position - 1], volume.availablePages[position + 1]]) {
      if (pageNumber == null) continue;
      const key = `${state.volumeId}:${pageNumber}`;
      if (state.prefetchedPages.has(key)) continue;
      state.prefetchedPages.set(
        key,
        fetch(canonicalPageUrl(state.volumeId, pageNumber)).catch(() => null)
      );
    }
  }

  el.previous.addEventListener('click', () => movePage(-1));
  el.next.addEventListener('click', () => movePage(1));
  el.volume.addEventListener('change', () => {
    const volume = state.index.volumes.find((item) => item.id === el.volume.value);
    navigateTo(volume.id, volume.availablePages[0], true);
  });
  el.pageInput.addEventListener('change', () => {
    const requested = Number(el.pageInput.value);
    const volume = currentVolume();
    if (volume.availablePages.includes(requested)) navigateTo(state.volumeId, requested, true);
    else {
      setMessage(`Page ${requested} is outside the selected chronicle ranges.`);
      updateNavigation();
    }
  });
  el.graphToggle.addEventListener('click', () => {
    if (!window.ChronicleGraph) return;
    const wasOpen = state.graphOpen;
    state.graphOpen = true;
    syncSiteNavigation();
    if (!wasOpen) {
      history.pushState(
        { volumeId: state.volumeId, pageNumber: state.pageNumber, graphOpen: true },
        '',
        currentPagePath()
      );
    }
    const annotation = annotationById(activeAnnotationId());
    const categories = annotation && annotation.axial && annotation.axial.status === 'accepted'
      ? {
          entities: [annotation.axial.subject.id, annotation.axial.object.id],
          relations: [annotation.axial.relation.id]
        }
      : null;
    window.ChronicleGraph.open({
      volumeId: state.volumeId,
      pageNumber: state.pageNumber,
      categories
    });
  });
  window.addEventListener('chronicle-graph-closed', () => {
    if (!state.graphOpen) return;
    state.graphOpen = false;
    syncSiteNavigation();
    history.pushState(
      { volumeId: state.volumeId, pageNumber: state.pageNumber, graphOpen: false },
      '',
      currentPagePath()
    );
  });
  el.textScroll.addEventListener('scroll', drawRelationConnector, { passive: true });
  const hidePopupForUserScroll = () => { el.hoverPopup.hidden = true; };
  el.textScroll.addEventListener('wheel', hidePopupForUserScroll, { passive: true });
  el.textScroll.addEventListener('touchmove', hidePopupForUserScroll, { passive: true });
  window.addEventListener('resize', () => {
    fitCanonicalText();
    syncRelationOverlaySize();
    drawRelationConnector();
  });
  new ResizeObserver(() => {
    syncRelationOverlaySize();
    drawRelationConnector();
  }).observe(el.textScroll);
  window.addEventListener('popstate', (event) => {
    const match = location.pathname.match(/^\/(chronicles|knowledge-graph)\/(vol\d+)\/(\d+)$/);
    if (!match) return;
    const graphOpen = match[1] === 'knowledge-graph';
    state.graphOpen = graphOpen;
    syncSiteNavigation();
    navigateTo(match[2], Number(match[3]), false);
    if (graphOpen && window.ChronicleGraph && !window.ChronicleGraph.isOpen()) {
      el.graphToggle.click();
    } else if (!graphOpen && window.ChronicleGraph && window.ChronicleGraph.isOpen()) {
      window.ChronicleGraph.close();
    }
  });
  document.addEventListener('keydown', (event) => {
    if (event.target.matches('input, select, textarea')) return;
    if (event.key === 'Escape') {
      state.pinnedAnnotationId = null;
      state.hoveredAnnotationId = null;
      state.pinnedSentenceId = null;
      updateActiveFragments();
    } else if (event.key === 'ArrowLeft' || event.key === 'PageUp') {
      event.preventDefault(); movePage(-1);
    } else if (event.key === 'ArrowRight' || event.key === 'PageDown') {
      event.preventDefault(); movePage(1);
    } else if (event.key.toLowerCase() === 'g' && window.ChronicleGraph) {
      el.graphToggle.click();
    }
  });

  initialize();
})();
