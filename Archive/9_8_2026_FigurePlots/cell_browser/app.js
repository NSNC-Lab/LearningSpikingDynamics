(() => {
  'use strict';

  const data = window.CELL_BROWSER_DATA;
  const cells = data.cells;
  const responseOrder = ['Onset', 'Offset', 'Both', 'Neither', 'Unclassified'];
  const qualityOrder = [
    'High stereotypy', 'Moderate stereotypy', 'Low stereotypy',
    'Not really stereotyped', 'Non-stationary', 'Noise', 'Artifacting',
    'Unclassified'
  ];
  const responseColors = {
    Onset: '#c43c35', Offset: '#287aa2', Both: '#80589c',
    Neither: '#657078', Unclassified: '#857462'
  };

  const controls = {
    search: document.querySelector('#search'),
    response: document.querySelector('#response-filter'),
    quality: document.querySelector('#quality-filter'),
    layer: document.querySelector('#layer-filter'),
    notes: document.querySelector('#notes-only'),
    sort: document.querySelector('#sort-order')
  };
  const grid = document.querySelector('#cell-grid');
  const empty = document.querySelector('#empty-state');
  const matchCount = document.querySelector('#match-count');
  const totalCount = document.querySelector('#total-count');

  fillSelect(controls.response, responseOrder.filter(value => cells.some(cell => cell.response === value)));
  fillSelect(controls.quality, qualityOrder.filter(value => cells.some(cell => cell.quality.includes(value))));
  fillSelect(controls.layer, unique(cells.map(cell => cell.layer)).sort(naturalCompare));
  totalCount.textContent = `${cells.length} classified cells`;

  Object.values(controls).forEach(control => control.addEventListener('input', render));
  document.querySelector('#reset-filters').addEventListener('click', resetFilters);
  document.querySelector('[data-reset]').addEventListener('click', resetFilters);

  function fillSelect(select, values) {
    values.forEach(value => select.add(new Option(value, value)));
  }

  function unique(values) {
    return [...new Set(values.filter(Boolean))];
  }

  function naturalCompare(a, b) {
    return String(a).localeCompare(String(b), undefined, { numeric: true });
  }

  function filteredCells() {
    const query = controls.search.value.trim().toLowerCase();
    return cells.filter(cell => {
      const searchable = `cell ${cell.id} ${cell.note} ${cell.tuning}`.toLowerCase();
      return (!query || searchable.includes(query)) &&
        (!controls.response.value || cell.response === controls.response.value) &&
        (!controls.quality.value || cell.quality.includes(controls.quality.value)) &&
        (!controls.layer.value || cell.layer === controls.layer.value) &&
        (!controls.notes.checked || Boolean(cell.note));
    });
  }

  function sortCells(matches) {
    const sorted = [...matches];
    if (controls.sort.value === 'response') {
      sorted.sort((a, b) => responseOrder.indexOf(a.response) - responseOrder.indexOf(b.response) || a.id - b.id);
    } else if (controls.sort.value === 'classification') {
      sorted.sort((a, b) => qualityOrder.indexOf(a.quality[0]) - qualityOrder.indexOf(b.quality[0]) || a.id - b.id);
    } else {
      sorted.sort((a, b) => a.id - b.id);
    }
    return sorted;
  }

  function render() {
    const matches = sortCells(filteredCells());
    matchCount.textContent = matches.length;
    grid.replaceChildren(...matches.map(cellCard));
    grid.hidden = matches.length === 0;
    empty.hidden = matches.length !== 0;
    renderStats(matches);
  }

  function cellCard(cell) {
    const card = document.createElement('article');
    card.className = 'cell-card';
    card.style.setProperty('--response-color', responseColors[cell.response]);
    card.innerHTML = `
      <header class="card-header">
        <div>
          <h3 class="cell-title">Cell ${cell.id}</h3>
          <p class="cell-meta">${escapeHtml(cell.layer || 'Layer unavailable')} · ${escapeHtml(cell.tuning)}</p>
        </div>
        <span class="response-label">${escapeHtml(cell.response)}</span>
      </header>
      <div class="tag-row">${cell.quality.map(value => `<span class="tag">${escapeHtml(value)}</span>`).join('')}</div>
      ${cellFigure(cell)}
      <p class="note ${cell.note ? '' : 'empty'}">${escapeHtml(cell.note || 'No classification notes')}</p>`;
    return card;
  }

  function cellFigure(cell) {
    const width = 720;
    const left = 52;
    const right = 708;
    const plotWidth = right - left;
    const x = value => left + (value / data.duration) * plotWidth;
    const eventBands = data.eventWindows.map(window => {
      const x1 = x(window[0]);
      const x2 = x(window[1]);
      return `<rect class="event-band" x="${x1}" y="8" width="${x2 - x1}" height="212"/>
        <line class="event-line" x1="${x1}" x2="${x1}" y1="8" y2="220"/>
        <line class="event-line" x1="${x2}" x2="${x2}" y1="8" y2="220"/>`;
    }).join('');

    const spikes = cell.trials.map((trial, trialIndex) => {
      const y1 = 16 + trialIndex * 7.3;
      const times = Array.isArray(trial) ? trial : [trial];
      return times.map(time => `<line class="spike-line" x1="${x(time).toFixed(2)}" x2="${x(time).toFixed(2)}" y1="${y1.toFixed(1)}" y2="${(y1 + 5.2).toFixed(1)}"/>`).join('');
    }).join('');

    const psthMax = Math.max(1, ...cell.psth);
    const psthPoints = cell.psth.map((value, index) => {
      const px = x(data.binTimes[index]);
      const py = 170 - (value / psthMax) * 55;
      return `${px.toFixed(1)},${py.toFixed(1)}`;
    }).join(' ');

    const wavePoints = data.waveform.map((value, index) => {
      const px = x(data.waveformTimes[index]);
      const py = 201 - value * 15;
      return `${px.toFixed(1)},${py.toFixed(1)}`;
    }).join(' ');

    const ticks = [0, 0.5, 1, 1.5, 2, 2.5].map(value => `
      <line class="grid-line" x1="${x(value)}" x2="${x(value)}" y1="8" y2="220"/>
      <text class="axis-text" x="${x(value)}" y="235" text-anchor="middle">${value.toFixed(1)}</text>`).join('');

    return `<svg class="cell-figure" viewBox="0 0 ${width} 244" role="img" aria-label="Raster, PSTH, and target waveform for cell ${cell.id}">
      ${eventBands}${ticks}
      <line class="grid-line" x1="${left}" x2="${right}" y1="100" y2="100"/>
      <line class="grid-line" x1="${left}" x2="${right}" y1="178" y2="178"/>
      ${spikes}
      <polyline class="plot-line psth-line" points="${psthPoints}"/>
      <polyline class="plot-line" points="${wavePoints}"/>
      <text class="axis-text" x="8" y="54">Trials</text>
      <text class="axis-text" x="8" y="143">PSTH</text>
      <text class="axis-text" x="8" y="204">Target</text>
      <text class="axis-text" x="380" y="242" text-anchor="middle">Time (s)</text>
      <text class="axis-text" x="${right - 2}" y="113" text-anchor="end">max ${psthMax.toFixed(1)}</text>
    </svg>`;
  }

  function renderStats(matches) {
    renderStatGroup(document.querySelector('#response-stats'), responseOrder, matches, 'response');
    renderStatGroup(document.querySelector('#quality-stats'), qualityOrder, matches, 'quality');
  }

  function renderStatGroup(container, categories, matches, kind) {
    const rows = categories.map(category => {
      const count = matches.filter(cell => kind === 'response' ? cell.response === category : cell.quality.includes(category)).length;
      return { category, count };
    }).filter(item => item.count > 0);
    const max = Math.max(1, ...rows.map(item => item.count));
    container.replaceChildren(...rows.map(item => {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'stat-row';
      button.title = `Filter to ${item.category}`;
      button.innerHTML = `<span class="stat-label">${escapeHtml(item.category)}</span>
        <span class="stat-track"><span class="stat-fill" style="display:block;width:${(item.count / max) * 100}%"></span></span>
        <span class="stat-value">${item.count}</span>`;
      button.addEventListener('click', () => {
        controls[kind].value = item.category;
        render();
        window.scrollTo({ top: 0, behavior: 'smooth' });
      });
      return button;
    }));
  }

  function resetFilters() {
    controls.search.value = '';
    controls.response.value = '';
    controls.quality.value = '';
    controls.layer.value = '';
    controls.notes.checked = false;
    render();
  }

  function escapeHtml(value) {
    return String(value).replace(/[&<>'"]/g, character => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'
    })[character]);
  }

  render();
})();
