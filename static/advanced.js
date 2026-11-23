(() => {
  const form = document.getElementById('advanced-form');
  const model = document.getElementById('model');
  let metadataGeneration = 0;
  async function loadTracks(reset = false) {
    const selects = [...form.querySelectorAll('[data-track-select]')];
    if (!selects.length) return;
    const generation = ++metadataGeneration;
    const status = document.getElementById('track-help');
    status.textContent = 'Loading output track names. The first checkpoint load can take a minute…';
    try {
      const response = await fetch('/api/model-info?model=' + encodeURIComponent(model.value));
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Could not load model tracks.');
      if (generation !== metadataGeneration) return;
      for (const select of selects) {
        const previous = reset ? (select.hasAttribute('data-optional') ? '' : '0') : select.value;
        select.replaceChildren();
        if (select.hasAttribute('data-optional')) select.add(new Option('None — maximize the on-target score', ''));
        data.tasks.forEach((name, index) => select.add(new Option(`${index}: ${name}`, String(index))));
        select.value = [...select.options].some(option => option.value === previous) ? previous : select.options[0].value;
      }
      status.textContent = `${data.n_tasks} output track${data.n_tasks === 1 ? '' : 's'} · native model prediction scores; one output is passed through, multiple positions are averaged${data.tasks_truncated ? ' · first 500 tracks shown' : ''}.`;
    } catch (error) {
      if (generation === metadataGeneration) status.textContent = `${error.message} Track index 0 remains available; the server validates the selection.`;
    }
  }
  model.addEventListener('change', () => loadTracks(true));
  loadTracks();
  form.addEventListener('submit', () => {
    document.getElementById('run-progress').hidden = false;
    document.getElementById('advanced-results').setAttribute('aria-busy', 'true');
  });
  function download(text, type, filename) {
    const url = URL.createObjectURL(new Blob([text], { type }));
    const link = document.createElement('a');
    link.href = url; link.download = filename; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  const recordElement = document.getElementById('run-record');
  if (!recordElement) return;
  const record = JSON.parse(recordElement.textContent);
  document.getElementById('download-fasta')?.addEventListener('click', () => {
    download(record.results.fasta_text, 'text/plain', 'grelu-designed-sequence.fasta');
  });
  const cell = value => {
    let text = value == null ? '' : String(value);
    if (typeof value === 'string' && /^[\t\r\n]|^\s*[=+\-@＝＋－＠]/.test(text)) text = "'" + text;
    return '"' + text.replaceAll('"', '""') + '"';
  };
  document.getElementById('download-csv')?.addEventListener('click', () => {
    let rows;
    if (record.workflow === 'design') {
      rows = [['design', 'iteration', 'objective', 'mutation_position_1based', 'from', 'to']];
      for (const design of record.results.designs) for (const step of design.history)
        rows.push([design.seed_index + 1, step.iter, step.score, step.mutation?.pos_1based, step.mutation?.from, step.mutation?.to]);
    } else if (record.workflow === 'vep') {
      rows = [['rank', 'variant_id', 'ref', 'alt', 'track', 'reference_score', 'alternate_score', 'delta', 'empirical_p', 'rc_sign_agreement']];
      for (const variant of record.results.ranked_variants) for (const [track, delta] of Object.entries(variant.effects))
        rows.push([variant.rank, variant.variant_id, variant.ref, variant.alt, track, variant.ref_scores[track], variant.alt_scores[track], delta, variant.pvalues?.[track], variant.rc_sign_agreement]);
    } else if (record.workflow === 'attribution') {
      rows = [['position_1based', 'reference', 'signed_contribution', 'track', 'reference_score', 'method']];
      for (const position of record.results.positions) rows.push([position.position, position.reference, position.contribution, record.results.track, record.results.reference_score, record.results.method]);
    } else {
      rows = [['position_1based', 'reference', 'alternate', 'delta', 'track', 'reference_score', 'status']];
      for (const position of record.results.positions) {
        if (position.skipped) rows.push([position.position, position.reference, '', '', record.results.track, record.results.reference_score, position.skipped]);
        else for (const [base, delta] of Object.entries(position.deltas))
          rows.push([position.position, position.reference, base, delta, record.results.track, record.results.reference_score, 'scored']);
      }
    }
    download(rows.map(row => row.map(cell).join(',')).join('\r\n') + '\r\n', 'text/csv', `grelu-${record.workflow}.csv`);
  });
  for (const plot of document.querySelectorAll('.attribution-plot')) {
    const positions = JSON.parse(plot.dataset.positions);
    const scale = Math.max(...positions.map(row => Math.abs(row.contribution))) || 1;
    const width = 510 / positions.length;
    const add = (tag, attributes, text) => {
      const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
      for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, value);
      if (text !== undefined) node.textContent = text;
      plot.append(node);
      return node;
    };
    for (const value of [-scale, 0, scale]) {
      const y = 100 - value / scale * 75;
      add('line', { x1: 75, x2: 585, y1: y, y2: y, stroke: '#e0e5de' });
      add('text', { x: 2, y: y + 4, 'font-size': 10, fill: '#6b7b73' }, value.toPrecision(3));
    }
    positions.forEach((row, index) => {
      const height = Math.abs(row.contribution) / scale * 75;
      const bar = add('rect', { x: 75 + index * width, y: row.contribution >= 0 ? 100 - height : 100, width: Math.max(.5, width - .4), height, fill: row.contribution >= 0 ? '#286c52' : '#ad6a30' });
      const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
      title.textContent = `${row.position}: ${row.reference} ${row.contribution.toPrecision(5)}`;
      bar.append(title);
    });
    for (const index of new Set([0, Math.floor(positions.length / 2), positions.length - 1]))
      add('text', { x: 75 + (index + .5) * width, y: 202, 'text-anchor': 'middle', 'font-size': 10, fill: '#6b7b73' }, positions[index].position);
    add('text', { x: 330, y: 219, 'text-anchor': 'middle', 'font-size': 10, fill: '#6b7b73' }, 'Input position (1-based)');
  }
  for (const plot of document.querySelectorAll('.history-plot')) {
    const history = JSON.parse(plot.dataset.history);
    const scores = history.map(step => step.score);
    const minimum = Math.min(...scores), maximum = Math.max(...scores);
    const span = maximum - minimum || 1;
    const maxIteration = Math.max(1, ...history.map(step => step.iter));
    const y = score => maximum === minimum ? 80 : 140 - (score - minimum) * 115 / span;
    const add = (tag, attributes, text) => {
      const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
      for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, value);
      if (text !== undefined) node.textContent = text;
      plot.append(node);
    };
    for (const value of new Set([minimum, (minimum + maximum) / 2, maximum])) {
      add('line', { x1: 85, x2: 580, y1: y(value), y2: y(value), stroke: '#e0e5de' });
      add('text', { x: 2, y: y(value) + 4, fill: '#6b7b73', 'font-size': 11 }, value.toPrecision(4));
    }
    const points = history.map(step => `${85 + step.iter / maxIteration * 495},${y(step.score)}`).join(' ');
    add('polyline', { points, fill: 'none', stroke: '#286c52', 'stroke-width': 2 });
    for (const step of history) add('circle', { cx: 85 + step.iter / maxIteration * 495, cy: y(step.score), r: 3, fill: '#286c52' });
    add('text', { x: 335, y: 175, 'text-anchor': 'middle', fill: '#6b7b73', 'font-size': 11 }, 'Accepted optimization steps (iteration)');
  }
})();
