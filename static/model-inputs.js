// Model-aware hints and local FASTA checks. The server remains authoritative.
(() => {
  const model = document.getElementById('model');
  if (!model) return;
  const form = model.form;
  const areas = [...form.querySelectorAll('textarea')];
  const hint = document.getElementById('model-input-hint');
  const help = document.getElementById('model-help');
  let feedback = document.getElementById('input-validation');
  if (!feedback && areas.length) {
    feedback = document.createElement('p');
    feedback.id = 'input-validation';
    feedback.className = 'input-validation';
    feedback.setAttribute('role', 'status');
    feedback.setAttribute('aria-live', 'polite');
    (form.querySelector('.actions') || form.lastElementChild).before(feedback);
  }
  function records(text) {
    const lines = text.split(/\r?\n/).map(line => line.trim()).filter(Boolean);
    if (!lines.length) return [];
    if (!lines[0].startsWith('>')) return [{ name: 'sequence', sequence: lines.join('') }];
    const result = [];
    for (const line of lines) {
      if (line.startsWith('>')) result.push({ name: line.slice(1).split(/\s/)[0] || `record_${result.length + 1}`, sequence: '' });
      else result[result.length - 1].sequence += line;
    }
    return result;
  }
  function update() {
    const option = model.selectedOptions[0];
    const expected = Number(option?.dataset.length);
    if (!expected) return;
    help.textContent = `${option.dataset.label} · requires ${expected.toLocaleString()} bases per record.`;
    hint.textContent = model.value === 'demo'
      ? 'Use quick-demo-200bp files. Random weights demonstrate the workflow only.'
      : expected === 2114
        ? 'Use trained-model-2114bp files. Quick-demo files are 200 bases and will not fit this model.'
        : 'Use aligned full-length windows for the model’s training assembly.';
    const notices = [];
    let invalid = false;
    for (const area of areas) {
      area.setCustomValidity('');
      if (area.dataset.sequenceInput === 'csv') continue;
      const parsed = records(area.value);
      if (!parsed.length) continue;
      const allowShort = form.id === 'compare' && model.value === 'demo';
      const mismatches = parsed.filter(row => allowShort ? row.sequence.length > expected : row.sequence.length !== expected);
      const invalidBases = parsed.some(row => !/^[ACGTN]+$/i.test(row.sequence));
      const summary = parsed.slice(0, 4).map(row => `${row.name}: ${row.sequence.length.toLocaleString()} bp`).join(' · ');
      const isQc = location.pathname === '/qc';
      if (mismatches.length) {
        invalid = true;
        const correction = expected === 200 ? 'Use quick-demo-200bp files.' : expected === 2114
          ? 'Choose Demo CNN for 200 bp files, or use trained-model-2114bp.' : 'Use a full-length model window.';
        const message = `${summary}. Expected ${allowShort ? 'up to ' : ''}${expected.toLocaleString()} bp per record. ${correction}`;
        notices.push(message);
        if (!isQc) area.setCustomValidity(message);
      } else if (invalidBases) {
        invalid = true;
        const message = 'DNA must contain only A, C, G, T or N, with optional FASTA headers.';
        notices.push(message);
        if (!isQc) area.setCustomValidity(message);
      } else notices.push(`${summary} · ${isQc ? 'Length matches selected model' : 'Input length matches'}.`);
    }
    if (feedback) {
      feedback.textContent = notices.join(' ');
      feedback.classList.toggle('has-error', invalid);
    }
  }
  model.addEventListener('change', update);
  areas.forEach(area => area.addEventListener('input', update));
  const upload = form.querySelector('#file');
  if (upload && areas.length === 1) {
    upload.addEventListener('change', async () => {
      const file = upload.files[0];
      if (!file) return;
      try {
        if (file.size > 550000) throw new Error('Choose a text file under 550 KB.');
        const text = await file.text();
        if (upload.files[0] !== file) return;
        areas[0].value = text.replace(/^\uFEFF/, '');
        // Submit the editable preview once; do not send a duplicate file payload.
        upload.value = '';
        areas[0].dispatchEvent(new Event('input', { bubbles: true }));
        let loaded = form.querySelector('.upload-summary');
        if (!loaded) { loaded = document.createElement('p'); loaded.className = 'model-help upload-summary'; upload.closest('.field').after(loaded); }
        loaded.textContent = `Loaded ${file.name} into the editor. You can edit it before running.`;
      } catch (error) {
        upload.value = '';
        if (feedback) { feedback.textContent = error.message; feedback.classList.add('has-error'); }
      }
    });
  }
  update();
})();
