// Model-aware hints and local FASTA checks. The server remains authoritative.
(() => {
  const model = document.getElementById('model');
  if (!model) return;
  const form = model.form;
  const isQc = document.body.dataset.workflow === 'qc';
  const areas = [...form.querySelectorAll('textarea')];
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
    const lines = text.replace(/\r\n?/g, '\n').split('\n').map(line => line.replace(/^[ \t\v\f]+|[ \t\v\f]+$/g, '')).filter(Boolean);
    if (!lines.length) return [];
    if (!lines[0].startsWith('>')) return [{ name: 'sequence', sequence: lines.join('') }];
    const result = [];
    for (const line of lines) {
      if (line.startsWith('>')) result.push({ name: line.slice(1).trim().split(/\s/)[0], sequence: '' });
      else result[result.length - 1].sequence += line;
    }
    return result;
  }
  function update() {
    const option = model.selectedOptions[0];
    const expected = Number(option?.dataset.length);
    if (!expected) {
      const message = option?.dataset.lengthError || 'Required input length is unknown. Verified compatibility is unavailable.';
      help.textContent = message;
      areas.forEach(area => area.setCustomValidity(message));
      if (feedback) { feedback.textContent = message; feedback.classList.add('has-error'); feedback.classList.remove('has-warning'); }
      return;
    }
    help.textContent = form.id === 'compare' && model.value === 'demo'
      ? `${option.dataset.label} · up to ${expected.toLocaleString()} bases; shorter inputs are center N-padded under the configured demo policy and effective inputs are recorded.`
      : isQc
        ? `${option.dataset.label} · checks compatibility against ${expected.toLocaleString()} bases per record. QC accepts other lengths.`
        : `${option.dataset.label} · requires ${expected.toLocaleString()} bases per record.`;
    const notices = [];
    let invalid = false;
    let warning = false;
    for (const area of areas) {
      area.setCustomValidity('');
      if (area.dataset.sequenceInput === 'csv') continue;
      const parsed = records(area.value);
      if (!parsed.length) continue;
      const allowShort = form.id === 'compare' && model.value === 'demo';
      const mismatches = parsed.filter(row => allowShort ? row.sequence.length > expected : row.sequence.length !== expected);
      const invalidBases = parsed.some(row => !/^[ACGTN]+$/i.test(row.sequence));
      const invalidIdentifiers = parsed.some(row => !row.name) || new Set(parsed.map(row => row.name)).size !== parsed.length;
      const summary = parsed.slice(0, 4).map(row => `${row.name}: ${row.sequence.length.toLocaleString()} bp`).join(' · ');
      if (invalidIdentifiers) {
        invalid = true;
        const message = 'Every FASTA record needs a nonempty, distinct identifier.';
        notices.push(message);
        area.setCustomValidity(message);
      } else if (invalidBases) {
        invalid = true;
        const message = 'DNA must contain only A, C, G, T or N, with optional FASTA headers.';
        notices.push(message);
        area.setCustomValidity(message);
      } else if (mismatches.length) {
        invalid = !isQc;
        warning = isQc;
        const correction = isQc
          ? 'QC can still run; length mismatches will be included in the results.'
          : 'Select Load example for matching inputs, or choose a model that accepts your sequence length.';
        const mismatchedSummary = mismatches.slice(0, 4).map(row => `${row.name}: ${row.sequence.length.toLocaleString()} bp`).join(' · ');
        const message = `${isQc ? 'Model compatibility warning: ' : ''}${mismatchedSummary}. Expected ${allowShort ? 'up to ' : ''}${expected.toLocaleString()} bp per record. ${correction}`;
        notices.push(message);
        if (!isQc) area.setCustomValidity(message);
      } else notices.push(`${summary} · ${isQc ? 'Length matches selected model' : 'Input length matches'}.`);
    }
    if (feedback) {
      feedback.textContent = notices.join(' ');
      feedback.classList.toggle('has-error', invalid);
      feedback.classList.toggle('has-warning', warning && !invalid);
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
        if (feedback) { feedback.textContent = error.message; feedback.classList.add('has-error'); feedback.classList.remove('has-warning'); }
      }
    });
  }
  update();
})();
