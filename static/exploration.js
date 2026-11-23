(() => {
  for (const input of document.querySelectorAll('[data-load-target]')) {
    input.addEventListener('change', async () => {
      const file = input.files[0];
      if (!file) return;
      try {
        const limit = input.dataset.loadTarget === 'matrices' ? 100000 : 450000;
        if (file.size > limit) throw new Error(`Choose a file under ${(limit/1000).toFixed(0)} KB.`);
        const text = await file.text();
        if (input.files[0] !== file) return;
        const editor = document.getElementById(input.dataset.loadTarget);
        editor.value = text.replace(/^\uFEFF/, '');
        input.value = '';
        editor.dispatchEvent(new Event('input', { bubbles: true }));
        document.getElementById('file-status').textContent = `Loaded ${file.name} into the editor.`;
      } catch (error) {
        input.value = '';
        document.getElementById('file-status').textContent = error.message;
      }
    });
  }
  for (const select of document.querySelectorAll('[data-track-source]')) {
    const source = document.getElementById(select.dataset.trackSource);
    let generation = 0;
    const load = async reset => {
      const current = ++generation;
      const status = document.getElementById('track-status');
      status.textContent = 'Loading output tracks. The first checkpoint load can take a minute…';
      try {
        const response = await fetch('/api/model-info?model=' + encodeURIComponent(source.value));
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Could not inspect model tracks.');
        if (generation !== current) return;
        const previous = reset ? '0' : select.value;
        select.replaceChildren(...data.tasks.map((name,index) => new Option(`${index}: ${name}`,String(index))));
        select.value = [...select.options].some(option => option.value === previous) ? previous : '0';
        status.textContent = 'Track names loaded. Choose tracks with comparable biological meaning.';
      } catch (error) { if (generation === current) status.textContent = error.message; }
    };
    source.addEventListener('change', () => load(true));
    load(false);
  }
  document.getElementById('exploration-form').addEventListener('submit', () => {
    document.getElementById('exploration-progress').hidden = false;
  });
  const element = document.getElementById('run-record');
  if (!element) return;
  const result = JSON.parse(element.textContent);
  document.getElementById('exploration-csv').addEventListener('click', () => {
    let keys,rows;
    if (result.workflow === 'pwm-scan') { keys = ['record','motif','start','end','strand','matched_sequence','score','relative_score']; rows=result.hits; }
    else if (result.workflow === 'motif-simulation') { keys=['position','strand','partner_gap','score','delta','control_score','motif_minus_control']; rows=result.results.conditions; }
    else { keys=['id','score_a','score_b','raw_difference','transformation_a','transformation_b']; rows=result.results.comparisons; }
    const cell = value => {
      let text = value == null ? '' : String(value);
      if (typeof value === 'string' && text !== '+' && text !== '-' && /^[\t\r\n]|^\s*[=+\-@＝＋－＠]/.test(text)) text = "'" + text;
      return '"' + text.replaceAll('"','""') + '"';
    };
    const csv = [keys,...rows.map(row => keys.map(key => row[key]))].map(row => row.map(cell).join(',')).join('\r\n')+'\r\n';
    const url = URL.createObjectURL(new Blob([csv],{type:'text/csv'}));
    const link = document.createElement('a'); link.href=url;link.download=`grelu-${result.workflow}.csv`;link.click();
    setTimeout(() => URL.revokeObjectURL(url),1000);
  });
})();
