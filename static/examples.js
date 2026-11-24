// Loading and downloading examples only prepares inputs; it never submits a form.
(() => {
  const generations = new WeakMap();
  for (const form of document.forms) {
    generations.set(form,0);
    for (const type of ['input','change']) form.addEventListener(type,()=>generations.set(form,(generations.get(form)||0)+1));
  }
  const control = (form,name) => form.querySelector(`[name="${name}"], [id="${name}"]`);
  const query = form => {
    const result = new URLSearchParams();
    for (const name of ['model','model_a','model_b','length','kind','repo']) {
      const field = control(form,name);
      if (field && field.value !== '') result.set(name,field.value);
    }
    return result.toString();
  };
  for (const button of document.querySelectorAll('[data-load-example], [data-download-example]')) {
    button.addEventListener('click',async()=>{
      const form = button.form;
      const status = form.querySelector('[data-example-status]');
      const download = button.hasAttribute('data-download-example');
      const workflow = button.dataset.loadExample || button.dataset.downloadExample;
      const generation = (generations.get(form)||0)+1;
      generations.set(form,generation);
      button.disabled = true;
      const pendingMessage = download ? 'Preparing example files…' : 'Loading example inputs…';
      if (status) status.textContent = pendingMessage;
      const stillCurrent = () => {
        if (generations.get(form)===generation) return true;
        if (status && status.textContent===pendingMessage) status.textContent='Inputs changed. Select Load example again to match the current settings.';
        return false;
      };
      try {
        const url = `/api/examples/${encodeURIComponent(workflow)}${download?'/download':''}?${query(form)}`;
        const response = await fetch(url, { cache: 'no-store' });
        if (!response.ok) {
          const error = await response.json().catch(()=>({}));
          throw new Error(error.error || 'Could not load the example. Please try again.');
        }
        if (download) {
          const blob = await response.blob();
          if (!stillCurrent()) return;
          const header = response.headers.get('Content-Disposition') || '';
          const filename = /filename="?([^";]+)"?/.exec(header)?.[1] || workflow+'-example.txt';
          const objectUrl = URL.createObjectURL(blob);
          const anchor = document.createElement('a');anchor.href=objectUrl;anchor.download=filename;anchor.click();
          setTimeout(()=>URL.revokeObjectURL(objectUrl),1000);
          if (status) status.textContent = 'Example files downloaded. Your inputs have not changed.';
        } else {
          const data = await response.json();
          if (!stillCurrent()) return;
          form.querySelectorAll('input[type="file"]').forEach(input=>{input.value='';});
          form.querySelectorAll('.upload-summary').forEach(node=>node.remove());
          for (const [name,value] of Object.entries(data.fields)) {
            const field = control(form,name);
            if (!field) continue;
            field.setCustomValidity('');
            if (field.type==='checkbox') field.checked=Boolean(value);
            else {
              if (field.tagName==='SELECT' && ![...field.options].some(option=>option.value===String(value)))
                field.add(new Option(`Track index ${value}`,String(value)));
              field.value=String(value);
            }
            field.dispatchEvent(new Event('input',{bubbles:true}));
            field.dispatchEvent(new Event('change',{bubbles:true}));
          }
          if (status) status.textContent = data.message;
        }
      } catch(error) {
        if (generations.get(form)===generation && status) status.textContent=error.message;
      } finally {button.disabled=false;}
    });
  }
})();
