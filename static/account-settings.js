// Preview a selected picture locally; saving still requires an explicit submit.
(() => {
  const input = document.getElementById('picture');
  const preview = document.getElementById('picture-preview');
  const current = document.getElementById('current-picture');
  if (!input || !preview || !current) return;
  let previewUrl;
  const clear = () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = undefined;
    preview.hidden = true;
    preview.removeAttribute('src');
    current.hidden = false;
  };
  input.addEventListener('change', () => {
    clear();
    input.setCustomValidity('');
    const file = input.files[0];
    if (!file) return;
    if (file.size > 400000) {
      input.setCustomValidity('Choose a profile picture smaller than 400 KB.');
      input.reportValidity();
      return;
    }
    previewUrl = URL.createObjectURL(file);
    preview.src = previewUrl;
    preview.hidden = false;
    current.hidden = true;
  });
  preview.addEventListener('error', () => {
    clear();
    input.setCustomValidity('Choose a valid PNG, JPEG, or WebP image.');
    input.reportValidity();
  });
  window.addEventListener('pagehide', clear);
})();
