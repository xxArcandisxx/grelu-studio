const menu = document.querySelector('.menu-toggle');
menu?.addEventListener('click', () => {
  const open = menu.getAttribute('aria-expanded') !== 'true';
  menu.setAttribute('aria-expanded', String(open));
  document.body.classList.toggle('navigation-open', open);
});
document.querySelector('.navigation-close')?.addEventListener('click', () => {
  document.body.classList.remove('navigation-open');
  menu?.setAttribute('aria-expanded', 'false');
  menu?.focus();
});
document.addEventListener('keydown', event => {
  if (event.key === 'Escape') {
    document.body.classList.remove('navigation-open');
    menu?.setAttribute('aria-expanded', 'false');
  }
});
for (const form of document.querySelectorAll('form[method="post"]:not(#scan-form)')) {
  form.addEventListener('submit', () => {
    const button = form.querySelector('button[type="submit"], button:not([type])');
    if (button) { button.textContent = 'Working…'; form.setAttribute('aria-busy', 'true'); }
  });
}
for (const input of document.querySelectorAll('textarea')) {
  const hint = document.createElement('div');
  hint.className = 'input-meta';
  input.after(hint);
  const update = () => { hint.textContent = input.value.trim() ? `${input.value.length.toLocaleString()} characters entered` : 'Paste your data or upload a file'; };
  input.addEventListener('input', update);
  update();
}
