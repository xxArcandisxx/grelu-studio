const menu = document.querySelector('.menu-toggle');
const navigation = document.querySelector('#navigation');
const navigationBackdrop = document.querySelector('.navigation-backdrop');
const workspaceShell = document.querySelector('.workspace-shell');
const mobileNavigation = window.matchMedia('(max-width: 760px)');

function setNavigation(open, restoreFocus = false) {
  open = Boolean(open && mobileNavigation.matches);
  menu?.setAttribute('aria-expanded', String(open));
  document.body.classList.toggle('navigation-open', open);
  if (navigationBackdrop) navigationBackdrop.hidden = !open;
  if (navigation) navigation.inert = mobileNavigation.matches && !open;
  if (workspaceShell) workspaceShell.inert = open;
  if (open) navigation?.querySelector('.navigation-close')?.focus();
  else if (restoreFocus) menu?.focus();
}

menu?.addEventListener('click', () => setNavigation(menu.getAttribute('aria-expanded') !== 'true'));
document.querySelector('.navigation-close')?.addEventListener('click', () => setNavigation(false, true));
navigationBackdrop?.addEventListener('click', () => setNavigation(false, true));
mobileNavigation.addEventListener('change', () => setNavigation(false));
document.addEventListener('keydown', event => {
  if (menu?.getAttribute('aria-expanded') !== 'true') return;
  if (event.key === 'Escape') setNavigation(false, true);
  if (event.key === 'Tab' && navigation) {
    const controls = Array.from(navigation.querySelectorAll('a[href], button:not(:disabled), summary, [tabindex="0"]'))
      .filter(element => element.offsetParent !== null);
    const first = controls[0];
    const last = controls[controls.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last?.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first?.focus();
    }
  }
});
setNavigation(false);
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
