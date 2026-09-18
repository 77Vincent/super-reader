import Offcanvas from 'bootstrap/js/dist/offcanvas';

// Close Doks' mobile menu after following an anchor on this single-page site.
const menu = document.querySelector('#offcanvasNavMain');
menu?.addEventListener('click', (event) => {
  const link = event.target.closest('a[href]');
  if (link?.hash && link.origin === location.origin && link.pathname === location.pathname) {
    Offcanvas.getInstance(menu)?.hide();
  }
});
