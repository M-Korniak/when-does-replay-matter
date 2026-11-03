'use strict';

const backToTop = document.querySelector('.scroll-to-top');
const syncScrollButton = () => backToTop.classList.toggle('visible', window.scrollY > 500);
window.addEventListener('scroll', syncScrollButton, { passive: true });
syncScrollButton();

const copyButton = document.querySelector('.copy-bibtex-btn');
const copyStatus = document.querySelector('.copy-status');
copyButton.addEventListener('click', async () => {
  const citation = document.querySelector('#bibtex-code code').textContent;
  try {
    await navigator.clipboard.writeText(citation);
    copyButton.textContent = 'Copied!';
    copyStatus.textContent = 'Citation copied to clipboard.';
    window.setTimeout(() => { copyButton.textContent = 'Copy citation'; }, 2500);
  } catch {
    const range = document.createRange();
    range.selectNodeContents(document.querySelector('#bibtex-code'));
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    copyStatus.textContent = 'Citation selected. Use your browser’s Copy command, or download the .bib file.';
  }
});

const dialog = document.querySelector('#figure-dialog');
const enlargedFigure = document.querySelector('#enlarged-figure');
let figureTrigger;
document.querySelectorAll('[data-zoom]').forEach(link => {
  link.addEventListener('click', event => {
    if (!dialog.showModal || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    const original = link.querySelector('img');
    figureTrigger = link;
    enlargedFigure.src = link.href;
    enlargedFigure.alt = original.alt;
    enlargedFigure.classList.toggle('portrait', original.naturalHeight > original.naturalWidth * 0.85);
    document.querySelector('#figure-dialog-title').textContent = original.alt;
    dialog.showModal();
    document.body.classList.add('figure-open');
    dialog.querySelector('.zoom-scroll').scrollTo(0, 0);
  });
});
dialog.querySelector('.close-dialog').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', event => { if (event.target === dialog) {
  const box = dialog.getBoundingClientRect();
  if (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom) dialog.close();
}});
dialog.addEventListener('close', () => {
  document.body.classList.remove('figure-open');
  figureTrigger?.focus({ preventScroll: true });
});

