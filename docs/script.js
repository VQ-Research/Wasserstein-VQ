const tabs = document.querySelectorAll('[data-tab]');

tabs.forEach((tab) => {
  tab.addEventListener('click', () => {
    tabs.forEach((item) => {
      const selected = item === tab;
      item.classList.toggle('active', selected);
      item.setAttribute('aria-selected', String(selected));
      const panel = document.getElementById(item.dataset.tab);
      panel.hidden = !selected;
      panel.classList.toggle('active', selected);
    });
  });
});

const copyButton = document.querySelector('[data-copy]');
copyButton.addEventListener('click', async () => {
  const citation = document.getElementById('bibtex').innerText;
  let copied = false;

  try {
    await navigator.clipboard.writeText(citation);
    copied = true;
  } catch (_) {
    const textArea = document.createElement('textarea');
    textArea.value = citation;
    textArea.setAttribute('readonly', '');
    textArea.style.position = 'fixed';
    textArea.style.opacity = '0';
    document.body.appendChild(textArea);
    textArea.select();
    copied = document.execCommand('copy');
    textArea.remove();
  }

  copyButton.textContent = copied ? 'Copied' : 'Copy failed';
  setTimeout(() => { copyButton.textContent = 'Copy BibTeX'; }, 1600);
});
