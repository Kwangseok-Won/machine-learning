(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  $('study-page').addEventListener('change', event => { location.href = event.target.value; });
  const data = window.BOOK;
  const kind = document.body.dataset.viewer;
  if (!kind) return;
  const items = kind === 'blocks' ? data.blocks : data.lessons;
  const catalog = $('catalog');
  const groups = new Map();
  items.forEach((item, index) => {
    const groupName = kind === 'blocks' ? '첨부 그림 · 아래에서 위로' : item.group;
    if (!groups.has(groupName)) {
      const section = document.createElement('section');
      const heading = document.createElement('h3'); heading.textContent = groupName;
      section.append(heading); catalog.append(section); groups.set(groupName, section);
    }
    const a = document.createElement('a');
    a.href = '#fn=' + encodeURIComponent(item.key); a.textContent = item.label || item.key;
    a.dataset.index = String(index); groups.get(groupName).append(a);
  });
  let itemIndex = 0, stepIndex = 0;
  function go(index, step = 0) {
    if (index < 0 || index >= items.length) return;
    location.hash = 'fn=' + encodeURIComponent(items[index].key) + '&step=' + step;
  }
  function link(label, href, parent) {
    const a = document.createElement('a'); a.textContent = label; a.href = href; parent.append(a);
  }
  function renderCode(item, step) {
    const pre = $('code'); pre.replaceChildren();
    const activeLines = step.lines || [step.line];
    item.source.split('\n').forEach((line, i) => {
      const span = document.createElement('span');
      const number = item.start + i;
      span.className = 'line' + (activeLines.includes(number) ? ' active' : '');
      span.dataset.line = String(number);
      const num = document.createElement('span'); num.className = 'num'; num.textContent = String(number);
      span.append(num, document.createTextNode(line)); pre.append(span);
    });
    const active = pre.querySelector('.active');
    if (active) pre.scrollTop = Math.max(0, active.offsetTop - pre.clientHeight / 3);
  }
  function render() {
    const item = items[itemIndex], step = item.steps[stepIndex];
    $('name').textContent = item.label || item.key;
    $('purpose').textContent = item.title;
    $('where').textContent = item.file + ' · 원본 파일 ' + item.start + '행부터';
    $('given').textContent = item.given; $('result').textContent = item.result;
    $('caller').textContent = '어디서 부르나? ' + item.caller;
    $('step-title').textContent = step.title;
    $('before').textContent = step.before; $('after').textContent = step.after;
    $('explanation').textContent = step.text;
    $('position').textContent = '설명 ' + (stepIndex + 1) + ' / ' + item.steps.length;
    $('item-count').textContent = (itemIndex + 1) + ' / ' + items.length + (kind === 'blocks' ? '개 블록' : '개 항목');
    $('prev').disabled = stepIndex === 0; $('next').disabled = stepIndex === item.steps.length - 1;
    $('prev-item').disabled = itemIndex === 0; $('next-item').disabled = itemIndex === items.length - 1;
    catalog.querySelectorAll('a').forEach(a => {
      if (Number(a.dataset.index) === itemIndex) a.setAttribute('aria-current', 'page');
      else a.removeAttribute('aria-current');
    });
    renderCode(item, step);
    const calls = $('calls'); calls.replaceChildren();
    (item.calls || []).forEach(key => link(key, 'Transformer_Code_Walkthrough_ELI5.html#fn=' + encodeURIComponent(key), calls));
    link('파일 전체 원문', 'Transformer_AI_Code_Book.html#file=' + encodeURIComponent(item.file), calls);
    if (kind === 'blocks') {
      const highlight = $('highlight'); highlight.replaceChildren();
      (item.rects || []).forEach(r => {
        const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        ['x', 'y', 'width', 'height'].forEach((attr, i) => rect.setAttribute(attr, r[i]));
        rect.setAttribute('rx', 5); highlight.append(rect);
      });
      (item.paths || []).forEach(d => {
        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        path.setAttribute('d', d); highlight.append(path);
      });
      $('shape').textContent = item.shape;
      $('block-position').textContent = item.label;
    }
  }
  function loadRoute() {
    const query = new URLSearchParams(location.hash.slice(1));
    const found = items.findIndex(item => item.key === query.get('fn'));
    itemIndex = found < 0 ? 0 : found;
    const step = Number(query.get('step') || 0);
    stepIndex = Number.isInteger(step) ? Math.max(0, Math.min(step, items[itemIndex].steps.length - 1)) : 0;
    render();
  }
  $('prev').addEventListener('click', () => go(itemIndex, stepIndex - 1));
  $('next').addEventListener('click', () => go(itemIndex, stepIndex + 1));
  $('prev-item').addEventListener('click', () => go(itemIndex - 1));
  $('next-item').addEventListener('click', () => go(itemIndex + 1));
  $('filter').addEventListener('input', event => {
    const value = event.target.value.toLowerCase().trim();
    catalog.querySelectorAll('a').forEach(a => { a.hidden = !a.textContent.toLowerCase().includes(value); });
    groups.forEach(group => { group.hidden = !Array.from(group.querySelectorAll('a')).some(a => !a.hidden); });
  });
  window.addEventListener('hashchange', loadRoute);
  loadRoute();
})();
