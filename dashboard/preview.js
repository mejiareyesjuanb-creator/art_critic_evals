// Render the dashboard page locally with a small stand-in for the hosted dashboard runtime,
// and save screenshots. Checks for script errors and horizontal overflow at desktop and phone widths.
//
//   python dashboard/export.py                       (writes dashboard/build/*.json)
//   cd dashboard && npm install && node preview.js   (writes dashboard/build/preview-*.png)
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

const dir = __dirname;
const build = path.join(dir, 'build');
const page = fs.readFileSync(path.join(dir, 'index.html'), 'utf8');
const data = Object.fromEntries(['runs', 'results'].map((n) => {
  const file = path.join(build, `${n}.json`);
  if (!fs.existsSync(file)) { console.error(`Missing ${file}. Run: python dashboard/export.py`); process.exit(1); }
  return [n, JSON.parse(fs.readFileSync(file, 'utf8'))];
}));

// Stand-in values for the design tokens the hosted page gets from its host.
const TOKENS = `:root{--color-fg:#111;--color-fg-muted:#666;--color-bg:#fff;--color-panel:#fafafa;--color-border-line:#ddd;
--color-ok:#0a0;--color-warn:#e90;--color-bad:#c33;--cds-chart-grid:#eee;--cds-chart-reference:#999;--cds-text-secondary:#666;
--cds-surface-popover:#fff;--cds-border:#ddd;--cds-gap-xs:4px;--cds-gap-sm:8px;--cds-gap-md:12px;--cds-gap-xl:28px;
--cds-pad-sm:6px;--cds-pad-md:10px;--cds-pad-lg:16px;--cds-radius:8px;--cds-font-size-caption:12px;--cds-font-size-heading:18px;
--cds-font-size-title:28px;--font-anthropic-serif:Georgia,serif;--font-anthropic-sans:system-ui}
body{font-family:system-ui;margin:0;padding:16px;background:var(--color-bg)}`;

(async () => {
  const browser = await chromium.launch();
  let failed = false;
  for (const [width, scheme] of [[1200, 'light'], [400, 'dark']]) {
    const p = await browser.newPage({ viewport: { width, height: 900 }, colorScheme: scheme });
    const errors = [];
    p.on('pageerror', (e) => errors.push(e.message));
    p.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
    await p.setContent(`<style>${TOKENS}</style><div id="dash-root"></div>`);
    await p.addScriptTag({ path: path.join(dir, 'node_modules', 'd3', 'dist', 'd3.min.js') });
    await p.evaluate(({ page, data }) => {
      const calcs = {}, draws = {}, params = {};
      const value = (id) => calcs[id]
        ? { status: 'ok', data: calcs[id].fn(...calcs[id].inputs.map((i) => value(i).data)) }
        : { status: 'ok', data: data[id] };
      const redraw = () => {
        for (const [name, d] of Object.entries(draws)) {
          for (const el of document.querySelectorAll(`[data-draw="${name}"]`)) {
            if (typeof d === 'function') d(el);
            else { el._made ??= (el.replaceChildren(), d.mount(el)); d.update(el, el._made); }
          }
        }
      };
      window.dash = {
        data: value, calc: (id, o) => { calcs[id] = o; }, loader() {}, onData: (fn) => fn(),
        draw: (name, fn) => { if (fn) draws[name] = fn; }, colors: ['#2a78d6'], params: () => ({ ...params }),
        setParams: (o) => { Object.assign(params, o); redraw(); }, resetParams() {}, setLink() {}, refresh() {},
      };
      const root = document.getElementById('dash-root');
      root.innerHTML = page;
      for (const s of [...root.querySelectorAll('script')]) {
        const n = document.createElement('script'); n.textContent = s.textContent; s.replaceWith(n);
      }
      redraw();
    }, { page, data });
    await p.waitForTimeout(300);
    const shot = path.join(build, `preview-${width}.png`);
    await p.screenshot({ path: shot, fullPage: true });
    const overflow = await p.evaluate(() => document.documentElement.scrollWidth > window.innerWidth);
    console.log(`${width}px (${scheme}): errors=${errors.length ? errors.join(' | ') : 'none'}, overflow=${overflow} -> ${path.relative(dir, shot)}`);
    console.log('  takeaway:', await p.locator('.vg-take').innerText());
    failed ||= errors.length > 0 || overflow;
    await p.close();
  }
  await browser.close();
  process.exit(failed ? 1 : 0);
})();
