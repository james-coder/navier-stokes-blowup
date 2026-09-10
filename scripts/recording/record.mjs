// ScreenCLI's exported capture API records a deterministic, real installation.
// Usage: node scripts/recording/record.mjs /path/on/bulk/storage 0.1.0
import { launchSession } from 'screencli';
import { createServer } from 'node:http';
import { spawn } from 'node:child_process';
import { mkdir, writeFile, readFile, copyFile } from 'node:fs/promises';
import { resolve, join } from 'node:path';
import { createHash } from 'node:crypto';
import assert from 'node:assert/strict';

const [destination, version = '0.1.0'] = process.argv.slice(2);
if (!destination || !/^\d+\.\d+\.\d+$/.test(version)) throw new Error('Supply a fresh work directory and a numeric PyPI version');
const work = resolve(destination);
// Refuse reuse: a fresh venv and a fresh capture directory are part of the evidence.
await mkdir(work);
await mkdir(join(work, 'capture'));
const python = join(work, 'venv', 'bin', 'python');
const demo = `from ns_blowup import Simulation\nresult = Simulation((32,32), nu=0.05, device="cpu").run("taylor-green", t_end=1, frames=21)\nprint("Velocity shape:", result.velocity.shape)\nprint("Device:", result.metadata["device"])\nprint("Final energy:", result.diagnostics()["energy"][-1])\nprint("NPZ:", result.save("flow.npz"))\nprint("VTK:", result.export_vtk("flow.vtk"))\nresult.plot(quantity="vorticity", path="vorticity.png")\nprint("Plot: vorticity.png")\n`;
await writeFile(join(work, 'demo.py'), demo);
const steps = [
  ['Create a clean Python environment', 'python3', ['-m', 'venv', 'venv']],
  [`Install ns-blowup ${version} from PyPI — CPU`, python, ['-m', 'pip', 'install', '--quiet', '--disable-pip-version-check', `ns-blowup[plot]==${version}`]],
  ['Verify Python, package and device', python, ['-c', 'import sys, ns_blowup, jax; print(sys.version); print("ns-blowup", ns_blowup.__version__); print("JAX", jax.__version__); print("Devices:", jax.devices())']],
  ['Simulate the 2D Taylor–Green vortex and export results', python, ['demo.py']],
  ['Generate the offline Observatory', python, ['-m', 'ns_blowup.observatory', '--output', 'observatory.html']],
];
let current = -1, busy = false, failed = false, terminal = '', caption = 'A real clean installation, followed by simulation and the offline app.';
const transcript = [];
const html = `<!doctype html><html lang="en"><meta charset="utf-8"><title>Install and explore ns-blowup</title><style>body{margin:0;padding:38px;background:#111c2b;color:#e8f0fa;font:20px/1.5 system-ui}h1{margin:0;font-size:34px}p{color:#bfd3ea}button,a{font:inherit;padding:10px 18px;background:#dceeff;color:#102d48;border:0;border-radius:5px}a{display:inline-block;text-decoration:none;margin-right:12px}pre{height:420px;overflow:auto;background:#07101c;border:1px solid #45617a;padding:20px;font:18px/1.5 monospace;white-space:pre-wrap;overflow-wrap:anywhere}.note{font-size:16px}#caption{min-height:36px}</style><h1>Install and explore ns-blowup ${version}</h1><p id="caption"></p><pre id="terminal"></pre><button id="next">Run next step</button><a href="/vorticity.png" id="plot">View generated vorticity</a><a href="/observatory.html" id="app">Open generated Observatory</a><p class="note">CPU · 32×32 periodic Taylor–Green flow · The Observatory is a component preview; the complete smooth blowup field is not implemented.</p><script>async function refresh(){const s=await(await fetch('/state')).json();document.querySelector('#caption').textContent=s.caption;const p=document.querySelector('#terminal');if(p.textContent!==s.terminal){p.textContent=s.terminal;p.scrollTop=p.scrollHeight;}document.querySelector('#next').disabled=s.busy||s.done;window.recordingState=s;}document.querySelector('#next').onclick=()=>fetch('/next',{method:'POST'});setInterval(refresh,150);refresh();</script></html>`;
const server = createServer(async (req, res) => {
  if (req.url === '/state') {
    res.setHeader('Content-Type', 'application/json');
    return res.end(JSON.stringify({ current, busy, failed, done: current >= steps.length-1, terminal, caption }));
  }
  if (req.url === '/next' && req.method === 'POST') {
    if (busy || current >= steps.length-1) { res.statusCode = 409; return res.end(); }
    busy = true;
    const [title, command, args] = steps[++current];
    caption = title;
    const display = [command === python ? 'venv/bin/python' : command, ...args.map(a => /[\s;"\[\]]/.test(a) ? JSON.stringify(a) : a)].join(' ');
    terminal = `$ ${display}\n`;
    const entry = { title, command: display, output: '', started_at: new Date().toISOString() };
    transcript.push(entry);
    const child = spawn(command, args, { cwd: work, env: { ...process.env, JAX_PLATFORMS: 'cpu', JAX_ENABLE_X64: '1', MPLBACKEND: 'Agg', PYTHONUNBUFFERED: '1', PIP_CACHE_DIR: join(work, 'pip-cache') } });
    const append = chunk => { entry.output += chunk.toString(); terminal += chunk.toString(); };
    child.stdout.on('data', append); child.stderr.on('data', append);
    child.on('error', error => { append(String(error)); failed = true; busy = false; });
    child.on('close', code => { entry.exit_code = code; failed ||= code !== 0; terminal += `\nExit status: ${code}\n`; busy = false; });
    return res.end('started');
  }
  const assets = { '/': [html, 'text/html'], '/vorticity.png': ['vorticity.png', 'image/png'], '/observatory.html': ['observatory.html', 'text/html'] };
  if (req.url === '/favicon.ico') { res.statusCode = 204; return res.end(); }
  const asset = assets[req.url];
  if (!asset) { res.statusCode = 404; return res.end(); }
  try { res.setHeader('Content-Type', asset[1]); res.end(req.url === '/' ? asset[0] : await readFile(join(work, asset[0]))); }
  catch { res.statusCode = 404; res.end('Run the generation step first.'); }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const base = `http://127.0.0.1:${server.address().port}`;
const session = await launchSession({ headless: true, viewport: { width: 1280, height: 800 }, slowMo: 160, recordDir: join(work, 'capture') });
try {
  const page = session.page;
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(base);
  for (let i=0; i<steps.length; i++) {
    await page.locator('#next').click();
    await page.waitForFunction(index => window.recordingState?.current === index && !window.recordingState.busy, i, { timeout: 300000 });
    assert.equal(failed, false, `Step ${i} failed`);
    await page.waitForTimeout(2200);
    if (i === 3) {
      await page.locator('#plot').click();
      await page.waitForTimeout(4000);
      await page.screenshot({ path: join(work, 'thumbnail.png') });
      await page.goto(base);
    }
  }
  await page.locator('#app').click();
  await page.waitForFunction(() => document.querySelector('#tau')?.textContent);
  await page.waitForTimeout(2500);
  await page.locator('#time').fill('90');
  await page.locator('#resolution').selectOption('512');
  await page.locator('#follow').selectOption('world');
  await page.waitForTimeout(2200);
  await page.locator('#follow').selectOption('follow');
  await page.locator('#play').click();
  await page.waitForTimeout(1600);
  await page.locator('#play').click();
  await page.locator('#download').scrollIntoViewIfNeeded();
  const downloadEvent = page.waitForEvent('download');
  await page.locator('#download').click();
  await (await downloadEvent).saveAs(join(work, 'downloaded-observatory.json'));
  assert.deepEqual(JSON.parse(await readFile(join(work, 'downloaded-observatory.json'))), JSON.parse(await readFile(join(work, 'observatory.json'))));
  await page.waitForTimeout(2500);
  assert.deepEqual(errors, []);
} finally {
  const raw = await session.close();
  if (raw) await copyFile(raw, join(work, 'onboarding.webm'));
  server.close();
  const capture = await readFile(join(work, 'onboarding.webm'));
  await writeFile(join(work, 'recording.json'), JSON.stringify({ package_version: version, screencli: '0.3.12', playwright: '1.59.1', node: process.version, mode: 'ScreenCLI exported launchSession API with deterministic browser actions; local capture, no cloud agent', sha256: createHash('sha256').update(capture).digest('hex'), transcript }, null, 2)+'\n');
  await writeFile(join(work, 'transcript.md'), `# Actual installation transcript\n\nCPU, PyPI ns-blowup ${version}. Captured by ScreenCLI 0.3.12.\n\n`+transcript.map(entry => `## ${entry.title}\n\n\`\`\`console\n$ ${entry.command}\n${entry.output}\nExit status: ${entry.exit_code}\n\`\`\`\n`).join('\n'));
}
console.log(work);
