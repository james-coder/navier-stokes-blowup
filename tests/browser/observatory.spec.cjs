const { test, expect } = require('@playwright/test');
const fs = require('node:fs/promises');

// Check both the committed demo and a freshly generated, nondefault-h artifact.
for (const artifact of ['docs/observatory', 'outputs/browser/observatory']) {
  test.describe(artifact, () => {
    let errors;
    test.beforeEach(async ({ page }) => {
      errors = [];
      page.on('pageerror', error => errors.push(error.message));
      page.on('console', message => {
        if (message.type() === 'error') errors.push(message.text());
      });
      await page.route('**/favicon.ico', route => route.fulfill({ status: 204 }));
      await page.goto(`/${artifact}.html`);
      await expect(page.locator('#tau')).not.toBeEmpty();
    });
    test.afterEach(() => expect(errors).toEqual([]));

    test('controls agree with data, including every time and resolution extreme', async ({ page }) => {
      const data = JSON.parse(await fs.readFile(`${artifact}.json`, 'utf8'));
      for (const time of [0, 60, 120]) {
        await page.locator('#time').fill(String(time));
        for (const resolution of ['32', '65536']) {
          await page.locator('#resolution').selectOption(resolution);
          await expect(page.locator('#tau')).toHaveText(data.tau[time].toExponential(2));
          await expect(page.locator('#speed')).toHaveText(data.scales.tangential_speed[time].toExponential(3));
          await expect(page.locator('#energy')).toHaveText(data.scales.core_energy_scale[time].toExponential(3));
          await expect(page.locator('#cells')).toHaveText((Math.sqrt(2 * data.tau[time]) * Number(resolution)).toPrecision(3));
        }
      }
      const picture = () => page.locator('#core').evaluate(canvas => canvas.toDataURL());
      const following = await picture();
      await page.locator('#follow').selectOption('world');
      expect(await picture()).not.toBe(following);
      await page.locator('#follow').selectOption('follow');
      expect(await picture()).toBe(following);
      await expect(page.locator('#measurements tr')).toHaveCount(data.convergence.filter(row => row.precision === 'float64').length);
      const downloadEvent = page.waitForEvent('download');
      await page.locator('#download').click();
      const download = await downloadEvent;
      expect(download.suggestedFilename()).toBe('observatory-data.json');
      expect(JSON.parse(await fs.readFile(await download.path(), 'utf8'))).toEqual(data);
    });

    test('animation pauses, reaches the end and restarts', async ({ page }) => {
      await page.clock.install({ time: new Date('2026-09-10T00:00:00Z') });
      await page.clock.pauseAt(new Date('2026-09-10T00:00:01Z'));
      await page.locator('#time').fill('118');
      await page.locator('#play').click();
      await page.clock.runFor(100);
      await expect(page.locator('#time')).toHaveValue('119');
      await page.locator('#play').click();
      await page.clock.runFor(500);
      await expect(page.locator('#time')).toHaveValue('119');
      await page.locator('#play').click();
      await page.clock.runFor(300);
      await expect(page.locator('#time')).toHaveValue('120');
      await expect(page.locator('#play')).toHaveText('Animate');
      await page.locator('#play').click();
      await expect(page.locator('#time')).toHaveValue('0');
      await page.clock.runFor(100);
      await expect(page.locator('#time')).toHaveValue('1');
      await page.locator('#play').click();
    });

    test('resizes without horizontal overflow or blank canvases', async ({ page }, testInfo) => {
      for (const width of [1440, 390, 768]) {
        await page.setViewportSize({ width, height: 900 });
        await expect.poll(() => page.locator('#core').evaluate(c => c.width)).toBeGreaterThan(0);
        expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
        for (const id of ['core', 'scales', 'convergence']) {
          const hasDrawing = await page.locator(`#${id}`).evaluate(c => {
            const pixels = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
            return pixels.some((value, index) => index % 4 === 3 && value > 0);
          });
          expect(hasDrawing).toBe(true);
        }
        await page.screenshot({ path: testInfo.outputPath(`viewport-${width}.png`), fullPage: true });
      }
    });

    test('data remains inert when a preview service executes all script elements', async ({ page }) => {
      // HTMLPreview recreates inline scripts without preserving their types.
      // Exercise that behavior in a fresh document without depending on its uptime.
      const markup = await fs.readFile(`${artifact}.html`, 'utf8');
      await page.goto('about:blank');
      await page.setContent(markup.replace(/<script\b/g, '<script type="text/preview"'));
      await page.evaluate(() => {
        for (const original of document.querySelectorAll('script')) {
          const executable = document.createElement('script');
          executable.textContent = original.textContent;
          document.body.appendChild(executable);
        }
      });
      await expect(page.locator('#tau')).not.toBeEmpty();
      await page.locator('#time').fill('120');
      await expect(page.locator('#tau')).toHaveText('1.00e-12');
    });
  });
}
