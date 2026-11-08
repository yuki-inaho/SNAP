import { test, expect, Page } from '@playwright/test';
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

/**
 * End-to-end browser test for the interSeg3D-Studio frontend.
 *
 * It exercises the real UI: upload -> create object -> positive click ->
 * negative click -> inference -> mask summary -> mask visibility -> download.
 *
 * The backend must be running separately at the URL the frontend proxies to
 * (default http://127.0.0.1:9500). When MOCK_BACKEND=1 the assertions about
 * the mask shrinking after a negative click apply only to the deterministic
 * mock, never to PinPoint3D.
 *
 * The camera is placed through a DEV-only hook (window.__interSeg3D) so the
 * chair is unoccluded; all point picking still goes through the app's own
 * raycasting and UI events.
 */

const BASE_URL = process.env.BASE_URL || 'http://127.0.0.1:3001';
const SAMPLE_PLY =
  process.env.SAMPLE_PLY ||
  path.resolve(__dirname, '../../outputs/ply/scene0025_00.ply');
const MOCK = process.env.MOCK_BACKEND !== '0';
const CUBE_SIZE = process.env.CUBE_SIZE || (MOCK ? '0.05' : '0.02');
const ARTIFACTS = path.resolve(
  process.env.ARTIFACT_DIR || 'artifacts/run'
);

// Chair instance 13 in SNAP data_examples/ScanNet/scene0025_00.
const CHAIR_POINT: [number, number, number] = [3.1206, 1.2300, 0.8969]; // point index 74485
// Unlabeled clutter that is actually inside the positive chair mask. A floor
// click is a no-op for the real model (the mask never contains floor), so the
// real-backend shrink regression uses this proven coordinate (point 79693).
const CLUTTER_POINT: [number, number, number] = [
  2.9083800315856934,
  1.544044017791748,
  0.1839209944009781,
];
const CAMERA_POS: [number, number, number] = [1.53, 3.61, 3.81];
const CAMERA_TARGET: [number, number, number] = [2.8155, 1.3826, 0.75];
// From the default camera the chair seat occludes the clutter point and the
// click lands on the seat (z ~ 0.49). This low view down -X sees the clutter
// directly, so the background click selects the intended point.
const CLUTTER_CAMERA_POS: [number, number, number] = [4.308, 1.544, 0.184];

type Projected = { x: number; y: number; z: number };

type MetadataObject = {
  id: number;
  part_id?: number;
  mask_label?: number;
  label: string;
  color: number[];
  points?: number;
};

type MetadataClick = {
  obj_idx: number;
  part_idx?: number;
  obj_name?: string;
  is_positive: boolean;
  position: number[];
};

type Metadata = {
  objects: MetadataObject[];
  object_counts: Record<string, number>;
  mask_label_counts?: Record<string, number>;
  object_part_counts?: Array<{
    object_id: number;
    part_id: number;
    mask_label: number;
    points: number;
  }>;
  click_data: MetadataClick[];
  file_info: { ply_file: string; original_file: string };
};

type PlyColorSummary = {
  format: string;
  vertexCount: number;
  colorCounts: Map<string, number>;
};

const PLY_TYPE_SIZE: Record<string, number> = {
  char: 1,
  uchar: 1,
  short: 2,
  ushort: 2,
  int: 4,
  uint: 4,
  float: 4,
  double: 8,
};

function rgbKey(r: number, g: number, b: number): string {
  return `${r},${g},${b}`;
}

/** Parse the RGB histogram of a binary little-endian PLY export. */
function parsePlyColors(filePath: string): PlyColorSummary {
  const buf = fs.readFileSync(filePath);
  const marker = Buffer.from('end_header');
  const headerEnd = buf.indexOf(marker);
  if (headerEnd < 0) throw new Error(`PLY header end not found in ${filePath}`);
  const header = buf.subarray(0, headerEnd).toString('ascii');
  const lines = header
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  const format = (lines.find((line) => line.startsWith('format ')) || '')
    .split(/\s+/)[1];
  if (format !== 'binary_little_endian') {
    throw new Error(`Unsupported PLY format: ${format || 'unknown'}`);
  }

  let vertexCount = 0;
  let inVertex = false;
  const properties: Array<{ name: string; size: number }> = [];
  for (const line of lines) {
    const parts = line.split(/\s+/);
    if (parts[0] === 'element') {
      inVertex = parts[1] === 'vertex';
      if (inVertex) vertexCount = Number(parts[2]);
    } else if (parts[0] === 'property' && inVertex) {
      const size = PLY_TYPE_SIZE[parts[1]];
      if (!size) throw new Error(`Unsupported PLY property type: ${parts[1]}`);
      properties.push({ name: parts[parts.length - 1], size });
    }
  }
  if (!vertexCount) throw new Error(`No vertex element in ${filePath}`);

  let dataStart = headerEnd + marker.length;
  while (buf[dataStart] === 0x0d || buf[dataStart] === 0x0a) dataStart += 1;

  let stride = 0;
  const offsets: Record<string, number> = {};
  for (const property of properties) {
    offsets[property.name] = stride;
    stride += property.size;
  }
  for (const channel of ['red', 'green', 'blue']) {
    if (!(channel in offsets)) throw new Error(`PLY is missing ${channel} in ${filePath}`);
  }
  if (dataStart + vertexCount * stride > buf.length) {
    throw new Error('PLY vertex data is truncated');
  }

  const colorCounts = new Map<string, number>();
  for (let i = 0; i < vertexCount; i += 1) {
    const base = dataStart + i * stride;
    const key = rgbKey(
      buf[base + offsets.red],
      buf[base + offsets.green],
      buf[base + offsets.blue]
    );
    colorCounts.set(key, (colorCounts.get(key) ?? 0) + 1);
  }
  return { format, vertexCount, colorCounts };
}

/** Count points whose RGB matches `rgb` within `tolerance` per channel. */
function countColorWithin(
  counts: Map<string, number>,
  rgb: [number, number, number],
  tolerance: number
): number {
  let total = 0;
  for (const [key, count] of counts) {
    const [r, g, b] = key.split(',').map(Number);
    if (
      Math.abs(r - rgb[0]) <= tolerance &&
      Math.abs(g - rgb[1]) <= tolerance &&
      Math.abs(b - rgb[2]) <= tolerance
    ) {
      total += count;
    }
  }
  return total;
}

/** Read the "Labelled object points: 1,933" number from the mask card. */
function labelledCount(text: string): number {
  const match = /Labelled object points:\s*([\d,]+)/.exec(text);
  if (!match) throw new Error(`No labelled count in: ${text}`);
  return Number(match[1].replace(/,/g, ''));
}

async function closeInstructions(page: Page) {
  const close = page.getByRole('button', { name: 'Close' });
  if (await close.isVisible().catch(() => false)) {
    await close.click();
  }
}

async function placeCamera(
  page: Page,
  pos: [number, number, number] = CAMERA_POS,
  target: [number, number, number] = CAMERA_TARGET
) {
  await page.evaluate(
    ({ nextPos, nextTarget }) => {
      const hook = (window as any).__interSeg3D;
      if (!hook) throw new Error('DEV projection hook is unavailable');
      const camera = hook.getCamera();
      const controls = hook.getControls();
      camera.position.set(nextPos[0], nextPos[1], nextPos[2]);
      controls.target.set(nextTarget[0], nextTarget[1], nextTarget[2]);
      controls.update();
    },
    { nextPos: pos, nextTarget: target }
  );
  await page.waitForTimeout(600);
}

async function project(page: Page, point: [number, number, number]): Promise<Projected> {
  const projected = await page.evaluate(
    (p) => (window as any).__interSeg3D.project(p),
    point
  );
  if (!projected) throw new Error('projection failed');
  return projected as Projected;
}

async function clickWorldPoint(page: Page, point: [number, number, number]) {
  const p = await project(page, point);
  await page.mouse.click(p.x, p.y);
  await page.waitForTimeout(500);
}

async function selectedCoordinate(page: Page): Promise<number[]> {
  const text = await page.locator('.selection-info').innerText();
  const match = /Selected: \(([^)]+)\)/.exec(text);
  if (!match) throw new Error(`No selected coordinate in: ${text}`);
  return match[1].split(',').map(v => Number(v.trim()));
}

async function setCubeSize(page: Page, value: number) {
  const slider = page.getByTestId('cube-size-slider');
  const box = await slider.boundingBox();
  if (!box) throw new Error('cube size slider not visible');
  const fraction = (value - 0.005) / (0.1 - 0.005);
  // The Vuetify track has small horizontal padding; use the visible centre band.
  const x = box.x + 12 + fraction * (box.width - 24);
  await page.mouse.click(x, box.y + box.height / 2);
  await expect(page.locator('body')).toContainText(
    `Selection size: ${value.toFixed(3)} m`
  );
}

test('positive/negative point picking, mask and export', async ({ page }, testInfo) => {
  fs.mkdirSync(ARTIFACTS, { recursive: true });
  // Record every browser request so the test can prove that saving results did
  // not fall back to an external recognition service.
  const requestUrls: string[] = [];
  page.on('request', (request) => requestUrls.push(request.url()));
  await page.goto(BASE_URL);
  await closeInstructions(page);

  // 1. Upload the bundled ScanNet sample through the file input.
  await page.locator('input[type=file]').first().setInputFiles(SAMPLE_PLY);
  await expect
    .poll(async () => page.locator('.selection-info').innerText(), { timeout: 60_000 })
    .toMatch(/Points: [1-9][0-9,]*/);
  await page.screenshot({ path: path.join(ARTIFACTS, '01-uploaded.png') });

  // 2. Create the object that positive clicks belong to.
  const nameInput = page.getByTestId('new-object-name').locator('input');
  await nameInput.fill('chair');
  await page.getByTestId('create-object-btn').click();
  await expect(page.getByTestId('mode-object-btn')).toBeEnabled();

  await setCubeSize(page, Number(CUBE_SIZE));
  await placeCamera(page);
  await page.screenshot({ path: path.join(ARTIFACTS, '02-before-points.png') });

  // 3. Positive (object) click on a verified chair point.
  await page.getByTestId('mode-object-btn').click();
  await clickWorldPoint(page, CHAIR_POINT);
  await expect(page.getByTestId('positive-count')).toContainText('Positive: 1');
  const positiveCoord = await selectedCoordinate(page);
  expect(positiveCoord[0]).toBeGreaterThan(2.45);
  expect(positiveCoord[0]).toBeLessThan(3.25);
  expect(positiveCoord[2]).toBeGreaterThan(0.02);
  expect(positiveCoord[2]).toBeLessThan(1.05);
  await page.screenshot({ path: path.join(ARTIFACTS, '03-positive.png') });

  // 4. Inference with the positive-only prompt.
  await page.getByTestId('run-segmentation-btn').click();
  await expect(page.getByTestId('mask-card')).toBeVisible({ timeout: 60_000 });
  await expect(page.getByTestId('mask-labelled')).not.toContainText('0');
  const maskBefore = await page.getByTestId('mask-labelled').innerText();
  await page.screenshot({ path: path.join(ARTIFACTS, '04-mask-positive.png') });
  testInfo.annotations.push({ type: 'mask-before-negative', description: maskBefore });

  // 5. Negative (background) click on clutter that sits inside the chair mask.
  await page.getByTestId('mode-background-btn').click();
  await placeCamera(page, CLUTTER_CAMERA_POS, CLUTTER_POINT);
  await clickWorldPoint(page, CLUTTER_POINT);
  await expect(page.getByTestId('negative-count')).toContainText('Negative: 1');
  const negativeCoord = await selectedCoordinate(page);
  expect(negativeCoord[0]).toBeGreaterThan(2.7);
  expect(negativeCoord[2]).toBeLessThan(0.25);
  const clutterMiss = Math.hypot(
    negativeCoord[0] - CLUTTER_POINT[0],
    negativeCoord[1] - CLUTTER_POINT[1],
    negativeCoord[2] - CLUTTER_POINT[2]
  );
  expect(clutterMiss).toBeLessThan(0.05);
  await page.screenshot({ path: path.join(ARTIFACTS, '05-negative.png') });
  // Restore the overview camera for the remaining screenshots and interactions.
  await placeCamera(page);

  // 6. Inference with positive + negative prompts. The mask card is already
  // visible from the first inference, so wait until the second result replaces
  // it instead of reading the stale positive-only count.
  await page.getByTestId('run-segmentation-btn').click();
  await expect(page.getByTestId('mask-card')).toBeVisible();
  await expect
    .poll(async () => page.getByTestId('mask-labelled').innerText(), { timeout: 120_000 })
    .not.toBe(maskBefore);
  const maskAfter = await page.getByTestId('mask-labelled').innerText();
  await page.screenshot({ path: path.join(ARTIFACTS, '06-mask-positive-negative.png') });
  testInfo.annotations.push({ type: 'mask-after-negative', description: maskAfter });

  // The clutter negative is inside the mask, so both the real model and the
  // deterministic mock must remove points. (A floor negative is a no-op for
  // the real model, which is why it is not used for this regression.)
  const labelledBefore = labelledCount(maskBefore);
  const labelledAfter = labelledCount(maskAfter);
  expect(labelledAfter).toBeGreaterThan(0);
  expect(labelledAfter).toBeLessThan(labelledBefore);
  testInfo.annotations.push({
    type: 'mask-shrink',
    description: `${labelledBefore} -> ${labelledAfter} (${labelledBefore - labelledAfter} points removed)`,
  });

  // 7. Mask visibility toggle.
  const maskSwitch = page.getByTestId('mask-colors-switch');
  await maskSwitch.click();
  await expect(maskSwitch.locator('input')).not.toBeChecked();
  await page.screenshot({ path: path.join(ARTIFACTS, '07-mask-hidden.png') });
  await maskSwitch.click();
  await expect(maskSwitch.locator('input')).toBeChecked();
  await page.screenshot({ path: path.join(ARTIFACTS, '08-mask-shown.png') });

  // 9. Select mode opens the object editor from a segmented point.
  await page.getByRole('button', { name: 'Select' }).click();
  const maskPixel = await project(page, CHAIR_POINT);
  await page.mouse.click(maskPixel.x, maskPixel.y);
  await expect(page.locator('.object-edit-panel')).toBeVisible({ timeout: 15_000 });
  await expect(page.locator('.object-edit-panel')).toContainText('Edit Object');
  await page.screenshot({ path: path.join(ARTIFACTS, '09-select-object.png') });
  await page.locator('.object-edit-panel').getByRole('button', { name: 'CANCEL' }).click();

  // 10. Save/download results.
  const downloadPromise = page.waitForEvent('download', { timeout: 60_000 });
  await page.getByTestId('save-results-btn').click();
  const download = await downloadPromise;
  const zipPath = path.join(ARTIFACTS, download.suggestedFilename());
  await download.saveAs(zipPath);
  const size = fs.statSync(zipPath).size;
  expect(size).toBeGreaterThan(1000);
  testInfo.annotations.push({ type: 'download', description: `${zipPath} (${size} bytes)` });

  // No recognition endpoint and no non-local host may be contacted. The Save
  // Results path must stay local: update_objects only relabels the session.
  expect(
    requestUrls.filter((url) => url.includes('/recognition/mask_obj_recognition'))
  ).toEqual([]);
  const external = requestUrls.filter(
    (url) => /^https?:/i.test(url) && !/^https?:\/\/(127\.0\.0\.1|localhost)(:\d+)?\//i.test(url)
  );
  expect(external).toEqual([]);

  if (!MOCK) {
    // Regression assertions on the real backend's Save Results package.
    const extractDir = path.join(ARTIFACTS, 'extracted');
    fs.rmSync(extractDir, { recursive: true, force: true });
    fs.mkdirSync(extractDir, { recursive: true });
    execFileSync('unzip', ['-o', '-q', zipPath, '-d', extractDir]);

    const metadataPath = path.join(extractDir, 'metadata.json');
    const metadata = JSON.parse(fs.readFileSync(metadataPath, 'utf8')) as Metadata;

    expect(metadata.file_info.ply_file).toBe('scene_with_colored_objects.ply');
    expect(metadata.file_info.original_file).toBe(path.basename(SAMPLE_PLY));

    // Object ids must be the UI ids (1), not the raw mask label (101).
    expect(metadata.objects).toHaveLength(1);
    expect(Object.keys(metadata.object_counts)).toEqual(['1']);
    const chairCount = metadata.object_counts['1'];
    expect(chairCount).toBeGreaterThan(0);
    expect(metadata.objects[0]).toMatchObject({
      id: 1,
      part_id: 1,
      mask_label: 101,
      label: 'chair',
      points: chairCount,
    });

    // The raw model label is preserved for callers that need it.
    expect(metadata.mask_label_counts).toEqual({ '101': chairCount });
    expect(metadata.object_part_counts).toEqual([
      { object_id: 1, part_id: 1, mask_label: 101, points: chairCount },
    ]);

    // UI count, export count and raw mask label all agree.
    expect(labelledAfter).toBe(chairCount);

    // Click metadata: background click is negative, object click is positive.
    expect(metadata.click_data).toHaveLength(2);
    const backgroundClick = metadata.click_data.find((click) => click.obj_idx === 0);
    const objectClick = metadata.click_data.find((click) => click.obj_idx === 1);
    expect(backgroundClick, 'background click in metadata').toBeTruthy();
    expect(backgroundClick!.is_positive).toBe(false);
    expect(backgroundClick!.part_idx).toBe(0);
    expect(backgroundClick!.position[0]).toBeGreaterThan(2.7);
    expect(backgroundClick!.position[2]).toBeLessThan(0.25);
    const exportedClutterMiss = Math.hypot(
      backgroundClick!.position[0] - CLUTTER_POINT[0],
      backgroundClick!.position[1] - CLUTTER_POINT[1],
      backgroundClick!.position[2] - CLUTTER_POINT[2]
    );
    expect(exportedClutterMiss).toBeLessThan(0.05);
    expect(objectClick, 'object click in metadata').toBeTruthy();
    expect(objectClick!.is_positive).toBe(true);
    expect(objectClick!.part_idx).toBe(1);
    expect(objectClick!.position[0]).toBeGreaterThan(2.45);
    expect(objectClick!.position[0]).toBeLessThan(3.25);

    // The exported PLY must colour exactly the object's points with the colour
    // recorded in metadata (old exports recorded the label-1 colour but painted
    // label 101). Background must stay the uniform 0.7 grey.
    const ply = parsePlyColors(path.join(extractDir, metadata.file_info.ply_file));
    expect(ply.vertexCount).toBeGreaterThan(100_000);
    const sum = [...ply.colorCounts.values()].reduce((acc, count) => acc + count, 0);
    expect(sum).toBe(ply.vertexCount);

    const expectedRgb = metadata.objects[0].color.map((channel) =>
      Math.round(channel * 255)
    ) as [number, number, number];
    expect(countColorWithin(ply.colorCounts, expectedRgb, 1)).toBe(chairCount);

    const modal = [...ply.colorCounts.entries()].sort((a, b) => b[1] - a[1])[0];
    const [greyR, greyG, greyB] = modal[0].split(',').map(Number);
    expect(modal[0]).not.toBe(rgbKey(...expectedRgb));
    expect(greyR).toBe(greyG);
    expect(greyG).toBe(greyB);
    expect(modal[1]).toBe(ply.vertexCount - chairCount);

    testInfo.annotations.push({
      type: 'export-metadata',
      description: `object id 1 / mask label 101 / ${chairCount} points / colour ${expectedRgb.join(',')}`,
    });
  }

  await page.screenshot({ path: path.join(ARTIFACTS, '10-final.png') });
});
