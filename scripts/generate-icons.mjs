import { mkdir, writeFile } from 'node:fs/promises';
import { crc32, deflateSync } from 'node:zlib';

// Pixel-aligned geometry keeps the small toolbar icon crisp. This is the
// standalone brand symbol; its wider stroke stays visible at 16px.
const root = new URL('../', import.meta.url);
const rect = { x: 5, y: 1, width: 6, height: 14 };
const colors = { on: [255, 23, 68], off: [139, 144, 152] };

function chunk(type, data) {
  const tag = Buffer.from(type);
  const length = Buffer.alloc(4);
  length.writeUInt32BE(data.length);
  const checksum = Buffer.alloc(4);
  checksum.writeUInt32BE(crc32(Buffer.concat([tag, data])));
  return Buffer.concat([length, tag, data, checksum]);
}

function png(size, color) {
  const scale = size / 16;
  const stride = 1 + size * 4;
  const pixels = Buffer.alloc(size * stride); // RGBA with filter byte 0 per row.
  for (let y = rect.y * scale; y < (rect.y + rect.height) * scale; y += 1) {
    for (let x = rect.x * scale; x < (rect.x + rect.width) * scale; x += 1) {
      pixels.set([...color, 255], y * stride + 1 + x * 4);
    }
  }
  const header = Buffer.alloc(13);
  header.writeUInt32BE(size, 0);
  header.writeUInt32BE(size, 4);
  header[8] = 8; // Bit depth.
  header[9] = 6; // RGBA.
  return Buffer.concat([
    Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]),
    chunk('IHDR', header),
    chunk('IDAT', deflateSync(pixels)),
    chunk('IEND', Buffer.alloc(0)),
  ]);
}

await mkdir(new URL('icons/', root), { recursive: true });
const red = `#${colors.on.map((value) => value.toString(16).padStart(2, '0')).join('')}`;
await writeFile(new URL('icons/divider.svg', root),
  `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" width="128" height="128">\n` +
  `  <rect x="${rect.x}" y="${rect.y}" width="${rect.width}" height="${rect.height}" fill="${red}"/>\n</svg>\n`);
for (const [state, sizes] of [['on', [16, 32, 48, 128]], ['off', [16, 32]]]) {
  for (const size of sizes) {
    await writeFile(new URL(`icons/${state}-${size}.png`, root), png(size, colors[state]));
  }
}
await writeFile(new URL('site/static/favicon.png', root), png(32, colors.on));
console.log('Generated divider SVG, six extension icons and the site favicon.');
