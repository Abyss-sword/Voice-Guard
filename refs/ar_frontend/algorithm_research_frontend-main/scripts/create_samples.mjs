import { mkdir, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { encodeWav, addAudioMarker } from '../src/services/algorithms.js';

const destination = new URL('../public/samples/', import.meta.url);
await mkdir(destination, { recursive: true });
const artwork = `<svg xmlns="http://www.w3.org/2000/svg" width="960" height="540" viewBox="0 0 960 540"><defs><linearGradient id="sky" x2="0" y2="1"><stop stop-color="#dbe9dc"/><stop offset="1" stop-color="#f1efda"/></linearGradient><linearGradient id="hill" x2="0" y2="1"><stop stop-color="#7b997d"/><stop offset="1" stop-color="#4f765f"/></linearGradient></defs><path fill="url(#sky)" d="M0 0h960v540H0z"/><circle cx="710" cy="130" r="65" fill="#f5f0d8"/><path fill="#adc3a2" d="M0 380 190 170 380 345 540 210 790 390 960 230v310H0z"/><path fill="url(#hill)" d="M0 440 260 285 500 430 690 315 960 425v115H0z"/><path fill="#c7d4b5" d="M0 475q320-85 540 5t420-5v65H0z"/><path d="M490 540q-60-90 70-165" stroke="#e3e8cd" stroke-width="25" fill="none"/><g fill="#4c7057"><path d="m103 395 22-80 22 80z"/><path d="m137 410 25-95 25 95z"/><path d="m806 405 20-90 20 90z"/></g><text x="36" y="493" fill="#3c5f45" font-family="sans-serif" font-size="12" letter-spacing="4">A QUIET LANDSCAPE / PLATFORM SAMPLE</text></svg>`;
await writeFile(new URL('image_original.svg', destination), artwork);
await writeFile(new URL('image_marked.svg', destination), artwork.replace('</svg>', '<text x="480" y="270" text-anchor="middle" dominant-baseline="middle" fill="white" opacity=".3" font-family="sans-serif" font-weight="600" font-size="32">ALGORITHM HUB</text></svg>'));
const sampleRate = 24000;
const original = Float32Array.from({ length: sampleRate * 4 }, (_, frame) => {
  const seconds = frame / sampleRate;
  return 0.18 * Math.sin(2 * Math.PI * (220 + Math.floor(seconds) * 55) * seconds) * Math.sin(Math.PI * (seconds % 1)) ** 2;
});
const marked = addAudioMarker([original], sampleRate, { strength: 0.06, frequency: 1200, interval: 1 });
await writeFile(new URL('audio_original.wav', destination), Buffer.from(encodeWav([original], sampleRate)));
await writeFile(new URL('audio_marked.wav', destination), Buffer.from(encodeWav(marked, sampleRate)));
console.log(`Created project-owned samples: ${fileURLToPath(destination)}`);
