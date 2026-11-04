import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { validateManifest, validateParameters, validateFile, normalizeParameters, safeUrl, defaultsFor } from '../src/lib/contract.js';
import { runAlgorithm, encodeWav, addAudioMarker } from '../src/services/algorithms.js';

const readJson = async path => JSON.parse(await readFile(new URL(path, import.meta.url), 'utf8'));
const image = await readJson('../public/methods/image_watermark.json');
const audio = await readJson('../public/methods/audio_marker.json');
const video = await readJson('../public/templates/video_defense.json');

test('内置方法、视频模板及参数默认值符合统一契约', () => {
  for (const method of [image, audio, video]) {
    assert.equal(validateManifest(method), method);
    assert.deepEqual(validateParameters(method, defaultsFor(method)), {});
  }
});

test('接入配置拒绝缺失样例、字段冲突、不安全地址和无效默认值', () => {
  for (const change of [method => method.examples = [], method => method.parameters[0].name = method.inputs[0].name, method => method.examples[0].output.src = 'javascript:alert(1)', method => method.parameters[1].default = 10]) {
    const method = structuredClone(image); change(method);
    assert.throws(() => validateManifest(method));
  }
  assert.throws(() => safeUrl('file:///tmp/result.mp4'));
  assert.throws(() => safeUrl('https://name:secret@example.com/result'));
});

test('用户参数保留类型，并拒绝空白、越界和非法选项', () => {
  const values = { ...defaultsFor(image), opacity: '0.4', font_size: '32', repeat: false };
  assert.deepEqual(validateParameters(image, values), {});
  assert.equal(normalizeParameters(image, values).opacity, 0.4);
  assert.equal(normalizeParameters(image, values).repeat, false);
  assert.ok(validateParameters(image, { ...values, text: ' ', font_size: '12.5', opacity: ' ', position: 'wrong' }).text);
  assert.equal(Object.keys(validateParameters(image, { ...values, text: '', font_size: '12.5', opacity: 3, position: 'wrong' })).length, 4);
});

test('上传类型、大小与可选素材独立校验', () => {
  assert.match(validateFile(video.inputs[0], new File(['test'], 'wrong.wav', { type: 'audio/wav' })), /格式/);
  assert.match(validateFile(image.inputs[0], null), /上传/);
  assert.equal(validateFile({ ...image.inputs[0], required: false }, null), '');
  assert.match(validateFile({ ...image.inputs[0], max_mb: 0.000001 }, new File(['test'], 'image.png', { type: 'image/png' })), /超过/);
});

test('音频标记修改采样并导出合法双声道 PCM WAV，原始输入保持不变', () => {
  const input = new Float32Array(2400);
  const marked = addAudioMarker([input, input], 24000, { strength: 0.1, frequency: 1200, interval: 1 });
  assert.ok(marked[0].some(sample => sample !== 0));
  assert.ok(input.every(sample => sample === 0));
  const wav = encodeWav(marked, 24000);
  const header = new DataView(wav);
  assert.equal(header.getUint16(22, true), 2);
  assert.equal(header.getUint32(24, true), 24000);
  assert.equal(header.getUint32(40, true), 2400 * 2 * 2);
  assert.equal(Buffer.from(wav).subarray(0, 4).toString(), 'RIFF');
});

test('外部算法收到实际文件和数值参数，页面使用实际返回文件', async () => {
  const originalFetch = globalThis.fetch;
  const file = new File(['video_data'], 'my_clip.mp4', { type: 'video/mp4' });
  try {
    globalThis.fetch = async (url, options) => {
      assert.equal(url, video.adapter.endpoint);
      assert.equal(await options.body.get('input_file').text(), 'video_data');
      assert.equal(options.body.get('input_file').name, 'my_clip.mp4');
      assert.deepEqual(JSON.parse(options.body.get('parameters')), { strength: 0.1, seed: 42 });
      assert.equal(options.credentials, 'omit');
      return Response.json({ artifacts: [{ kind: 'video', url: '/files/new_result.mp4', filename: 'new_result.mp4' }] });
    };
    const result = await runAlgorithm(video, { input_file: file }, { strength: '0.1', seed: '42' }, new AbortController().signal);
    assert.equal(result.artifacts[0].url, 'https://your_algorithm.example/files/new_result.mp4');
    globalThis.fetch = async () => Response.json({ artifacts: [{ kind: 'video', url: 'file:///tmp/private.mp4' }] });
    await assert.rejects(runAlgorithm(video, { input_file: file }, defaultsFor(video), new AbortController().signal), /HTTP/);
    globalThis.fetch = async () => Response.json({ error: '输入解码失败' }, { status: 422 });
    await assert.rejects(runAlgorithm(video, { input_file: file }, defaultsFor(video), new AbortController().signal), /输入解码失败/);
  } finally { globalThis.fetch = originalFetch; }
});
