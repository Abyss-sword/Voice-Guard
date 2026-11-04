import { normalizeParameters, safeUrl } from '../lib/contract.js';

export function encodeWav(channels, sampleRate) {
  const length = channels[0].length;
  const buffer = new ArrayBuffer(44 + length * channels.length * 2);
  const view = new DataView(buffer);
  const writeText = (offset, text) => [...text].forEach((letter, index) => view.setUint8(offset + index, letter.charCodeAt(0)));
  writeText(0, 'RIFF');
  view.setUint32(4, buffer.byteLength - 8, true);
  writeText(8, 'WAVE');
  writeText(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, channels.length, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * channels.length * 2, true);
  view.setUint16(32, channels.length * 2, true);
  view.setUint16(34, 16, true);
  writeText(36, 'data');
  view.setUint32(40, length * channels.length * 2, true);
  for (let frame = 0; frame < length; frame++) {
    for (let channel = 0; channel < channels.length; channel++) {
      const sample = Math.max(-1, Math.min(1, channels[channel][frame]));
      view.setInt16(44 + (frame * channels.length + channel) * 2, sample * (sample < 0 ? 32768 : 32767), true);
    }
  }
  return buffer;
}

export function addAudioMarker(channels, sampleRate, parameters) {
  return channels.map(channel => Float32Array.from(channel, (sample, frame) => {
    const seconds = frame / sampleRate;
    const phase = seconds % parameters.interval;
    const envelope = phase < 0.12 ? Math.sin(Math.PI * phase / 0.12) ** 2 : 0;
    return Math.max(-1, Math.min(1, sample + parameters.strength * envelope * Math.sin(2 * Math.PI * parameters.frequency * seconds)));
  }));
}

async function watermarkImage(file, parameters, signal) {
  const image = await createImageBitmap(file);
  try {
    signal.throwIfAborted();
    if (image.width * image.height > 24000000) throw new Error('图像过大，请缩小至 2400 万像素以内。');
    const canvas = document.createElement('canvas');
    canvas.width = image.width;
    canvas.height = image.height;
    const context = canvas.getContext('2d');
    context.drawImage(image, 0, 0);
    context.globalAlpha = parameters.opacity;
    context.fillStyle = '#ffffff';
    context.shadowColor = '#102b23';
    context.shadowBlur = 4;
    context.font = `600 ${parameters.font_size}px sans-serif`;
    context.textBaseline = 'middle';
    const width = context.measureText(parameters.text).width;
    if (parameters.repeat) {
      context.textAlign = 'center';
      for (let top = parameters.font_size; top < canvas.height; top += parameters.font_size * 3) {
        for (let left = width / 2 + 16; left < canvas.width + width; left += width + 60) context.fillText(parameters.text, left, top);
      }
    } else {
      context.textAlign = parameters.position === 'bottom_right' ? 'right' : 'center';
      context.fillText(parameters.text, parameters.position === 'bottom_right' ? canvas.width - 24 : canvas.width / 2, parameters.position === 'bottom_right' ? canvas.height - parameters.font_size : canvas.height / 2);
    }
    const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/png'));
    if (!blob) throw new Error('图像导出失败。');
    return { artifacts: [{ blob, kind: 'image', label: '添加水印后的图像', filename: 'watermarked.png' }], message: '已在当前浏览器中完成可见水印处理。' };
  } finally { image.close(); }
}

async function markAudio(file, parameters, signal) {
  const context = new AudioContext();
  try {
    const audio = await context.decodeAudioData(await file.arrayBuffer());
    signal.throwIfAborted();
    if (audio.duration > 180) throw new Error('浏览器音频样例最多处理 3 分钟音频，请截取片段后重试。');
    const channels = Array.from({ length: audio.numberOfChannels }, (_, index) => audio.getChannelData(index));
    const marked = addAudioMarker(channels, audio.sampleRate, parameters);
    return { artifacts: [{ blob: new Blob([encodeWav(marked, audio.sampleRate)], { type: 'audio/wav' }), kind: 'audio', label: '加入标记后的音频', filename: 'marked_audio.wav' }], message: '已加入周期提示音。此样例不提供不可感知水印能力。' };
  } catch (error) {
    if (error.name === 'EncodingError') throw new Error('浏览器无法解码此音频，请转换为 WAV 或 MP3 后重试。');
    throw error;
  } finally { await context.close(); }
}

export async function runAlgorithm(method, files, values, signal) {
  const parameters = normalizeParameters(method, values);
  if (method.adapter.type === 'browser') {
    if (method.adapter.handler === 'image_watermark') return watermarkImage(files.image, parameters, signal);
    return markAudio(files.audio, parameters, signal);
  }
  const body = new FormData();
  Object.entries(files).forEach(([name, file]) => { if (file) body.append(name, file, file.name); });
  body.append('method_id', method.id);
  body.append('parameters', JSON.stringify(parameters));
  let response;
  try { response = await fetch(method.adapter.endpoint, { method: 'POST', body, signal, credentials: 'omit' }); }
  catch (error) {
    if (error.name === 'AbortError') throw error;
    throw new Error('无法连接算法接口，请确认服务已启动、允许跨域，且接口协议与网页一致。');
  }
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(typeof detail.error === 'string' ? detail.error.slice(0, 500) : `算法服务返回 HTTP ${response.status}。`);
  }
  const result = await response.json();
  if (!Array.isArray(result.artifacts) || !result.artifacts.length || result.artifacts.length > 20) throw new Error('接口需要返回 1–20 个 artifacts 结果文件。');
  const artifacts = result.artifacts.map(artifact => {
    if (!['image', 'audio', 'video', 'json', 'text'].includes(artifact.kind)) throw new Error('接口返回了不支持的结果类型。');
    return { ...artifact, url: safeUrl(artifact.url, method.adapter.endpoint), filename: typeof artifact.filename === 'string' ? artifact.filename.replace(/[\\/]/g, '_') : 'result', label: typeof artifact.label === 'string' ? artifact.label : '处理结果' };
  });
  return { artifacts, message: typeof result.message === 'string' ? result.message : '算法处理完成。' };
}
