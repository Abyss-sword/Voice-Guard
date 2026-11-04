export const identifier = /^[a-z][a-z0-9_]{1,63}$/;
export const kinds = ['image', 'audio', 'video'];
export const parameterTypes = ['text', 'textarea', 'number', 'integer', 'select', 'boolean'];
export const kindLabels = { image: '图像', audio: '音频', video: '视频' };

export function safeUrl(value, base = globalThis.location?.href || 'http://localhost/') {
  if (typeof value !== 'string' || !value.trim()) throw new Error('素材或接口地址不能为空。');
  const parsed = new URL(value, base);
  if (!['http:', 'https:'].includes(parsed.protocol)) throw new Error('地址必须是相对路径或 HTTP(S) 地址。');
  if (parsed.username || parsed.password) throw new Error('地址中不能包含账号密码。');
  return parsed.href;
}

export function validateManifest(method) {
  if (!method || typeof method !== 'object' || Array.isArray(method)) throw new Error('方法配置必须是 JSON 对象。');
  if (method.schema_version !== '1.0') throw new Error('schema_version 应为 1.0。');
  if (!identifier.test(method.id)) throw new Error('方法 ID 请使用小写字母、数字和下划线，长度 2–64。');
  for (const field of ['title', 'summary', 'description', 'team', 'version']) {
    if (typeof method[field] !== 'string' || !method[field].trim()) throw new Error(`请填写 ${field}。`);
  }
  if (!['内容水印', '主动防御'].includes(method.category)) throw new Error('请选择内容水印或主动防御分类。');
  if (!Array.isArray(method.inputs) || !method.inputs.length || method.inputs.length > 8) throw new Error('每个方法需要 1–8 个素材输入。');
  const names = new Set();
  for (const input of method.inputs) {
    if (!identifier.test(input.name) || names.has(input.name)) throw new Error('输入字段名无效或重复。');
    names.add(input.name);
    if (!kinds.includes(input.kind) || !input.label?.trim()) throw new Error('请填写素材名称与类型。');
    if (!Array.isArray(input.accept) || !input.accept.length || input.accept.some(mime => typeof mime !== 'string' || !mime.startsWith(`${input.kind}/`))) throw new Error('素材 MIME 类型必须与素材类型一致。');
    if (!Number.isFinite(input.max_mb) || input.max_mb <= 0 || input.max_mb > 500) throw new Error('单文件大小限制应在 0–500 MB 之间。');
  }
  if (!Array.isArray(method.parameters) || method.parameters.length > 40) throw new Error('参数应为数组，最多 40 项。');
  for (const parameter of method.parameters) {
    if (!identifier.test(parameter.name) || names.has(parameter.name)) throw new Error('参数字段名无效或与其他字段重复。');
    names.add(parameter.name);
    if (!parameterTypes.includes(parameter.type) || !parameter.label?.trim()) throw new Error('参数类型或名称无效。');
    if (parameter.type === 'select' && (!Array.isArray(parameter.options) || !parameter.options.length || parameter.options.some(option => typeof option.value !== 'string' || !option.label))) throw new Error('选择参数需要至少一个有效选项。');
    if (['number', 'integer'].includes(parameter.type)) {
      if (parameter.min != null && !Number.isFinite(parameter.min) || parameter.max != null && !Number.isFinite(parameter.max)) throw new Error('参数上下限必须为数字。');
      if (parameter.min != null && parameter.max != null && parameter.min > parameter.max) throw new Error('参数下限不能高于上限。');
      if (parameter.step != null && (!Number.isFinite(parameter.step) || parameter.step <= 0)) throw new Error('参数步长应为正数。');
    }
  }
  const defaultValues = Object.fromEntries(method.parameters.filter(parameter => parameter.default !== undefined).map(parameter => [parameter.name, parameter.default]));
  const defaultErrors = validateParameters({ parameters: method.parameters.filter(parameter => parameter.default !== undefined) }, defaultValues);
  if (Object.keys(defaultErrors).length) throw new Error(`参数默认值不合法：${Object.keys(defaultErrors).join('、')}。`);
  if (!Array.isArray(method.examples) || !method.examples.length || method.examples.length > 12) throw new Error('每个方法必须提供 1–12 组样例。');
  for (const example of method.examples) {
    if (!example.title?.trim() || !example.description?.trim()) throw new Error('样例需要标题与效果说明。');
    for (const item of [example.input, example.output]) {
      if (!item || !kinds.includes(item.kind)) throw new Error('样例需要输入、输出素材及其类型。');
      safeUrl(item.src);
    }
  }
  if (method.adapter?.type === 'http') {
    if (!/^https?:\/\//.test(method.adapter.endpoint)) throw new Error('算法接口需要完整的 HTTP(S) 地址。');
    safeUrl(method.adapter.endpoint);
  } else if (method.adapter?.type !== 'browser' || !['image_watermark', 'audio_marker'].includes(method.adapter.handler)) {
    throw new Error('请选择 HTTP 接口或平台内置的浏览器示例。');
  }
  return method;
}

export function defaultsFor(method) {
  return Object.fromEntries(method.parameters.map(parameter => [parameter.name, parameter.default ?? (parameter.type === 'boolean' ? false : '')]));
}

export function validateParameters(method, values) {
  const errors = {};
  for (const parameter of method.parameters) {
    const value = values[parameter.name];
    if (value === '' || value == null) {
      if (parameter.required) errors[parameter.name] = `请填写${parameter.label}。`;
      continue;
    }
    if (['number', 'integer'].includes(parameter.type)) {
      const numeric = Number(value);
      if (typeof value === 'boolean' || typeof value === 'object' || typeof value === 'string' && !value.trim() || !Number.isFinite(numeric) || parameter.type === 'integer' && !Number.isInteger(numeric)) errors[parameter.name] = '请输入有效数字（整数参数不支持小数）。';
      else if (parameter.min != null && numeric < parameter.min || parameter.max != null && numeric > parameter.max) errors[parameter.name] = `范围：${parameter.min ?? '不限'} 至 ${parameter.max ?? '不限'}。`;
    } else if (parameter.type === 'select' && !parameter.options.some(option => option.value === value)) errors[parameter.name] = '请选择有效选项。';
    else if (parameter.type === 'boolean' && typeof value !== 'boolean') errors[parameter.name] = '此参数必须为布尔值。';
    else if (['text', 'textarea'].includes(parameter.type) && (typeof value !== 'string' || value.length > (parameter.max_length ?? 2000) || parameter.required && !value.trim())) errors[parameter.name] = '文字为空或超过长度限制。';
  }
  return errors;
}

export function normalizeParameters(method, values) {
  return Object.fromEntries(method.parameters.filter(parameter => values[parameter.name] !== '' && values[parameter.name] != null).map(parameter => [parameter.name, ['number', 'integer'].includes(parameter.type) ? Number(values[parameter.name]) : values[parameter.name]]));
}

export function validateFile(input, file) {
  if (!file) return input.required ? `请上传${input.label}。` : '';
  if (file.size > input.max_mb * 1024 * 1024) return `文件不能超过 ${input.max_mb} MB。`;
  if (!input.accept.some(mime => mime.endsWith('/*') ? file.type.startsWith(mime.slice(0, -1)) : mime === file.type)) return `请选择支持的${kindLabels[input.kind]}格式。`;
  return '';
}
