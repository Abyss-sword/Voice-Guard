import { validateManifest } from './contract.js';

export const assetUrl = path => new URL(path, new URL(import.meta.env.BASE_URL, location.href)).href;
const storageKey = 'algorithm_frontend_methods_v1';

export function readLocalMethods() {
  try { return JSON.parse(localStorage.getItem(storageKey) || '[]').map(validateManifest); }
  catch { return []; }
}

export function saveLocalMethod(method, builtins) {
  validateManifest(method);
  if (builtins.some(item => item.id === method.id)) throw new Error('这个 ID 已被仓库方法使用，请换一个 ID。');
  const current = readLocalMethods();
  if (current.some(item => item.id === method.id)) throw new Error('这个 ID 已在浏览器登记，请先移除旧登记或更换 ID。');
  localStorage.setItem(storageKey, JSON.stringify([...current, method]));
}

export function removeLocalMethod(id) {
  localStorage.setItem(storageKey, JSON.stringify(readLocalMethods().filter(method => method.id !== id)));
}

export async function loadCatalog(signal) {
  const response = await fetch(assetUrl('methods/index.json'), { signal });
  if (!response.ok) throw new Error('方法目录加载失败，请检查本地服务或站点文件。');
  const filenames = await response.json();
  const settled = await Promise.allSettled(filenames.map(async filename => {
    const result = await fetch(assetUrl(`methods/${filename}`), { signal });
    if (!result.ok) throw new Error(`${filename} 加载失败。`);
    return validateManifest(await result.json());
  }));
  const methods = settled.filter(item => item.status === 'fulfilled').map(item => item.value);
  const warnings = settled.filter(item => item.status === 'rejected').map(item => item.reason.message);
  return { methods, warnings };
}

export function downloadFile(blob, name) {
  const href = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = href;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(href), 1000);
}

export function downloadJson(data, name) {
  downloadFile(new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }), name);
}
