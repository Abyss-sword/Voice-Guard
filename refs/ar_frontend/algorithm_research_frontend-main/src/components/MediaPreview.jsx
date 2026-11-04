import { useEffect, useState } from 'react';
import { ImageIcon, AudioLines, Video, FileText, AlertCircle } from 'lucide-react';

export const mediaIcons = { image: ImageIcon, audio: AudioLines, video: Video, json: FileText, text: FileText };

export function useFileUrl(file) {
  const [url, setUrl] = useState('');
  useEffect(() => {
    if (!file) { setUrl(''); return; }
    const next = URL.createObjectURL(file);
    setUrl(next);
    return () => URL.revokeObjectURL(next);
  }, [file]);
  return url;
}

export function MediaPreview({ src, kind, label = '素材预览', compact = false }) {
  const [failed, setFailed] = useState(false);
  useEffect(() => setFailed(false), [src]);
  const Icon = mediaIcons[kind] || FileText;
  if (!src || failed) return <div className="media_empty"><AlertCircle size={24} /><strong>{failed ? '暂时无法预览素材' : '还没有素材'}</strong><span>{failed ? '请检查文件地址、格式和接口访问权限。' : '上传后将在这里预览'}</span></div>;
  return <div className={`media_preview ${kind} ${compact ? 'compact' : ''}`}>
    {kind === 'image' && <img src={src} alt={label} onError={() => setFailed(true)} />}
    {kind === 'video' && <video src={src} controls preload="metadata" playsInline aria-label={label} onError={() => setFailed(true)} />}
    {kind === 'audio' && <><div className="audio_visual"><AudioLines size={42} /><div><strong>{label}</strong><span>试听声音，观察处理效果</span></div><div className="wave_bars">{Array.from({ length: 21 }, (_, index) => <i key={index} style={{ height: `${12 + (index * 17 % 45)}px` }} />)}</div></div><audio src={src} controls preload="metadata" aria-label={label} onError={() => setFailed(true)} /></>}
    {['json', 'text'].includes(kind) && <div className="document_preview"><Icon size={32} /><strong>{label}</strong><a href={src} target="_blank" rel="noreferrer">打开结果文件 ↗</a></div>}
  </div>;
}
