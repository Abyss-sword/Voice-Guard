import { useEffect, useRef, useState } from 'react';
import { Upload, ArrowUpRight, CheckCircle2, RotateCcw, Download, SlidersHorizontal, Loader2, X, Play, ShieldCheck } from 'lucide-react';
import { MediaPreview, mediaIcons, useFileUrl } from './MediaPreview.jsx';
import ParameterField from './ParameterField.jsx';
import { defaultsFor, kindLabels, validateFile, validateParameters, normalizeParameters } from '../lib/contract.js';
import { assetUrl, downloadFile, downloadJson } from '../lib/catalog.js';
import { runAlgorithm } from '../services/algorithms.js';

function FileInput({ input, file, onChange, disabled, error }) {
  const url = useFileUrl(file);
  const Icon = mediaIcons[input.kind];
  return <div className="file_field">
    <div className="field_heading"><label htmlFor={`file_${input.name}`}>{input.label}{input.required && <em> *</em>}</label><span>{kindLabels[input.kind]} · 最大 {input.max_mb} MB</span></div>
    {file && <MediaPreview src={url} kind={input.kind} label={file.name} />}
    <label className={`upload_area ${file ? 'has_file' : ''} ${error ? 'invalid' : ''}`}>
      <input id={`file_${input.name}`} type="file" accept={input.accept.join(',')} disabled={disabled} onChange={event => { const selected = event.target.files[0]; if (selected) onChange(selected); event.target.value = ''; }} />
      {file ? <CheckCircle2 size={24} /> : <Upload size={28} />}<div><strong>{file ? file.name : `点击选择${kindLabels[input.kind]}文件`}</strong><span>{file ? '点击重新选择文件' : `${input.accept.join(' / ')} · 文件仅在点击运行后交给算法`}</span></div><ArrowUpRight size={20} />
    </label>
    {file && <button type="button" className="text_button" disabled={disabled} onClick={() => onChange(null)}><X size={14} />移除此素材</button>}
    {error && <p className="field_error" role="alert">{error}</p>}
  </div>;
}

function Artifact({ artifact }) {
  const localUrl = useFileUrl(artifact.blob);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const url = localUrl || artifact.url;
  async function download() {
    setError(''); setBusy(true);
    try {
      const blob = artifact.blob || await fetch(url, { credentials: 'omit' }).then(response => { if (!response.ok) throw new Error(); return response.blob(); });
      downloadFile(blob, artifact.filename);
    } catch { setError('下载失败，请检查结果文件地址与跨域设置。'); }
    finally { setBusy(false); }
  }
  return <article className="artifact"><div className="field_heading"><strong>{artifact.label}</strong><span>{artifact.filename}</span></div><MediaPreview src={url} kind={artifact.kind} label={artifact.label} /><button className="button secondary" onClick={download} disabled={busy}>{busy ? <Loader2 className="spin" size={17} /> : <Download size={17} />}下载结果</button>{error && <p role="alert" className="field_error">{error}</p>}</article>;
}

export default function Workbench({ method, example }) {
  const [files, setFiles] = useState({});
  const [values, setValues] = useState(() => defaultsFor(method));
  const [errors, setErrors] = useState({});
  const [status, setStatus] = useState('idle');
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);
  const [elapsed, setElapsed] = useState(0);
  const [loadingSample, setLoadingSample] = useState(false);
  const controller = useRef(null);
  const sampleController = useRef(null);
  const started = useRef(0);
  const resultRef = useRef(null);
  const processing = status === 'processing';
  useEffect(() => () => { controller.current?.abort(); sampleController.current?.abort(); }, []);
  useEffect(() => {
    if (!processing) return;
    const interval = setInterval(() => setElapsed((performance.now() - started.current) / 1000), 200);
    return () => clearInterval(interval);
  }, [processing]);
  function clearResult() { setResult(null); setErrors({}); setError(''); setStatus('uploaded'); }
  function chooseFile(name, file) { clearResult(); setFiles(previous => ({ ...previous, [name]: file })); }
  function reset() { controller.current?.abort(); sampleController.current?.abort(); setFiles({}); setValues(defaultsFor(method)); setResult(null); setErrors({}); setError(''); setStatus('idle'); setElapsed(0); }
  async function useSample() {
    const active = new AbortController();
    sampleController.current?.abort(); sampleController.current = active;
    setLoadingSample(true); setError('');
    try {
      const input = method.inputs[0];
      if (method.inputs.length !== 1 || example.input.kind !== input.kind) throw new Error('多输入方法请按各输入框手动选择素材。');
      const response = await fetch(assetUrl(example.input.src), { signal: active.signal, credentials: 'omit' });
      if (!response.ok) throw new Error('样例素材加载失败，请检查样例路径。');
      let blob = await response.blob();
      let filename = example.input.src.split('/').pop();
      if (input.kind === 'image' && blob.type.includes('svg')) {
        const url = URL.createObjectURL(blob);
        try {
          const image = new Image();
          image.src = url;
          await image.decode();
          const canvas = document.createElement('canvas');
          canvas.width = image.naturalWidth; canvas.height = image.naturalHeight;
          canvas.getContext('2d').drawImage(image, 0, 0);
          blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/png'));
          filename = 'sample_image.png';
        } finally { URL.revokeObjectURL(url); }
      }
      active.signal.throwIfAborted();
      clearResult();
      setFiles({ [input.name]: new File([blob], filename, { type: blob.type }) });
      setValues({ ...defaultsFor(method), ...example.parameters });
    } catch (problem) { if (problem.name !== 'AbortError') setError(problem.message); }
    finally { if (sampleController.current === active) setLoadingSample(false); }
  }
  async function run(event) {
    event.preventDefault();
    const nextErrors = validateParameters(method, values);
    for (const input of method.inputs) { const message = validateFile(input, files[input.name]); if (message) nextErrors[input.name] = message; }
    setErrors(nextErrors); setError('');
    if (Object.keys(nextErrors).length) { setError('请检查素材和参数后再开始处理。'); return; }
    controller.current?.abort();
    const active = new AbortController(); controller.current = active;
    started.current = performance.now(); setElapsed(0); setStatus('processing'); setResult(null);
    try {
      const next = await runAlgorithm(method, files, values, active.signal);
      active.signal.throwIfAborted();
      setElapsed((performance.now() - started.current) / 1000); setResult(next); setStatus('completed');
      setTimeout(() => resultRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 60);
    } catch (problem) {
      if (active.signal.aborted) { setStatus('uploaded'); setError('已取消等待。外部算法是否停止执行，由其服务端决定。'); }
      else { setStatus('failed'); setError(problem.message || '处理失败，请重试。'); }
    }
  }
  return <section className="workbench section_space" id="workspace">
    <div className="section_heading"><div><span className="eyebrow">YOUR WORKSPACE</span><h2>上传素材，运行你的方法</h2><p>选择文件，再填写此方法需要的参数。</p></div><span className="pill"><ShieldCheck size={15} />{method.adapter.type === 'browser' ? '素材在浏览器本地处理' : '调用该组提供的算法服务'}</span></div>
    <form onSubmit={run} noValidate>
      <div className="workspace_grid">
        <div className="panel upload_panel"><div className="panel_title"><span className="step_number">01</span><h3>输入素材</h3><button type="button" className="text_button" disabled={processing || loadingSample} onClick={useSample}>{loadingSample ? <Loader2 size={15} className="spin" /> : <Play size={15} />}用样例试一试</button></div>
          {method.inputs.map(input => <FileInput key={input.name} input={input} file={files[input.name]} onChange={file => chooseFile(input.name, file)} disabled={processing || loadingSample} error={errors[input.name]} />)}
          <p className="input_note">{method.adapter.type === 'browser' ? '这个基础样例不会将文件上传到服务器；刷新或清空后会释放本页素材。' : '点击开始后，文件将发送到该方法的接口。请使用获准处理的素材。'}</p>
        </div>
        <div className="panel parameters_panel"><div className="panel_title"><span className="step_number">02</span><h3>方法参数</h3><SlidersHorizontal size={18} /></div><p className="muted">以下输入框由当前方法自行定义。</p>
          {method.parameters.length ? method.parameters.map(parameter => <ParameterField key={parameter.name} parameter={parameter} value={values[parameter.name]} error={errors[parameter.name]} disabled={processing || loadingSample} onChange={value => { clearResult(); setValues(previous => ({ ...previous, [parameter.name]: value })); }} />) : <p className="empty_parameters">此方法无需额外参数，选择素材即可开始。</p>}
        </div>
      </div>
      <div className="run_bar"><div><strong>{processing ? '算法正在处理素材' : status === 'completed' ? '处理已完成，可下载结果' : '准备好后，即可开始'}</strong><span>{processing ? `已等待 ${elapsed.toFixed(1)} 秒 · ${method.runtime || '独立算法服务'}` : `${method.title} · v${method.version}`}</span></div><div className="actions"><button className="button quiet" type="button" onClick={reset} disabled={processing || loadingSample}><RotateCcw size={17} />清空本页</button>{processing ? <button key="cancel" className="button secondary" type="button" onClick={event => { event.preventDefault(); controller.current?.abort(); }}>取消等待</button> : <button key="submit" className="button primary" type="submit" disabled={loadingSample}>开始处理<ArrowUpRight size={18} /></button>}</div></div>
    </form>
    {error && <div className="notice error" role="alert">{error}</div>}
    {processing && <div className="processing" role="status"><Loader2 size={25} className="spin" /><div><strong>{method.adapter.type === 'browser' ? '正在读取并处理文件…' : '文件已提交请求，正在等待算法返回…'}</strong><p>此处显示真实等待状态，完成时间由素材和算法决定。</p><div className="indeterminate"><i /></div></div></div>}
    {result && <section className="result_panel" ref={resultRef}><div className="section_heading"><div><span className="eyebrow"><CheckCircle2 size={15} /> RESULT</span><h2>处理结果</h2><p>{result.message}</p></div><button className="button secondary" onClick={() => downloadJson({ method_id: method.id, version: method.version, parameters: normalizeParameters(method, values), inputs: Object.fromEntries(Object.entries(files).filter(([, file]) => file).map(([name, file]) => [name, file.name])) }, `${method.id}_parameters.json`)}><Download size={16} />下载参数记录</button></div><div className="artifact_grid">{result.artifacts.map((artifact, index) => <Artifact key={index} artifact={artifact} />)}</div><p className="muted">本次耗时 {elapsed.toFixed(1)} 秒。清空本页会移除浏览器引用；外部服务的文件保留与删除由该算法服务负责。</p></section>}
  </section>;
}
