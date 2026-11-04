import { useState } from 'react';
import { ArrowLeft, ArrowDown, FlaskConical, Download, Trash2 } from 'lucide-react';
import { MediaPreview } from '../components/MediaPreview.jsx';
import Workbench from '../components/Workbench.jsx';
import { assetUrl, downloadJson, removeLocalMethod } from '../lib/catalog.js';
import { kindLabels } from '../lib/contract.js';

export default function MethodPage({ method, onChange }) {
  const [exampleIndex, setExampleIndex] = useState(0);
  const example = method.examples[exampleIndex];
  function remove() {
    if (!confirm(`确定移除“${method.title}”在此浏览器中的登记吗？仓库源码和算法服务文件不会被删除。`)) return;
    try { removeLocalMethod(method.id); onChange(); location.hash = '/'; }
    catch { alert('浏览器存储无法访问，暂时无法移除登记。'); }
  }
  return <div className="method_page"><a className="back_link" href="#/"><ArrowLeft size={16} />返回方法集合</a><header className="method_header"><div><div className="tag_row"><span>{method.category}</span>{[...new Set(method.inputs.map(input => input.kind))].map(kind => <span key={kind}>{kindLabels[kind]}</span>)}<span>{method.runtime || '独立算法服务'}</span></div><h1>{method.title}</h1><p>{method.summary}</p><div className="method_meta"><FlaskConical size={16} />{method.team}<span>v{method.version}</span><span>{method.english}</span></div></div><a className="button primary" href="#workspace" onClick={event => { event.preventDefault(); document.getElementById('workspace')?.scrollIntoView({ behavior: 'smooth' }); }}>运行方法<ArrowDown size={18} /></a></header>
    <section className="showcase"><div className="section_heading"><div><span className="eyebrow">EXAMPLE SHOWCASE</span><h2>先看看，这个方法能做什么</h2><p>{example.title}</p></div><span className="pill">由 {method.team} 提供</span></div>{method.examples.length > 1 && <div className="filter_tabs example_tabs">{method.examples.map((item, index) => <button key={index} onClick={() => setExampleIndex(index)} className={exampleIndex === index ? 'active' : ''}>{item.title}</button>)}</div>}<div className="example_pair">{[['输入样例', example.input], ['输出样例', example.output]].map(([label, item], index) => <article className="example_card" key={label}><div className="example_title"><span className="step_number">0{index + 1}</span><h3>{item.label || label}</h3><span>{label}</span></div><MediaPreview kind={item.kind} src={assetUrl(item.src)} label={item.label || label} /></article>)}</div><p className="example_description">{example.description}</p><details className="method_description"><summary>方法说明与适用范围</summary><p>{method.description}</p></details></section>
    <Workbench key={method.id} method={method} example={example} />
    <div className="method_bottom"><button className="text_button" onClick={() => { const { local, ...config } = method; downloadJson(config, `${method.id}.json`); }}><Download size={16} />下载方法配置</button>{method.local && <button className="text_button danger" onClick={remove}><Trash2 size={16} />移除本机登记</button>}</div>
  </div>;
}
