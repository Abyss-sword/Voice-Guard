import { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { Layers3, Plus, ArrowUpRight } from 'lucide-react';
import CatalogPage from './pages/CatalogPage.jsx';
import MethodPage from './pages/MethodPage.jsx';
import ConnectPage from './pages/ConnectPage.jsx';
import GuidePage from './pages/GuidePage.jsx';
import { loadCatalog, readLocalMethods } from './lib/catalog.js';
import './styles.css';
import './readability.css';

function App() {
  const [route, setRoute] = useState(location.hash.slice(1) || '/');
  const [builtins, setBuiltins] = useState([]);
  const [localMethods, setLocalMethods] = useState(readLocalMethods);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [warnings, setWarnings] = useState([]);
  useEffect(() => {
    const change = () => { setRoute(location.hash.slice(1) || '/'); window.scrollTo(0, 0); };
    addEventListener('hashchange', change);
    return () => removeEventListener('hashchange', change);
  }, []);
  useEffect(() => {
    const controller = new AbortController();
    loadCatalog(controller.signal).then(({ methods, warnings: nextWarnings }) => { setBuiltins(methods); setWarnings(nextWarnings); }).catch(problem => { if (!controller.signal.aborted) setError(problem.message); }).finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, []);
  const refresh = () => setLocalMethods(readLocalMethods());
  useEffect(() => { addEventListener('storage', refresh); return () => removeEventListener('storage', refresh); }, []);
  const methods = [...builtins, ...localMethods.filter(method => !builtins.some(builtin => builtin.id === method.id)).map(method => ({ ...method, local: true }))];
  const match = route.match(/^\/methods\/([a-z0-9_]+)$/);
  const method = match && methods.find(item => item.id === match[1]);
  return <><header className="navbar"><a className="brand" href="#/"><span className="brand_icon"><Layers3 size={22} /></span><span>算法研究平台<small>ALGORITHM HUB</small></span></a><nav aria-label="主导航"><a href="#/" className={route === '/' ? 'selected' : ''}>方法集合</a><a href="#/guide" className={route === '/guide' ? 'selected' : ''}>接入指南</a><a className="nav_connect" href="#/connect"><Plus size={16} />接入算法</a></nav></header><main>{error && <div className="notice error" role="alert">{error}<button className="text_button" onClick={() => location.reload()}>重新加载</button></div>}{warnings.length > 0 && <div className="notice" role="alert">部分方法配置未能加载：{warnings.join('；')}</div>}{loading ? <div className="loading_state">正在加载方法集合…</div> : route === '/' ? <CatalogPage methods={methods} /> : route === '/connect' ? <ConnectPage builtins={builtins} onChange={refresh} /> : route === '/guide' ? <GuidePage /> : method ? <MethodPage key={method.id} method={method} onChange={refresh} /> : <div className="empty_state"><h1>未找到这个方法</h1><p>方法可能已被移除，或尚未提交到仓库。</p><a className="button primary" href="#/">返回方法集合<ArrowUpRight size={17} /></a></div>}</main><footer><a className="brand footer_brand" href="#/"><Layers3 size={21} />算法研究平台</a><span>让方法被理解，让成果可体验。</span><a href="#/guide">输入 · 参数 · 结果<ArrowUpRight size={14} /></a></footer></>;
}

createRoot(document.getElementById('root')).render(<App />);
