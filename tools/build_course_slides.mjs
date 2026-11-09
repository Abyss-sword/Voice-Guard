// Editable course slides from measured evidence. Requires the bundled Artifact Tool runtime.
import fs from 'node:fs/promises';
import {readFileSync} from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';

const root=path.resolve('.');
const runtime=process.env.CODEX_PRIMARY_RUNTIME_NODE_MODULES || process.env.RUNTIME_NODE_MODULES;
if(!runtime) throw new Error('Set CODEX_PRIMARY_RUNTIME_NODE_MODULES to the bundled node_modules directory');
const skill=process.env.PRESENTATIONS_SKILL_DIR;
if(!skill) throw new Error('Set PRESENTATIONS_SKILL_DIR to the installed presentations skill');
const {Presentation,PresentationFile}=await import(pathToFileURL(path.join(runtime,'@oai/artifact-tool/dist/artifact_tool.mjs')).href);
const {finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const data=JSON.parse(await fs.readFile(path.join(root,'docs/evidence/2026-11-09/report-data.json'),'utf8'));
const stage=path.join(root,'outputs/course/slides');
await fs.mkdir(stage,{recursive:true});
const final=path.join(root,process.argv[2]||'deliverables/Voice_Guard_9分钟汇报.pptx');
await fs.mkdir(path.dirname(final),{recursive:true});
const p=Presentation.create({slideSize:{width:1280,height:720}});
const font='Microsoft YaHei';
const color={ink:'#172C40',muted:'#536476',accent:'#217A78',red:'#B23A48'};
const timings=[15,25,50,30,30,45,40,45,50,40,40,55,35,40];
if(timings.reduce((a,b)=>a+b,0)!==540) throw new Error('Talk must total 540 seconds');
const notes=[];
function text(s,content,x,y,w,h,size=26,ink=color.ink,bold=false){
  const shape=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  shape.text=content;shape.text.style={typeface:font,fontSize:size,color:ink,bold};return shape;
}
function slide(title,talk,source='docs/04_报告.md'){
  const s=p.slides.add();s.background.fill='#FFFFFF';const n=p.slides.items.length;
  text(s,title,66,40,1148,76,42,color.ink,true);
  text(s,`${n} / ${timings.length}`,1135,671,80,28,17,color.muted);
  s.speakerNotes.text=`建议用时 ${timings[n-1]} 秒。\n${talk}\n\n资料来源：${source}\n复现边界：闭集 ECAPA、第三方 MOSNet、小样本，非论文同协议总体性能。`;
  notes.push(`## 第 ${n} 页 ${title}（${timings[n-1]} 秒）\n\n${talk}\n\n来源：${source}\n`);return s;
}
function table(s,values,{y=155,height=360,width=1148,x=66,fontSize=25,widths}={}){
  const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width,height,values,...(widths?{columnWidths:widths}:{})});
  t.borders.assign({fill:'#D8E0E6',width:1,style:'solid'});
  t.cells.block({row:0,column:0,rowCount:values.length,columnCount:values[0].length}).assign({textStyle:{typeface:font,fontSize,color:color.ink},margins:{left:12,right:12,top:8,bottom:8}});
  t.cells.block({row:0,column:0,rowCount:1,columnCount:values[0].length}).assign({fill:'#E8F1F0',textStyle:{typeface:font,fontSize,bold:true,color:color.accent}});
  return t;
}
function figure(s,name,x,y,w,h){s.images.add({blob:readFileSync(path.join(root,'docs/figures/2026-11-09',name)),contentType:'image/png',alt:name,fit:'contain',position:{left:x,top:y,width:w,height:h}});}
const num=(v,d=3)=>v==null?'待评估':v.toFixed(d);
const pct=v=>v==null?'待评估':`${(v*100).toFixed(1)}%`;
const fraction=(k,n)=>n?`${k}/${n}`:'N/A';

let s=slide('Voice Guard','我们复现语音发布前的主动防御，并重点检查保护音频质量和转换端声纹两个条件。汇报将介绍方法实现、真实实验、失败分析和质量改进。');
text(s,'语音隐私主动防御\n复现与质量改进',66,180,1120,195,62,color.ink,true);
text(s,'A036 语音识别与合成课程大作业',68,420,1100,50,28,color.accent);
text(s,'华为云 Tesla T4 实测 · 技术记录 2026 年 11 月',68,497,1100,45,24,color.muted);
text(s,data.extension_complete?'原方法、冻结新句、配对信道与内容源实测':'新增实验进行中，以逐句证据为准',68,580,1100,38,22,color.muted);

s=slide('背景与问题','零样本语音转换能够把新句子的内容与目标说话人的声纹组合。Voice Guard 在目标语音发布前添加扰动。保护必须同时让语音可用、让转换后的身份偏离目标。仅让识别器认错人可能只是把声音破坏了，因此不能只报一个攻击成功率。','Voice Guard, IJCAI 2023, https://www.ijcai.org/proceedings/2023/535；docs/04_报告.md §1');
text(s,'公开语音可以成为声纹复制的目标样本',66,161,1120,65,35,color.accent,true);
table(s,[['目标音频 x','内容音频 t','语音转换 F(t,x)'],['提供说话人特征','提供要说的新句子','合成目标声纹的新语句']],{y:271,height:150,fontSize:28});
text(s,'防护要求：保护音频仍能正常使用，转换后难以复制目标身份。',66,480,1130,110,31);

s=slide('核心方法','可微 H 将时域波形变成说话人编码器输入，使梯度能回到波形。第一阶段把嵌入推向 least-likely 参考身份，并远离原目标。第二阶段加入心理声学掩蔽损失。实现保留官方代码 2ε 幅值和 c 搜索语义，3000/1500 不等于固定总步数。掩蔽约束的存在也不能替代实际音质评测。','docs/04_报告.md §2；vguard/features.py；vguard/masking.py；vguard/attack.py；third_party/voiceguard/');
text(s,'时域参数化   δ = 2ε tanh(w)',66,150,1148,60,32,color.accent,true);
text(s,'Lvc = MSE(Es(H(x+δ)), Es(H(y)))\n         - 0.1 MSE(Es(H(x+δ)), Es(H(x)))',66,252,1148,112,29);
text(s,'Lmask = mean(ReLU(PSD(δ) - θx))',66,406,1148,60,31);
text(s,'第一阶段嵌入优化，第二阶段联合 Lvc + c Lmask',66,514,1148,64,29);
text(s,'3000 / 1500 步及 c 搜索；θx 与扰动 PSD 使用同一 96 dB 基准',66,605,1148,38,22,color.muted);

s=slide('复现工程与替代评估器','补全缺失的心理声学 Transform，建立可微特征与 numpy 参考链对照。模型用官方 NTU AdaIN-VC，攻击 Es 与独立 ECAPA 评估库分离。MOSNet 是预测器，WER 对 VCTK 人工文本，谱距离只作项目内比较。这些替代意味着我们不能宣称精确复现论文表格。','docs/04_报告.md §2.3；docs/10_复现审计.md');
table(s,[['模块','本项目','复现边界'],['语音转换','官方 NTU AdaIN-VC，24 kHz','Griffin-Lim，非论文全模型配置'],['身份检查','20 人闭集 ECAPA','替代 GE2E，非真实身份认证'],['质量与内容','MOSNet 预测、Whisper WER','无人工 MOS，短句误差敏感'],['谱距离','mel 特征距离','非论文精确 MCD']],{height:363,fontSize:24,widths:[195,470,483]});
text(s,'身份评估模型与质量评估模型均不参与改进方法的优化损失。',66,575,1148,58,25,color.accent);

s=slide('实验协议与成功条件','数据共 20 位说话人、865 句。每人前 30 句注册，首批选 5 人各 1 句。先检查干净转换能否识别目标，再检查保护后身份保持、转换身份变化，最后加上 MOSNet 和 WER 门槛。p229 干净转换失败，须排除防护分母，但质量结果仍保留。','docs/evidence/2026-11-09/frozen-protocol.json；docs/04_报告.md §3');
text(s,'20 人注册库，865 句原始音频与人工文本',66,150,1148,55,32,color.accent,true);
table(s,[['检查层','通过条件'],['干净控制','ASV(干净转换) = 目标 X'],['身份维','ASV(保护音频) = X，ASV(保护后转换) ≠ X'],['联合门槛','身份维通过，MOSNet ≥ 3，目标文本 WER ≤ 10%']],{y:255,height:260,fontSize:26,widths:[230,918]});
text(s,'新验证：同一批已见说话人的 10 个新句子，参数事前冻结。',66,574,1148,65,26);

s=slide('原始方法与消融','VG 和无掩蔽消融身份维为 2/4，但联合门槛均为 0/4。emb 和 e2e 各联合通过 1/4，fb 为 0/4。原 VG 五句 MOSNet 均值 1.397、WER 62.76%，说明身份变化可能伴随严重质量损失。四档 ε 扫描也都没有联合通过，不能单凭降低或升高 ε 解决质量与防护的矛盾。主表与扫描 ε=.05 的实际产物有差异，原因尚未定位，保留两次不择优。','docs/evidence/2026-11-09/report-data.json；docs/04_报告.md §4');
table(s,[['方法','筛选后身份成功','保护 MOSNet','保护 WER'],...data.campaign.groups.filter(r=>r.name.startsWith('main_')).map(r=>[r.name.replace('main_',''),`${r.screened_successes}/${r.screened}`,num(r.mean_mos_defended),pct(r.mean_wer_defended)])],{height:366,fontSize:25,widths:[250,300,300,298]});
text(s,'联合通过：VG 0/4，VG-nop 0/4，emb 1/4，fb 0/4，e2e 1/4。',66,568,1148,60,26,color.red);

s=slide('原始 VG 的质量问题','左图比较每一句的 MOSNet，右图比较逐句 WER。五句都纳入均值，包括不通过和排除样本。原方法平均 MOSNet 1.397，改进后 3.192，WER 从 62.76% 下降至 5.71%。MOSNet 是第三方未裁剪原始预测，不能读成人类打分。工程门槛是本项目定义，不能把 WER 较低解释为超过论文。','docs/evidence/2026-11-04/quality/comparison.json；docs/figures/2026-11-09/01_quality_comparison.png');
figure(s,'01_quality_comparison.png',55,142,1170,396);
text(s,'五句均值：MOSNet 1.397 / 3.192，WER 62.76% / 5.71%',66,559,1148,52,29,color.accent,true);
text(s,'原始 VG / 谱投影后继续优化。均值保留所有失败，MOSNet 非人类 MOS。',66,622,1148,34,21,color.muted);

s=slide('谱投影后继续优化','直接缩小扰动能改善质量却损失防护，因此增加投影与嵌入目标交替优化。先后各 30 次谱投影，中间 1000 次 Adam，每步后再投影，学习率 0.0003，余量 3 dB。图显示改进大幅减少掩蔽超额，但平均 PSD 低于阈值并不代表每个时频点满足，也不能声称人耳不可感知。','scripts/refine_masked.py；vguard/attack.py refine_masked；docs/figures/2026-11-09/masking_p225.json');
figure(s,'03_masking_p225.png',55,145,1170,390);
text(s,'固定参数：1000 步，学习率 0.0003，掩蔽余量 3 dB',66,562,1148,49,29,color.accent);
text(s,'投影与嵌入优化交替执行。仍有超阈格，不能据此宣称无感知扰动。',66,621,1148,34,22,color.muted);

s=slide('逐句质量与身份联合验证','最终 PCM16 文件再评测，p225 和 p228 联合通过。p226 的音质虽恢复，转换声纹仍是原目标。p227 丢了开头两词，身份防护也未成功。p229 的干净转换已经失败，须排除。开发 1/2，固定参数验证 1/2，都是很小的样本。独立重跑 p225 仍通过，但波形不逐位一致。','docs/evidence/2026-11-04/quality/comparison.json；docs/12_质量改进实验.md');
table(s,[['句子','改进 MOS','改进 WER','联合结果'],...data.initial_quality.rows.map(r=>[r.utterance,num(r.refined_mos),pct(r.refined_wer),r.screened_joint_pass?'通过':(!r.clean_vc_identity_ok?'排除':'未通过')])],{height:361,fontSize:26,widths:[310,260,260,318]});
text(s,'开发集 1/2，冻结参数验证集 1/2；样本不足以外推总体防御率。',66,566,1148,65,27,color.red);

s=slide('轻声裁剪导致内容丢失','p227 原句是 It is not for any other purpose。top_db=15 裁掉了较轻的开头，连干净原音也少了 It is。恢复完整上下文后 WER 从 28.57% 变为 0，说明部分内容损失来自预处理。但完整保护音频 MOSNet 2.591，转换仍是 p227，所以修复内容不代表整个防护达标。','docs/evidence/2026-11-04/quality/comparison.json；docs/12_质量改进实验.md 完整上下文诊断');
figure(s,'04_crop_p227.png',55,162,1170,330);
text(s,'保留上下文：目标句 WER 28.57% 降为 0',66,519,1148,56,32,color.accent,true);
text(s,'该句仍未联合达标：MOSNet 2.591，转换仍识别为 p227。',66,599,1148,48,26,color.red);
if(data.context_posthoc){const v=data.context_posthoc.summary;text(s,`新 10 句完整上下文：WER ${pct(v.macro_wer)}，联合 ${fraction(v.joint_successes,v.screened)}（事后诊断）`,66,650,1000,25,19,color.muted);}

s=slide('冻结参数的新句验证','新增十句在运行前冻结句子、输入摘要和全部超参数。同一批五位说话人，每人两个未用于开发的新句；这验证新句迁移，不是未见说话人。原法和改进均对最终 PCM16 音频评测，所有句子纳入质量均值。请结合表中干净转换分母读联合结果，不把旧句开发结果代替新句验证。','docs/evidence/2026-11-09/frozen-protocol.json；docs/evidence/2026-11-09/report-data.json');
const fresh=Object.entries(data.new_utterance_quality);
table(s,[['方法','新句数','联合通过','保护 MOS','保护 WER'],...fresh.map(([name,value])=>[name,value.summary.total,fraction(value.summary.joint_successes,value.summary.screened),num(value.summary.mean_mos),pct(value.summary.macro_wer)])],{y:180,height:200,fontSize:26,widths:[230,170,230,259,259]});
text(s,'5 位已见说话人 × 2 新句；3000/1500 + 固定 1000 步改进',66,435,1148,60,28,color.accent,true);
const refinedFresh=data.new_utterance_quality.refined?.rows||[];
const qualityOnly=refinedFresh.filter(r=>r.clean_vc_identity_ok&&r.quality_ok&&!r.identity_ok).length;
text(s,refinedFresh.length?`改进中 ${qualityOnly} 句质量达标，但转换身份仍为目标。`:'质量均值保留全部句子；联合率只筛除配对干净转换失败。',66,535,1148,75,27);
text(s,'句子相关、小样本；不能外推未见人物或论文总体防御率。',66,623,1148,34,22,color.red);

s=slide('配对信道与内容源实验','配对处理干净与保护目标，每种条件重新转换并筛选。改进法 p225 在 MP3 128 kbps 下音质仍通过，但转换恢复为目标，是真实的防护丢失；32 kbps 下反而保留，码率关系不单调。G711 和8k使原样成功样本的干净控制失败，不能一概归为扰动失效。内容源换为 p225_002 时，开发组由1/2降为0/2。八种信道80次、三个内容源60次，都是相关的小样本实验室控制。','scripts/audit_channels.py；docs/04_报告.md §6');
const channelNames=['none','pcm16','mp3-128k','mp3-32k','aac-64k','opus-24k','g711','resample8k'];
function aggregateChannel(names,channel){
  const stats=names.map(name=>data.audits[name]?.[channel]).filter(Boolean);
  if(!stats.length)return '待评估';
  const n=stats.reduce((a,v)=>a+v.screened,0),k=stats.reduce((a,v)=>a+v.joint_successes,0);
  return n?`${k}/${n}`:'N/A';
}
table(s,[['处理条件','原 VG 联合通过','改进联合通过'],...channelNames.map(channel=>[channel,aggregateChannel(['channels_baseline'],channel),aggregateChannel(['channels_refined_dev','channels_refined_validation'],channel)])],{y:140,height:430,fontSize:22,widths:[370,389,389]});
text(s,'8 信道共 80 次配对试验；3 内容源 × 2 条件共 60 次。',66,596,1148,40,25,color.accent);
text(s,'ffmpeg 实验室控制，不是微信/抖音线上测量或人耳听评。',66,641,1148,25,19,color.muted);

s=slide('华为云与课程平台实测','统一 Pipeline 接入 Gradio 和课程 FastAPI。云端 Ubuntu22.04、Tesla T4、torch2.5.1，systemd 管理服务。真实浏览器从课程平台载入样例、调用 GPU、播放音频并下载文件，记录无页面错误。双声道及自动采样率 API 返回全部六个产物，运算期间健康检查仍响应。快速模式用于交互验证，正式实验使用完整步数。该在线例子的干净转换身份失败，页面正确显示未通过，不冒充成功防御。','docs/evidence/2026-11-09/platform-browser.json；platform-api-projected.json；gradio-api.json');
s.images.add({blob:readFileSync(path.join(root,'docs/evidence/2026-11-09/platform-results.png')),contentType:'image/png',alt:'真实课程平台 GPU 请求结果',fit:'contain',position:{left:66,top:145,width:730,height:453}});
text(s,'Tesla T4 / CUDA 12.4\nFastAPI + Gradio\n6 份产物下载通过\n5 份音频可播放',824,178,390,205,27,color.accent,true);
text(s,'浏览器 48.33 秒\nAPI 49.23 秒\nhealth 0.078 秒',824,432,390,145,25);
text(s,'服务绑定回环地址，经 SSH 隧道访问；没有宣称公开 HTTPS 生产部署。',66,627,1148,34,21,color.muted);

s=slide('结论与局限','本项目完成核心方法、独立质量和身份评估、华为云部署及真实课程前端调用。原法的声纹变化不能代表有效保护。谱投影继续优化改善首批质量，冻结新句和配对信道结果明确给出其可迁移性边界，负结果保留。裁剪控制定位了部分内容丢失的原因。大样本、论文原评估器、其他转换模型、真实移动端录音和人工听评尚缺，不能宣称论文总体质量已复现。','docs/04_报告.md §9–10；docs/13_课程要求与交付清单.md');
text(s,'可运行复现、配对控制与可复核的失败分析',66,167,1148,60,35,color.accent,true);
const freshJoint=name=>{const v=data.new_utterance_quality[name]?.summary;return v?fraction(v.joint_successes,v.screened):'待评估'};
text(s,`原 VG：身份成功 2/4，联合质量成功 0/4。\n谱投影优化：首批五句质量改善，联合通过 2/4。\n新 10 句联合通过：原法 ${freshJoint('baseline')}，改进 ${freshJoint('refined')}。`,66,271,1148,178,29);
text(s,'限制：闭集、小样本、替代评估器，无人工听评。\n尚不足以宣称达到论文总体质量或人耳不可感知。',66,515,1148,100,29,color.red);

await fs.writeFile(path.join(root,'docs/05_验收PPT大纲.md'),(`# 9 分钟汇报讲稿\n\n总计 ${timings.reduce((a,b)=>a+b,0)} 秒。真实成员确定后，按本人实际工作分配讲解，每人均需讲解。\n本汇报基于归档实测数据；新句及信道完整状态以报告为准。\n\n${notes.join('\n')}`).trimEnd()+'\n');
const candidate=path.join(stage,'candidate.pptx');
await(await PresentationFile.exportPptx(p)).save(candidate);
for(const [i,sl] of p.slides.items.entries()){
  const blob=await p.export({slide:sl,format:'png',scale:1});
  await fs.writeFile(path.join(stage,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await blob.arrayBuffer()));
}
const result=await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath:final,
  pythonExecutable:process.env.RUNTIME_PYTHON,integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...[2,4,5,6,9,11,12].flatMap(n=>['--require-native-table-slide',String(n)])],
  explicitTotalSlideCount:14,requiredNativeTableOwnerSlides:[2,4,5,6,9,11,12],requiredNativeChartOwnerSlides:[],
  fontPolicy:{basis:'design',families:[font]},verifyArtifactToolImport:true,receiptPath:path.join(stage,`${path.basename(final)}.validation.json`)});
console.log(JSON.stringify(result,null,2));
