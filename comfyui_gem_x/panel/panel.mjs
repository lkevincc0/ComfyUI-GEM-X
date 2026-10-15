import {t,language,canonical,applyStatic,setLanguage,chineseDefaults} from './i18n.mjs';
const $=id=>document.getElementById(id);
const template=await fetch('./workflow.json').then(r=>{if(!r.ok)throw new Error('工作流加载失败');return r.json();});
const clientId=crypto.randomUUID(), historyKey='gem-motion-studio-jobs';
let jobs=[];try{jobs=JSON.parse(localStorage.getItem(historyKey)||'[]');}catch{}
let mode='generate',busy=false,activeId='',watchedId='',socket,started=0,polling=false,liveStage='';
let currentOutputs={},sourceFile=null,sourcePromptId='',pendingVideo=null,sourceURL='',sourceDuration=0;
let selectedRig='148',modelKey='',viewer,clipPlaying=false;
let lastStatus={name:'',message:'就绪，等待开始',percent:0,params:{}};
const bvhFiles=new Map(),bvhPending=new Set();
async function api(path,body){
    const response=await fetch(path,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const text=await response.text();let data;try{data=JSON.parse(text);}catch{throw new Error(text||`请求失败 (${response.status})`);}
    if(!response.ok||data.error){const details=Object.values(data.node_errors||{}).flatMap(n=>n.errors||[]).map(e=>e.message+': '+e.details).join('\n');throw new Error(details||data.error?.message||data.error||`请求失败 (${response.status})`);}return data;
}
function fileURL(file){return '/view?'+new URLSearchParams({filename:file.filename,subfolder:file.subfolder||'',type:file.type||'output'});}
function error(e){$('error').textContent=t(e.message||String(e));$('error').hidden=false;}
function stage(name,message,percent,params={}){lastStatus={name,message,percent,params};liveStage=name;$('status').textContent=t(message,params);$('progress').value=percent;for(const el of document.querySelectorAll('[data-stage]'))el.classList.toggle('active',el.dataset.stage===name);}
function classify(id){id=String(id);if(id==='144')return['capture','正在捕捉人物动作…',65];if(['147','148','149','150','151','152'].includes(id))return['retarget','正在重定向角色动画…',88];if(['142','143','146'].includes(id))return['capture','正在准备动捕片段…',55];return['generation','正在生成视频…',20];}
function refreshJobs(){const select=$('jobs');select.replaceChildren();for(const j of jobs){const option=new Option(`${j.label} · ${t(j.kind==='generate'?'视频':'动捕')} · ${t(j.state||'处理中')}`,j.id);select.add(option);}if(!jobs.length)select.add(new Option(t('暂无任务'),''));select.value=watchedId;localStorage.setItem(historyKey,JSON.stringify(jobs.slice(0,20)));}
function changeJob(id,state){const j=jobs.find(j=>j.id===id);if(j)j.state=state;refreshJobs();}
function setBusy(value){busy=value;$('submit').disabled=value;$('capture').disabled=value;}
function setVideo(id,file){const video=$(id+'-video'),url=typeof file==='string'?file:fileURL(file);if(video.getAttribute('src')!==url)video.src=url;video.hidden=false;$(id+'-empty').hidden=true;}
function useSource(file,parentId='',selection=null){
    const url=typeof file==='string'?file:fileURL(file);sourcePromptId=parentId;
    if(typeof file!=='string')sourceFile=file;
    if(sourceURL===url)return;
    sourceURL=url;sourceDuration=0;setVideo('source',file);
    $('source-video').onloadedmetadata=()=>{sourceDuration=$('source-video').duration;if(!Number.isFinite(sourceDuration))return;
        for(const id of ['trim-start','trim-end','trim-start-slider','trim-end-slider']){$(id).max=sourceDuration;}
        setSelection(selection?.start||0,selection?.end||sourceDuration);$('clip-controls').hidden=false;
    };
}
function setSelection(start,end,seek=false){
    start=Math.max(0,Math.min(start,sourceDuration));end=Math.max(start,Math.min(end,sourceDuration));
    $('trim-start').value=start.toFixed(3);$('trim-end').value=end.toFixed(3);$('trim-start-slider').value=start;$('trim-end-slider').value=end;
    $('clip-range').style.setProperty('--start',sourceDuration?start/sourceDuration*100+'%':'0%');$('clip-range').style.setProperty('--end',sourceDuration?end/sourceDuration*100+'%':'100%');
    $('clip-summary').textContent=t('{start}–{end} 秒 · 已选 {selected} / {total} 秒',{start:start.toFixed(2),end:end.toFixed(2),selected:(end-start).toFixed(2),total:sourceDuration.toFixed(2)});
    if(seek){$('source-video').pause();$('source-video').currentTime=start;clipPlaying=false;}
}
for(const id of ['trim-start','trim-start-slider'])$(id).oninput=()=>setSelection(Math.min(Number($(id).value),Number($('trim-end').value)),Number($('trim-end').value),true);
for(const id of ['trim-end','trim-end-slider'])$(id).oninput=()=>{const end=Math.max(Number($('trim-start').value),Number($(id).value));setSelection(Number($('trim-start').value),end);$('source-video').pause();$('source-video').currentTime=Math.min(end,sourceDuration-.001);clipPlaying=false;};
$('select-all').onclick=()=>setSelection(0,sourceDuration,true);
$('preview-clip').onclick=()=>{const video=$('source-video');video.currentTime=Number($('trim-start').value);video.loop=false;clipPlaying=true;video.play().catch(error);};
$('source-video').ontimeupdate=()=>{if(clipPlaying&&$('source-video').currentTime>=Number($('trim-end').value)){$('source-video').pause();clipPlaying=false;}};
function addLink(label,file,kind){const a=document.createElement('a');a.href=kind?`/download/${watchedId}/${kind}`:fileURL(file);a.download=file.filename;a.textContent='↓ '+t(label);$('downloads').append(a);}
function downloads(){
    $('downloads').replaceChildren();if(sourceFile)addLink('原始视频',sourceFile,'source');
    const result=currentOutputs['144']?.gem_result?.[0],video=currentOutputs['157']?.images?.[0];
    if(video&&(!sourceFile||fileURL(video)!==fileURL(sourceFile)))addLink('动捕片段 MP4',video,'clip');
    if(result){if(result.files.preview)addLink('关键点预览',result.files.preview,'preview');addLink('动作 NPZ',result.files.motion,'motion');addLink('结果信息',result.files.manifest,'manifest');}
    const bvh=bvhFiles.get(watchedId);if(bvh){addLink('BVH · 77 关节',bvh['77j'],'bvh77');addLink('BVH · 78 节点含 Root',bvh['78j'],'bvh78');}
    for(const [id,label]of[['148','Mixamo 动画 FBX'],['150','Unreal 动画 FBX'],['152','Unity 动画 FBX']]){const file=currentOutputs[id]?.['3d']?.[0];if(file)addLink(label,file,{'148':'mixamo','150':'unreal','152':'unity'}[id]);}
    if(!$('downloads').children.length)$('downloads').textContent=t('等待输出。');
    $('download-all').disabled=!['144','148','150','152'].every(id=>currentOutputs[id]);
}
async function ensureBVH(id){if(bvhFiles.has(id)||bvhPending.has(id))return;bvhPending.add(id);try{const result=await api('/export-bvh',{prompt_id:id});bvhFiles.set(id,result.files);if(watchedId===id)downloads();}catch(e){if(watchedId===id)error(e);}finally{bvhPending.delete(id);}}
function updateResults(outputs){
    currentOutputs={...currentOutputs,...outputs};const job=jobs.find(j=>j.id===watchedId),video=currentOutputs['157']?.images?.[0],result=currentOutputs['144']?.gem_result?.[0];
    if(job?.source)useSource(job.source,job.sourcePromptId,job.selection);else if(video)useSource(video,watchedId);
    if(result){if(result.files.preview)setVideo('pose',result.files.preview);$('job-detail').textContent=t('{frames} 帧 · {fps} FPS',{frames:result.frames,fps:result.fps});}
    downloads();loadRig().catch(error);
}
function finish(id,successful,message){
    const job=jobs.find(j=>j.id===id),generation=job?.kind==='generate';changeJob(id,successful?(generation?'视频已生成':'已完成'):'失败');
    if(activeId===id){activeId='';setBusy(false);}
    if(successful&&!generation)ensureBVH(id);
    if(id===watchedId){if(successful)stage(generation?'generation':'complete',generation?'视频已生成。选择片段后开始动作捕捉。':'动作已就绪，可以播放与下载。',100);else{stage('','任务失败',0);error(new Error(message));}}
}
async function pollJob(){
    if(!watchedId||polling)return;polling=true;const id=watchedId;
    try{const history=await api('/history/'+encodeURIComponent(id));if(id!==watchedId)return;const entry=history[id];
        if(entry){updateResults(entry.outputs||{});if(entry.status?.completed||entry.status?.status_str==='error'){const failure=entry.status?.messages?.find(m=>m[0]==='execution_error')?.[1];finish(id,entry.status.status_str==='success',failure?.exception_message||'执行失败，请查看 ComfyUI 日志。');return;}}
        if(busy&&activeId===id){const queue=await api('/queue');const position=queue.queue_pending.findIndex(j=>j[1]===id);if(position>=0)stage('','排队中 · 前方 {count} 个任务',0,{count:position});}
    }catch(e){$('connection').textContent=t('连接中断，正在重试…');if(!busy)error(e);}finally{polling=false;}
}
function connect(){
    const url=new URL('/ws',location.href);url.protocol=location.protocol==='https:'?'wss:':'ws:';url.searchParams.set('clientId',clientId);socket=new WebSocket(url);
    socket.onopen=()=>{$('connection').innerHTML='<span class="dot"></span>'+t('ComfyUI 已连接');};socket.onclose=()=>{$('connection').textContent=t('连接中断，正在重试…');setTimeout(connect,3000);};socket.onerror=()=>socket.close();
    socket.onmessage=async event=>{if(typeof event.data!=='string')return;const{type,data}=JSON.parse(event.data);if(!activeId||data.prompt_id!==activeId)return;
        if(type==='executing'&&data.node){changeJob(activeId,'执行中');if(watchedId===activeId)stage(...classify(data.node));}
        if(type==='progress'&&watchedId===activeId){stage(liveStage,'{stage} · {value}/{max}',liveStage==='generation'?15+data.value/data.max*35:$('progress').value,{stage:liveStage==='generation'?'视频生成':'正在处理',value:data.value,max:data.max});}
        if(type==='executed'&&watchedId===activeId)updateResults({[data.node]:data.output});
        if(type==='execution_error'||type==='execution_interrupted')finish(activeId,false,data.exception_message||'任务已中断');
        if(type==='execution_success'||(type==='executing'&&!data.node))await pollJob();
    };
}
function setMode(value){mode=value;$('generate-tab').setAttribute('aria-selected',value==='generate');$('upload-tab').setAttribute('aria-selected',value==='upload');$('generate-fields').hidden=value!=='generate';$('upload-fields').hidden=value!=='upload';$('submit').hidden=value==='upload';$('action').required=value==='generate';$('source-title').textContent=t(value==='generate'?'生成视频':'输入视频');}
$('generate-tab').onclick=()=>setMode('generate');$('upload-tab').onclick=()=>setMode('upload');
$('aspect').onchange=()=>{const value=$('aspect').value;if(value.startsWith('1:1')){$('width').value=640;$('height').value=640;}else if(value.startsWith('9:16')){$('width').value=480;$('height').value=864;}else{$('width').value=864;$('height').value=480;}};
$('video-file').onchange=()=>{const file=$('video-file').files[0];if(!file)return;pendingVideo=file;sourceFile=null;sourcePromptId='';if(sourceURL.startsWith('blob:'))URL.revokeObjectURL(sourceURL);useSource(URL.createObjectURL(file));};
async function upload(file){const form=new FormData();form.set('image',file);form.set('type','input');form.set('overwrite','false');const response=await fetch('/upload/image',{method:'POST',body:form});if(!response.ok)throw new Error('视频/图片上传失败');const result=await response.json();return{filename:result.name,subfolder:result.subfolder||'',type:result.type||'input'};}
function annotated(file){return(file.subfolder?file.subfolder+'/':'')+file.filename+` [${file.type||'input'}]`;}
function prune(graph,roots){const keep=new Set();function visit(id){if(keep.has(id))return;keep.add(id);for(const v of Object.values(graph[id].inputs))if(Array.isArray(v)&&typeof v[0]==='string'&&graph[v[0]])visit(v[0]);}for(const id of roots)visit(id);return Object.fromEntries(Object.entries(graph).filter(([id])=>keep.has(id)));}
function clearMotion(){currentOutputs={};modelKey='';if(viewer)viewer.clear();$('pose-video').hidden=true;$('pose-empty').hidden=false;$('model-empty').textContent=t('等待角色动画');$('model-empty').hidden=false;downloads();}
async function submitGraph(graph,roots,job,metadata={}){
    const result=await api('/prompt',{prompt:prune(graph,roots),client_id:clientId,extra_data:{motion_studio:metadata}});
    activeId=result.prompt_id;watchedId=activeId;clearMotion();
    jobs.unshift({id:activeId,label:new Date().toLocaleString('zh-CN',{month:'numeric',day:'numeric',hour:'2-digit',minute:'2-digit'}),state:'排队中',...job});refreshJobs();stage('','任务已提交，等待执行…',0);await pollJob();
}
$('form').onsubmit=async event=>{
    event.preventDefault();if(busy)return;$('error').hidden=true;setBusy(true);started=Date.now();stage('generation','正在准备生成视频…',0);
    try{const graph=structuredClone(template.prompt),input=graph['140:131'].inputs,seed=Number($('seed').value)||Math.floor(Math.random()*2**48);
        input.prompt=`${$('rules').value.trim()}\n\n---\n\n${$('action').value.trim()}`;input.width=Number($('width').value);input.height=Number($('height').value);
        graph['140:133'].inputs.value=Number($('duration').value);graph['140:129'].inputs.noise_seed=seed;
        graph['140:138'].inputs.value=Number($('steps').value);graph['140:137'].inputs.value=Number($('steps').value);graph['140:139'].inputs.value=$('turbo').checked;graph['140:134'].inputs.strength_model=Number($('lora-strength').value);
        for(const[fileId,name]of[['first-image','first_frame'],['last-image','last_frame']]){const file=$(fileId).files[0];if(file){graph[fileId]={class_type:'LoadImage',inputs:{image:annotated(await upload(file))}};input[name]=[fileId,0];}}
        sourceFile=null;sourceURL='';pendingVideo=null;sourceDuration=0;$('source-video').hidden=true;$('source-empty').hidden=false;$('clip-controls').hidden=true;
        await submitGraph(graph,['157'],{kind:'generate',seed});
    }catch(e){setBusy(false);stage('','未能提交任务',0);error(e);}
};
$('capture').onclick=async()=>{
    if(busy||!sourceDuration)return;$('error').hidden=true;setBusy(true);started=Date.now();stage('capture','正在准备动捕片段…',0);
    try{if(pendingVideo){sourceFile=await upload(pendingVideo);pendingVideo=null;}
        if(!sourceFile)throw new Error('请先生成或上传视频');
        const graph=structuredClone(template.prompt),start=Number($('trim-start').value),end=Number($('trim-end').value);
        if(end<=start)throw new Error('请选择非空的视频片段');
        graph.panel_video={class_type:'LoadVideo',inputs:{file:annotated(sourceFile)}};
        graph['146'].inputs={video:['panel_video',0],start_time:start,duration:start===0&&Math.abs(end-sourceDuration)<.01?0:end-start,strict_duration:false};
        graph['157'].inputs.video=['146',0];graph['157'].inputs.filename_prefix='video/MotionStudioCapture';
        for(const id of ['147','149','151'])Object.assign(graph[id].inputs,{root_motion:$('root-motion').checked,yaw_offset:Number($('yaw').value),material:$('material').value});
        const selection={start,end};await submitGraph(graph,['157','144','148','150','152'],{kind:'capture',source:sourceFile,sourcePromptId,selection},{source_video:sourceFile,generation_prompt_id:sourcePromptId,selection});
    }catch(e){setBusy(false);stage('','未能提交动捕',0);error(e);}
};
$('jobs').onchange=()=>{watchedId=$('jobs').value;sourceURL='';clearMotion();$('error').hidden=true;stage('','正在读取任务结果…',0);pollJob();};
$('download-all').onclick=async()=>{
    const id=watchedId;$('download-all').disabled=true;$('download-all').textContent=t('正在打包…');
    try{const response=await fetch('/download-all',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt_id:id})});if(!response.ok)throw new Error(await response.text());
        const url=URL.createObjectURL(await response.blob()),link=document.createElement('a');link.href=url;link.download=`motion_${id.slice(0,8)}.zip`;link.click();setTimeout(()=>URL.revokeObjectURL(url),60000);
    }catch(e){error(e);}finally{$('download-all').textContent=t('下载全部结果 ZIP');downloads();}
};
async function loadRig(){const output=currentOutputs[selectedRig],file=output?.['3d']?.[0];if(!file)return;const key=fileURL(file);if(key===modelKey)return;modelKey=key;$('model-empty').textContent=t('正在加载角色…');$('model-empty').hidden=false;
    if(!viewer){const{MotionViewer}=await import('./viewer.mjs');viewer=new MotionViewer($('viewer'));}
    try{await viewer.load(key,output.model_3d_info?.[0]);}catch(e){if(modelKey===key){modelKey='';$('model-empty').textContent=t('角色预览加载失败，可切换角色重试。');}throw e;}if(modelKey===key)$('model-empty').hidden=true;
}
for(const button of document.querySelectorAll('[data-rig]'))button.onclick=()=>{selectedRig=button.dataset.rig;for(const item of document.querySelectorAll('[data-rig]'))item.setAttribute('aria-selected',item===button);loadRig().catch(error);};
$('play').onclick=()=>viewer?.toggle();$('timeline').oninput=()=>viewer?.seek(Number($('timeline').value));$('speed').onchange=()=>viewer?.setSpeed(Number($('speed').value));$('reset-view').onclick=()=>viewer?.frame();
const extract=name=>template.defaults.prompt.match(new RegExp(`<${name}>([\\s\\S]*?)</${name}>`))?.[1].trim()||'';
const promptDefaults={zh:chineseDefaults,en:{rules:extract('rule'),action:extract('character')+'\n\n'+extract('action_description')}};
const drafts={};let promptLanguage=language;
function restoreDefaults(){const draft=drafts[language]||promptDefaults[language];$('rules').value=draft.rules;$('action').value=draft.action;}
restoreDefaults();$('duration').value=template.defaults.duration;$('language').value=language;applyStatic();
$('language').onchange=()=>{
    drafts[promptLanguage]={rules:$('rules').value,action:$('action').value};
    setLanguage($('language').value);promptLanguage=language;restoreDefaults();setMode(mode);refreshJobs();downloads();
    if(sourceDuration)setSelection(Number($('trim-start').value),Number($('trim-end').value));
    const result=currentOutputs['144']?.gem_result?.[0];if(result)$('job-detail').textContent=t('{frames} 帧 · {fps} FPS',{frames:result.frames,fps:result.fps});
    $('model-empty').textContent=t(canonical($('model-empty').textContent));
    $('play').textContent=t(viewer?.playing?'暂停':'播放');
    stage(lastStatus.name,lastStatus.message,lastStatus.percent,lastStatus.params);
    $('connection').innerHTML=socket?.readyState===WebSocket.OPEN?'<span class="dot"></span>'+t('ComfyUI 已连接'):t('连接中…');
};
refreshJobs();connect();
if(jobs.length){watchedId=jobs[0].id;$('jobs').value=watchedId;if(!['视频已生成','已完成','失败'].includes(jobs[0].state)){activeId=watchedId;setBusy(true);started=Date.now();}await pollJob();}
setInterval(()=>{if(busy||(watchedId&&!['视频已生成','已完成','失败'].includes(jobs.find(j=>j.id===watchedId)?.state)))pollJob();if(busy)$('elapsed').textContent=t('{seconds} 秒',{seconds:Math.round((Date.now()-started)/1000)});},2500);
