export let language=localStorage.getItem('gem-motion-language')||'zh';
const english={
'视频生成与动作捕捉':'Video to Motion','文字生成视频':'Generate video','上传视频动捕':'Upload video','约束提示词':'Constraint prompt','动作提示词':'Action prompt','视频时长 · 秒':'Video duration · seconds','画面比例':'Aspect ratio','1:1 正方形':'1:1 Square','9:16 竖屏':'9:16 Portrait','16:9 横屏':'16:9 Landscape','宽度':'Width','高度':'Height','采样步数':'Sampling steps','随机种子':'Seed','参考图与生成设置':'References & generation','首帧参考图':'First-frame reference','尾帧参考图':'Last-frame reference','Turbo 加速':'Turbo mode','Turbo LoRA 强度':'Turbo LoRA strength','选择单人视频':'Select a single-person video','更多设置':'More settings','保留角色在场景中的移动':'Keep root motion','角色朝向 · 度':'Facing offset · degrees','角色材质':'Character material','自动':'Auto','原始材质':'Original','红色':'Red','橙色':'Orange','灰色':'Gray','蓝色':'Blue','生成视频':'Generate video','输入视频':'Input video','任务记录':'Tasks','暂无任务':'No tasks','结果保存在服务器的 ComfyUI/output 中。':'Results are saved in ComfyUI/output on the server.','就绪，等待开始':'Ready','01 视频生成':'01 Video','02 动作捕捉':'02 Capture','03 角色重定向':'03 Retarget','04 完成':'04 Done','视频将在这里显示':'Video preview','动作捕捉预览':'Motion capture preview','等待动捕结果':'Waiting for motion capture','选择动捕片段':'Select capture segment','选择全部':'Select all','开始 · 秒':'Start · seconds','结束 · 秒':'End · seconds','播放选中片段':'Preview selection','开始动作捕捉':'Capture motion','角色动画':'Character animation','等待角色动画':'Waiting for character animation','播放':'Play','暂停':'Pause','复位视角':'Reset view','播放速度':'Playback speed','动画时间':'Animation timeline','片段起点':'Segment start','片段终点':'Segment end','下载全部结果 ZIP':'Download all results · ZIP','动作参数与三套角色的 FBX 下载链接将在这里显示。':'Motion and FBX downloads will appear here.',
'工作流加载失败':'Could not load the workflow','请求失败 ({code})':'Request failed ({code})','正在捕捉人物动作…':'Capturing motion…','正在重定向角色动画…':'Retargeting characters…','正在准备动捕片段…':'Preparing capture segment…','正在生成视频…':'Generating video…','视频':'Video','动捕':'Capture','处理中':'Processing','原始视频':'Source video','动捕片段 MP4':'Captured segment MP4','关键点预览':'Keypoint preview','动作 NPZ':'Motion NPZ','结果信息':'Result metadata','BVH · 77 关节':'BVH · 77 joints','BVH · 78 节点含 Root':'BVH · 78 nodes with Root','Mixamo 动画 FBX':'Mixamo animation FBX','Unreal 动画 FBX':'Unreal animation FBX','Unity 动画 FBX':'Unity animation FBX','等待输出。':'Waiting for results.','视频已生成':'Video ready','已完成':'Completed','失败':'Failed','视频已生成。选择片段后开始动作捕捉。':'Video ready. Select a segment, then capture motion.','动作已就绪，可以播放与下载。':'Motion ready to play and download.','任务失败':'Task failed','执行失败，请查看 ComfyUI 日志。':'Execution failed. Check the ComfyUI log.','连接中断，正在重试…':'Disconnected. Reconnecting…','连接中…':'Connecting…','ComfyUI 已连接':'ComfyUI connected','执行中':'Running','视频生成':'Video generation','正在处理':'Processing','任务已中断':'Task interrupted','视频/图片上传失败':'Could not upload the video or image','排队中':'Queued','任务已提交，等待执行…':'Task submitted. Waiting to start…','正在准备生成视频…':'Preparing video generation…','未能提交任务':'Could not submit the task','请先生成或上传视频':'Generate or upload a video first','请选择非空的视频片段':'Select a non-empty video segment','未能提交动捕':'Could not submit motion capture','正在读取任务结果…':'Loading task results…','正在打包…':'Creating ZIP…','正在加载角色…':'Loading character…','角色预览加载失败，可切换角色重试。':'Could not load the preview. Switch characters to retry.','角色文件读取失败':'Could not read the character file','{seconds} 秒':'{seconds} s','{frames} 帧 · {fps} FPS':'{frames} frames · {fps} FPS','{start}–{end} 秒 · 已选 {selected} / {total} 秒':'{start}–{end} s · Selected {selected} / {total} s','排队中 · 前方 {count} 个任务':'Queued · {count} tasks ahead','{stage} · {value}/{max}':'{stage} · {value}/{max}',
'请先完成动作捕捉':'Complete motion capture first','角色动画尚未全部完成':'Character retargeting is still in progress','任务尚未完成或不存在':'The task is not complete or no longer exists','结果文件不存在':'The result file no longer exists','无法连接本机 ComfyUI，请确认 6006 服务正在运行。':'Cannot connect to local ComfyUI. Check that it is running on port 6006.'
};
export function t(key,params={}){const text=language==='en'?(english[key]||key):key;return text.replace(/\{(\w+)\}/g,(_,name)=>{const value=params[name]??'';return typeof value==='string'?(language==='en'?(english[value]||value):value):String(value);});}
export function canonical(text){return Object.keys(english).find(key=>english[key]===text)||text;}
export function applyStatic(){
    document.documentElement.lang=language==='zh'?'zh-CN':'en';document.title=t('视频生成与动作捕捉');
    for(const element of document.querySelectorAll('[data-i18n]'))element.textContent=t(element.dataset.i18n);
    for(const element of document.querySelectorAll('[data-i18n-aria]'))element.setAttribute('aria-label',t(element.dataset.i18nAria));
}
export function setLanguage(value){language=value==='en'?'en':'zh';localStorage.setItem('gem-motion-language',language);applyStatic();}
export const chineseDefaults={rules:`视频格式：
严格采用 1:1 正方形画面。

背景：
纯绿幕背景（#00FF00），地面也为纯绿色。
背景和地面必须完全同色。
不得有场景、环境或其他物体。

相机：
固定机位，正面平视，全身远景。
相机必须完全静止。
不得平移、俯仰、旋转、跟随或缩放。

构图：
整个角色及其装备在动画全程保持完整可见。
角色四周至少保留 20% 的安全边距。
相机距离足够远，能够容纳跳跃、转身和大幅度挥动。
任何部位都不得接触或超出画面边界。
优先保证全身可见，而非放大角色。

动作：
所有动作以自然、正常的速度完成。
不得使用慢动作、变速、剪切或转场。
动作流畅、协调，符合真实物理规律。
保持平衡、合理的重心分配和自然的关节运动。
不得出现身体变形、穿模或额外肢体。

光照：
均匀平坦的光照。
不得出现阴影、反射、高光或光照变化。
背景始终保持纯绿色（#00FF00）。`,action:`一个光滑、纯亮红色的三维人形模型，具有正常的人体比例。

角色没有面部特征，不穿衣服，没有头发或装饰。

全身均为亮红色，人体结构清晰，关节自然衔接。

角色双手空着，不携带装备或武器。

角色表演一段充满活力的嘻哈舞。

先进行有节奏的侧步和交替摆臂，然后衔接快速脚步、转肩和流畅的身体波浪。

角色快速旋转 360 度，蹲低身体，双臂向外展开。

随后用力起跳，平稳落地，继续快速脚步和利落的上半身扭转。

最后以一个旋转舞步结束，并摆出自信的舞蹈姿势。`};
