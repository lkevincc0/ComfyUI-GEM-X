"""Serve Motion Studio on 6008 and proxy its API to local ComfyUI on 6006."""
import argparse
import asyncio
import io
import json
import os
import uuid
import zipfile
from pathlib import Path

from aiohttp import ClientSession, ClientTimeout, WSMsgType, web

COMFY_ROOT = Path('/ComfyUI')
BVH_LOCKS = web.AppKey('bvh_locks', dict)
PANEL = Path(__file__).resolve().parents[1] / 'comfyui_gem_x' / 'panel'
REQUEST_HEADERS = {'content-type', 'range', 'if-range', 'accept', 'accept-encoding'}
SESSION = web.AppKey('session', ClientSession)
RESPONSE_HEADERS = {'content-type', 'content-disposition', 'content-range', 'accept-ranges', 'cache-control', 'etag', 'last-modified', 'content-encoding'}


def make_app(upstream='http://127.0.0.1:6006'):
    app = web.Application(client_max_size=1024**3)
    app[BVH_LOCKS] = {}

    async def session_context(app):
        async with ClientSession(timeout=ClientTimeout(total=None, sock_connect=30), auto_decompress=False) as session:
            app[SESSION] = session
            yield
    app.cleanup_ctx.append(session_context)

    async def proxy(request):
        target = upstream + request.rel_url.raw_path_qs
        try:
            async with app[SESSION].request(request.method, target,
                    headers={k: v for k, v in request.headers.items() if k.lower() in REQUEST_HEADERS},
                    data=request.content if request.can_read_body else None) as result:
                response = web.StreamResponse(status=result.status,
                    headers={k: v for k, v in result.headers.items() if k.lower() in RESPONSE_HEADERS})
                await response.prepare(request)
                async for chunk in result.content.iter_chunked(256 * 1024):
                    await response.write(chunk)
                await response.write_eof()
                return response
        except (OSError, asyncio.TimeoutError) as exc:
            raise web.HTTPBadGateway(text='无法连接本机 ComfyUI，请确认 6006 服务正在运行。') from exc

    async def websocket(request):
        target = upstream.replace('http://', 'ws://').replace('https://', 'wss://') + request.rel_url.raw_path_qs
        try:
            remote = await app[SESSION].ws_connect(target, heartbeat=30)
        except (OSError, asyncio.TimeoutError) as exc:
            raise web.HTTPBadGateway(text='ComfyUI 未连接') from exc
        local = web.WebSocketResponse(heartbeat=30)
        await local.prepare(request)

        async def pipe(source, dest):
            async for message in source:
                if message.type == WSMsgType.TEXT:
                    await dest.send_str(message.data)
                elif message.type == WSMsgType.BINARY:
                    await dest.send_bytes(message.data)
                elif message.type == WSMsgType.ERROR:
                    break
        tasks = [asyncio.create_task(pipe(local, remote)), asyncio.create_task(pipe(remote, local))]
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await remote.close()
            await local.close()
        return local

    def file_path(file):
        kind = file.get('type', 'output')
        if kind not in ('output', 'temp', 'input'):
            raise web.HTTPBadRequest(text='无效文件目录')
        root = (COMFY_ROOT / kind).resolve()
        path = (root / file.get('subfolder', '') / file['filename']).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise web.HTTPNotFound(text='结果文件不存在')
        return path

    async def task_entry(prompt_id):
        try:
            uuid.UUID(prompt_id)
        except ValueError as exc:
            raise web.HTTPBadRequest(text='无效任务编号') from exc
        async with app[SESSION].get(upstream + '/history/' + prompt_id) as response:
            data = await response.json()
        if prompt_id not in data:
            raise web.HTTPNotFound(text='任务尚未完成或不存在')
        return data[prompt_id]

    async def ensure_bvh(entry):
        result = entry.get('outputs', {}).get('144', {}).get('gem_result', [])
        if not result:
            raise web.HTTPBadRequest(text='请先完成动作捕捉')
        motion = file_path(result[0]['files']['motion'])
        lock = app[BVH_LOCKS].setdefault(str(motion), asyncio.Lock())
        async with lock:
            if not all((motion.parent / name).is_file() for name in ('motion_77j.bvh', 'motion_78j.bvh')):
                root = Path(__file__).resolve().parents[1]
                env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4')
                system_cpp = '/usr/lib/x86_64-linux-gnu/libstdc++.so.6'
                if Path(system_cpp).is_file():
                    env['LD_PRELOAD'] = system_cpp
                with (motion.parent / 'bvh_export.log').open('w') as log:
                    process = await asyncio.create_subprocess_exec(str(root / '.venv/bin/python'),
                        str(root / 'scripts/comfyui_export_bvh.py'), '--motion', str(motion),
                        '--models-dir', str(COMFY_ROOT / 'models/gem_x'), cwd=root, env=env,
                        stdout=log, stderr=asyncio.subprocess.STDOUT)
                    try:
                        await asyncio.wait_for(process.wait(), 180)
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        process.kill()
                        await process.wait()
                        raise
                if process.returncode:
                    tail = (motion.parent / 'bvh_export.log').read_text(errors='replace')[-1800:]
                    raise web.HTTPInternalServerError(text='BVH 导出失败：' + tail)
        subfolder = motion.parent.relative_to((COMFY_ROOT / 'output').resolve()).as_posix()
        return {name: {'filename': f'motion_{name}.bvh', 'subfolder': subfolder, 'type': 'output'}
                for name in ('77j', '78j')}

    async def export_bvh(request):
        entry = await task_entry((await request.json())['prompt_id'])
        return web.json_response({'files': await ensure_bvh(entry)})

    async def download_one(request):
        prompt_id, kind = request.match_info['prompt_id'], request.match_info['kind']
        entry = await task_entry(prompt_id)
        outputs = entry.get('outputs', {})
        result = outputs.get('144', {}).get('gem_result', [{}])[0]
        files = result.get('files', {})
        video = outputs.get('157', {}).get('images', [None])[0]
        source = entry['prompt'][3].get('motion_studio', {}).get('source_video', video)
        if kind in ('source', 'clip'):
            file = source if kind == 'source' else video
        elif kind in ('preview', 'motion', 'manifest'):
            file = files.get(kind)
        elif kind in ('mixamo', 'unreal', 'unity'):
            node = {'mixamo': '148', 'unreal': '150', 'unity': '152'}[kind]
            file = outputs.get(node, {}).get('3d', [None])[0]
        elif kind in ('bvh77', 'bvh78'):
            file = (await ensure_bvh(entry))['77j' if kind == 'bvh77' else '78j']
        else:
            raise web.HTTPNotFound()
        if not file:
            raise web.HTTPNotFound(text='结果文件尚未生成')
        path = file_path(file)
        prefix = {'source': 'source_video', 'clip': 'captured_segment', 'preview': 'keypoints_preview',
            'motion': 'motion', 'manifest': 'manifest', 'mixamo': 'Mixamo_motion', 'unreal': 'Unreal_motion',
            'unity': 'Unity_motion', 'bvh77': 'motion_77j', 'bvh78': 'motion_78j'}[kind]
        filename = prefix + '_' + prompt_id[:8] + path.suffix
        return web.FileResponse(path, headers={'Content-Disposition': f'attachment; filename="{filename}"'})

    async def download_all(request):
        body = await request.json()
        prompt_id = body['prompt_id']
        entry = await task_entry(prompt_id)
        outputs = entry.get('outputs', {})
        if not all(node in outputs for node in ('144', '148', '150', '152')):
            raise web.HTTPBadRequest(text='角色动画尚未全部完成')
        bvh = await ensure_bvh(entry)
        result = outputs['144']['gem_result'][0]
        files = []
        metadata = entry['prompt'][3].get('motion_studio', {})
        generation = entry
        parent_id = metadata.get('generation_prompt_id')
        if parent_id:
            try:
                generation = await task_entry(parent_id)
            except web.HTTPNotFound:
                generation = entry
        generation_graph = generation['prompt'][2]
        source = metadata.get('source_video')
        if source:
            files.append((file_path(source), '01_video/source_video' + Path(source['filename']).suffix))
        video = outputs.get('157', {}).get('images', [])
        if video:
            files.append((file_path(video[0]), '01_video/captured_segment.mp4' if source else '01_video/generated_video.mp4'))
        for key, name in [('preview', 'keypoints_preview.mp4'), ('motion', 'motion.npz'), ('manifest', 'manifest.json')]:
            if key in result['files']:
                files.append((file_path(result['files'][key]), '02_motion/' + name))
        motion = file_path(result['files']['motion'])
        for name in ('hpe_results.pt', 'bvh_info.json'):
            if (motion.parent / name).is_file():
                files.append((motion.parent / name, '02_motion/' + name))
        for key, file in bvh.items():
            files.append((file_path(file), '02_motion/' + file['filename']))
        for node, name in [('148', 'Mixamo'), ('150', 'Unreal'), ('152', 'Unity')]:
            files.append((file_path(outputs[node]['3d'][0]), f'03_characters/{name}/animation.fbx'))

        def archive():
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED, compresslevel=3) as pack:
                for path, name in files:
                    pack.write(path, name)
                pack.writestr('04_settings/workflow_api.json', json.dumps(entry['prompt'][2], ensure_ascii=False, indent=2))
                if '140:131' in generation_graph:
                    pack.writestr('04_settings/generation_workflow_api.json', json.dumps(generation_graph, ensure_ascii=False, indent=2))
                    inputs = generation_graph['140:131']['inputs']
                    pack.writestr('04_settings/prompts.txt', inputs['prompt'])
                    def values(node):
                        return generation_graph.get(node, {}).get('inputs', {})
                    turbo = values('140:139').get('value', True)
                    settings = {'seed': values('140:129').get('noise_seed'),
                        'requested_duration_seconds': values('140:133').get('value'),
                        'turbo': turbo, 'steps': values('140:138' if turbo else '140:137').get('value'),
                        'video': generation.get('outputs', {}).get('157', {}).get('images', [{}])[0].get('metadata', {})}
                    pack.writestr('04_settings/generation_settings.json', json.dumps(settings, ensure_ascii=False, indent=2))
                pack.writestr('04_settings/task.json', json.dumps({'prompt_id': prompt_id, 'fps': result['fps'],
                    'frames': result['frames'], **metadata}, ensure_ascii=False, indent=2))
                pack.writestr('README.txt', '01_video: 原始视频及用于动捕的片段。\n02_motion: BVH、NPZ、原始 PT、关键点预览及元数据。\n03_characters: Mixamo / Unreal / Unity 重定向后的 FBX 动画。\n04_settings: 本次任务参数及 API 工作流。\n\nBVH 单位为米，Y 轴向上，保留原始帧率。Blender 导入时缩放设为 1。\n默认使用 motion_77j.bvh；motion_78j.bvh 含虚拟 Root。\n')
            return buffer.getvalue()
        data = await asyncio.to_thread(archive)
        return web.Response(body=data, content_type='application/zip',
            headers={'Content-Disposition': f'attachment; filename="motion_{prompt_id[:8]}.zip"'})

    async def index(request):
        return web.FileResponse(PANEL / 'index.html', headers={'Cache-Control': 'no-cache'})

    app.router.add_get('/', index)
    app.router.add_get('/ws', websocket)
    app.router.add_post('/prompt', proxy)
    app.router.add_post('/upload/image', proxy)
    app.router.add_get('/queue', proxy)
    app.router.add_get('/history/{prompt_id}', proxy)
    app.router.add_get('/view', proxy)
    app.router.add_post('/export-bvh', export_bvh)
    app.router.add_post('/download-all', download_all)
    app.router.add_get('/download/{prompt_id}/{kind}', download_one)
    app.router.add_static('/', PANEL, show_index=False)
    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=6008)
    parser.add_argument('--listen', default='0.0.0.0')
    args = parser.parse_args()
    web.run_app(make_app(), host=args.listen, port=args.port, access_log=None)
