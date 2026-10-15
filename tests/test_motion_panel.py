"""Exercise local ComfyUI HTTP/WebSocket forwarding without GPU inference."""
import importlib.util
import io
import json
import tempfile
import zipfile
from pathlib import Path
import unittest

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

MODULE = Path(__file__).resolve().parents[1] / 'scripts' / 'motion_panel_server.py'
spec = importlib.util.spec_from_file_location('motion_panel_server', MODULE)
panel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(panel)


class PanelServerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.original_root = panel.COMFY_ROOT
        panel.COMFY_ROOT = Path(self.temp.name)
        self.entry = None
        self.task_id = '23264788-63bc-4ade-b8c1-0b1cc64018e5'
        self.received = None
        upstream = web.Application()

        async def prompt(request):
            self.received = await request.json()
            return web.json_response({'prompt_id': 'panel-test'})

        async def view(request):
            return web.Response(status=206, body=b'video-bytes', headers={
                'Content-Type': 'video/mp4', 'Content-Range': 'bytes 0-10/100',
                'Accept-Ranges': 'bytes'})

        async def ws(request):
            response = web.WebSocketResponse()
            await response.prepare(request)
            await response.send_json({'type': 'status', 'client': request.query.get('clientId')})
            async for message in response:
                await response.send_str(message.data)
            return response

        async def history(request):
            return web.json_response({self.task_id: self.entry} if self.entry else {})

        upstream.router.add_get('/history/{id}', history)
        upstream.router.add_post('/prompt', prompt)
        upstream.router.add_get('/view', view)
        upstream.router.add_get('/ws', ws)
        self.upstream = TestServer(upstream)
        await self.upstream.start_server()
        self.client = TestClient(TestServer(panel.make_app(str(self.upstream.make_url('')).rstrip('/'))))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()
        await self.upstream.close()
        panel.COMFY_ROOT = self.original_root
        self.temp.cleanup()

    async def test_panel_is_open_and_unrelated_routes_are_not_proxied(self):
        response = await self.client.get('/')
        self.assertEqual(response.status, 200)
        self.assertIn('约束提示词', await response.text())
        response = await self.client.get('/history')
        self.assertEqual(response.status, 404)
        response = await self.client.post('/interrupt')
        self.assertEqual(response.status, 405)

    def complete_task(self):
        root = panel.COMFY_ROOT
        for folder in ('output/gem_x/test', 'output/video', 'temp', 'input'):
            (root / folder).mkdir(parents=True, exist_ok=True)
        def file(kind, subfolder, name):
            (root / kind / subfolder / name).write_bytes(name.encode())
            return {'filename': name, 'subfolder': subfolder, 'type': kind}
        capture = {key: file('output', 'gem_x/test', name) for key, name in
                   [('motion', 'motion.npz'), ('manifest', 'manifest.json'), ('preview', 'preview.mp4')]}
        for name in ('motion_77j.bvh', 'motion_78j.bvh', 'hpe_results.pt'):
            file('output', 'gem_x/test', name)
        outputs = {'144': {'gem_result': [{'files': capture, 'fps': 24, 'frames': 96}]},
                   '157': {'images': [file('output', 'video', 'clip.mp4')]}}
        for node in ('148', '150', '152'):
            outputs[node] = {'3d': [file('temp', '', node + '.fbx')]}
        source = file('input', '', 'source.mp4')
        self.entry = {'outputs': outputs, 'prompt': [0, self.task_id, {},
                      {'motion_studio': {'source_video': source}}, []]}

    async def test_zip_contains_original_clip_bvh_and_all_three_characters(self):
        self.complete_task()
        response = await self.client.post('/download-all', json={'prompt_id': self.task_id})
        self.assertEqual(response.status, 200)
        with zipfile.ZipFile(io.BytesIO(await response.read())) as archive:
            self.assertIsNone(archive.testzip())
            for name in ('01_video/source_video.mp4', '01_video/captured_segment.mp4',
                         '02_motion/motion_77j.bvh', '02_motion/motion_78j.bvh', '02_motion/hpe_results.pt',
                         '03_characters/Mixamo/animation.fbx', '03_characters/Unreal/animation.fbx',
                         '03_characters/Unity/animation.fbx', '04_settings/workflow_api.json'):
                self.assertIn(name, archive.namelist())

    async def test_individual_fbx_has_clear_download_filename(self):
        self.complete_task()
        response = await self.client.get('/download/' + self.task_id + '/mixamo')
        self.assertEqual(response.status, 200)
        self.assertIn('Mixamo_motion_23264788.fbx', response.headers['Content-Disposition'])
        self.assertEqual(await response.read(), b'148.fbx')

    async def test_zip_rejects_file_outside_comfy_output(self):
        self.complete_task()
        outside = panel.COMFY_ROOT / 'outside.txt'
        outside.write_text('not an output')
        self.entry['outputs']['144']['gem_result'][0]['files']['motion']['filename'] = '../../../outside.txt'
        response = await self.client.post('/download-all', json={'prompt_id': self.task_id})
        self.assertEqual(response.status, 404)

    async def test_submitted_prompt_reaches_local_comfyui_unchanged(self):
        payload = {'client_id': 'our-client', 'prompt': {'144': {'inputs': {'postprocess': False}}}}
        response = await self.client.post('/prompt', json=payload)
        self.assertEqual(await response.json(), {'prompt_id': 'panel-test'})
        self.assertEqual(self.received, payload)

    async def test_video_range_response_is_preserved(self):
        response = await self.client.get('/view?filename=preview.mp4&type=output', headers={'Range': 'bytes=0-10'})
        self.assertEqual(response.status, 206)
        self.assertEqual(response.headers['Content-Range'], 'bytes 0-10/100')
        self.assertEqual(await response.read(), b'video-bytes')

    async def test_websocket_proxies_both_directions_and_client_id(self):
        async with self.client.ws_connect('/ws?clientId=our-client') as socket:
            self.assertEqual((await socket.receive_json())['client'], 'our-client')
            await socket.send_str('progress-test')
            self.assertEqual((await socket.receive()).data, 'progress-test')


if __name__ == '__main__':
    unittest.main()
