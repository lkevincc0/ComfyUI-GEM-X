"""Build the panel API template from the user's saved MiniMax H3 workflow."""
import argparse
import copy
import json
from pathlib import Path


def convert(workflow):
    nodes = {n['id']: n for n in workflow['nodes']}
    generator = nodes[140]
    sub = next(s for s in workflow['definitions']['subgraphs'] if s['id'] == generator['type'])
    top_links = {l[0]: l for l in workflow['links']}
    links = {l['id']: l for l in sub['links']}
    graph = {}

    def top_source(link_id):
        l = top_links[link_id]
        return [f'140:{130}' if l[1] == 140 else str(l[1]), l[2]]

    def export(n, prefix=''):
        inputs = copy.deepcopy(n.get('widgets_values_named', {}))
        for key in ['control_after_generate', 'fixed']:
            inputs.pop(key, None)
        for socket in n.get('inputs', []):
            link = socket.get('link')
            if link is None:
                continue
            if not prefix:
                inputs[socket['name']] = top_source(link)
                continue
            edge = links[link]
            if edge['origin_id'] != -10:
                inputs[socket['name']] = [f"140:{edge['origin_id']}", edge['origin_slot']]
            else:
                name = sub['inputs'][edge['origin_slot']]['name']
                top_socket = next((s for s in generator['inputs'] if s['name'] == name), None)
                if top_socket and top_socket.get('link') is not None:
                    inputs[socket['name']] = top_source(top_socket['link'])
                elif name in generator['widgets_values_named']:
                    inputs[socket['name']] = generator['widgets_values_named'][name]
        if n['type'] == 'Preview3DAdvanced':
            inputs['width'] = inputs['height'] = 1024
            inputs['viewport_state'] = {'image': '', 'mask': '', 'normal': '', 'recording': '',
                'model_3d_info': n['properties'].get('Scene Config', {}).get('models', [])}
        graph[prefix + str(n['id'])] = {'class_type': n['type'], 'inputs': inputs}

    for n in workflow['nodes']:
        if n['type'] not in ['Note', 'MarkdownNote', generator['type']]:
            export(n)
    for n in sub['nodes']:
        export(n, '140:')
    graph['144']['inputs'].update(postprocess=False, preview='keypoints', backend='onnx')
    graph['157']['inputs'].update(filename_prefix='video/MotionStudio', format='mp4', **{'format.codec': 'h264', 'codec': 'h264'})
    prompt = generator['widgets_values_named']['prompt']
    return {'prompt': graph, 'defaults': {'prompt': prompt, 'duration': generator['widgets_values_named']['value_1'],
            'megapixels': nodes[115]['widgets_values_named']['megapixels']},
            'source': 'MinimaxH3 to Motion.json'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('workflow', type=Path)
    parser.add_argument('--output', type=Path, default=Path('comfyui_gem_x/panel/workflow.json'))
    args = parser.parse_args()
    args.output.write_text(json.dumps(convert(json.loads(args.workflow.read_text())), ensure_ascii=False, indent=2))
