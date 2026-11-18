"""
umi.tag_tracking.review and the HTTP handler of the review viewer
(scripts_tag_tracking/12_review_viewer.py) with a stand-in session.

python tests/test_review.py
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib.util
import http.server
import json
import pathlib
import tempfile
import threading
import urllib.request
from umi.tag_tracking.review import apply_review, check_review, empty_review, span_key

ROOT = pathlib.Path(__file__).parent.parent


def annotation():
    return {'annotation_source': 'model', 'annotation_model': 'm',
        'tasks': [{'task': 'pack screwdriver', 'outcome': 'success', 'completed_at_s': 9.0, 'actions': ['grasp driver']}],
        'events': [{'arm': 'right', 'start_s': 0.0, 'end_s': 2.0, 'verb': 'reset', 'object': '', 'destination': '',
                    'measured_phase': 'reset'},
                   {'arm': 'right', 'start_s': 2.0, 'end_s': 6.0, 'verb': 'grasp driver', 'object': 'driver',
                    'destination': '', 'measured_phase': 'approach'}],
        'key_events': [{'t_s': 6.0, 'kind': 'contact'}, {'t_s': 9.0, 'kind': 'subgoal_complete'}]}


def test_apply_review():
    a = annotation()
    assert apply_review(a, None) == a and apply_review(a, empty_review()) == a
    r = empty_review()
    r.update({'reviewed': True, 'reviewer': 'x', 'task': {'task': ' put the blue screwdriver in the kit ', 'outcome': 'failure'},
        'events': {span_key(a['events'][1]): {'verb': 'grasp blue screwdriver', 'object': '', 'destination': ''},
                   span_key(a['events'][0]): {'verb': 'wave', 'object': '', 'destination': ''}}})
    b = apply_review(a, r)
    t = b['tasks'][0]
    assert t['task'] == 'put the blue screwdriver in the kit' and t['outcome'] == 'failure' and t['completed_at_s'] is None
    assert [k['kind'] for k in b['key_events']] == ['contact']
    assert b['events'][1]['verb'] == 'grasp blue screwdriver' and b['events'][1]['object'] == 'driver'
    assert b['events'][0]['verb'] == 'reset'  # reset spans are measured, not editable
    assert t['actions'] == ['grasp blue screwdriver'] and b['annotation_source'] == 'model_reviewed'
    assert a['tasks'][0]['task'] == 'pack screwdriver'  # input untouched
    try:
        check_review({'task': {'outcome': 'great'}})
        assert False
    except AssertionError as e:
        assert 'outcome' in str(e)


def load_viewer():
    spec = importlib.util.spec_from_file_location('viewer', ROOT / 'scripts_tag_tracking' / '12_review_viewer.py')
    module = importlib.util.module_from_spec(spec)
    cwd = os.getcwd()
    spec.loader.exec_module(module)
    os.chdir(cwd)
    return module


class FakeSession:
    def __init__(self, d):
        self.proxies = pathlib.Path(d)
        self.reviews = dict()

    def list(self):
        return {'session': 'task1', 'episodes': []}

    def write_review(self, head, number, review):
        self.reviews[(head, number)] = review
        return {'saved': True}

    def declarations(self):
        return {'path': None}


def test_server():
    viewer = load_viewer()
    with tempfile.TemporaryDirectory() as d:
        pathlib.Path(d, 'clip.mp4').write_bytes(bytes(range(256)) * 4)
        state = FakeSession(d)
        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), viewer.handler_for(state))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            page = urllib.request.urlopen(base + '/').read().decode()
            assert '<canvas id="timeline"' in page
            assert json.load(urllib.request.urlopen(base + '/api/episodes'))['session'] == 'task1'
            r = urllib.request.urlopen(urllib.request.Request(base + '/video/clip.mp4', headers={'Range': 'bytes=10-19'}))
            assert r.status == 206 and r.read() == bytes(range(10, 20)) and r.headers['Content-Range'] == 'bytes 10-19/1024'
            r = urllib.request.urlopen(urllib.request.Request(base + '/video/clip.mp4', headers={'Range': 'bytes=1000-'}))
            assert r.read() == bytes(range(232, 256))
            assert len(urllib.request.urlopen(base + '/video/clip.mp4').read()) == 1024
            # no path tricks: only file names inside the clip folder
            try:
                urllib.request.urlopen(base + '/video/..%2F..%2Fetc%2Fpasswd')
                assert False
            except urllib.error.HTTPError as e:
                assert e.code == 404
            body = json.dumps({'reviewed': True}).encode()
            req = urllib.request.Request(base + '/api/review?head=h&n=3', data=body, headers={'Content-Type': 'application/json'})
            assert json.load(urllib.request.urlopen(req)) == {'saved': True} and state.reviews[('h', 3)] == {'reviewed': True}
        finally:
            server.shutdown()


if __name__ == '__main__':
    test_apply_review()
    test_server()
    print("all passed")
