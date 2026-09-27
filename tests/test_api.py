"""API boundary tests; no background model execution or shared user job writes."""
from fastapi.testclient import TestClient
from climb_app.server import app

client = TestClient(app)


def test_health_and_ui():
    assert client.get('/api/health').status_code == 200
    assert 'CRUX' in client.get('/').text


def test_missing_job_and_path_traversal():
    assert client.get('/api/jobs/missing').status_code == 404
    assert client.get('/api/jobs/missing/files/../../requirements.txt').status_code == 404


def test_invalid_upload_type():
    response = client.post('/api/upload', files={'file': ('evil.html', b'<script/>', 'text/html')})
    assert response.status_code == 415


def test_cross_origin_mutation_rejected():
    response = client.post('/api/samples/body-trajectory-input.mp4', headers={'Origin': 'https://other.example'})
    assert response.status_code == 403


def test_invalid_config_rejected():
    response = client.post('/api/jobs/missing/analyze', json={'static_speed': -1})
    assert response.status_code == 422
