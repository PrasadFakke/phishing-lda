from fastapi.testclient import TestClient
from app.main import app

def test_overview_and_dataset_route():
    client=TestClient(app)
    assert client.get('/').status_code==200
    response=client.get('/api/dataset')
    assert response.status_code==200 and response.json()['rows']==11054
