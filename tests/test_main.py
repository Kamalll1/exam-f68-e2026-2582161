import pytest
from httpx import AsyncClient
from pydantic import ValidationError

from app.main import app, predict_endpoint, PredictionRequest

@pytest.mark.anyio
def test_predict_success_basic():
    data = PredictionRequest(features=[3.5, 1.2, 4.9])
    resp = predict_endpoint(data)
    assert resp == {"predictions": [7.0, 2.4, 9.8]}


@pytest.mark.anyio
def test_predict_success_string():
    data = PredictionRequest(features=["3.5", "1.2", "4.9"])
    resp = predict_endpoint(data)
    assert resp == {"predictions": [7.0, 2.4, 9.8]}

@pytest.mark.anyio
def test_predict_invalid_data():
    with pytest.raises(ValidationError):
        PredictionRequest(features=["a", "b", "c"])


@pytest.mark.anyio
def test_predict_empty_data_invalid():
    with pytest.raises(ValidationError):
        PredictionRequest(features=[])


@pytest.mark.anyio
async def test_favicon_not_found():
    async with AsyncClient(app=app, base_url="http://test") as client:
        resp = await client.get("/favicon.ico")
    assert resp.status_code == 404