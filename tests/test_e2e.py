import pytest
from unittest.mock import patch, AsyncMock


@pytest.fixture
def context():
    return {
        "model": "test-model",
        "iterations": 2,
        "content_variants": ["prompt variant 1", "prompt variant 2"],
        "content_prompt": "Write something creative",
        "judge_prompt": "Rate the content",
        "judge_categories": ["creativity", "coherence", "impact"],
        "warm_cache": False,
    }


@pytest.fixture
def mock_responses():
    def get_content_response():
        return {
            "id": "0",
            "choices": [{"message": {"content": "This is generated content"}}],
        }

    def get_judge_response():
        return {
            "id": "1",
            "choices": [
                {
                    "message": {
                        "content": """
                        <creativity>Strong imaginative elements<score>8</score></creativity>
                        <coherence>Well structured<score>7</score></coherence>
                        <impact>Memorable message<score>9</score></impact>
                    """
                    }
                }
            ],
        }

    def get_details_response():
        return {"data": {"total_cost": 0.0125}}

    return get_content_response(), get_judge_response(), get_details_response()


@pytest.mark.asyncio
async def test_pipeline(context, mock_responses, capsys):
    content_response, judge_response, details_response = mock_responses

    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client

    mock_content = AsyncMock()
    mock_content.json.return_value = content_response
    mock_content.raise_for_status = AsyncMock()

    mock_judge = AsyncMock()
    mock_judge.json.return_value = judge_response
    mock_judge.raise_for_status = AsyncMock()

    mock_details = AsyncMock()
    mock_details.json.return_value = details_response
    mock_details.raise_for_status = AsyncMock()

    mock_client.post.side_effect = [mock_content, mock_judge] * 4
    mock_client.get.return_value = mock_details

    with patch("httpx.AsyncClient", return_value=mock_client):
        from judge import main

        await main(context)

        assert_api_calls(mock_client)
        assert_output(capsys.readouterr().out.strip().split("\n"))


def assert_api_calls(mock_client):
    assert mock_client.post.call_count == 8
    assert mock_client.get.call_count == 8

    for call in mock_client.post.call_args_list:
        url = call.args[0]
        assert url == "https://openrouter.ai/api/v1/chat/completions"

        json_data = call.kwargs["json"]
        assert "model" in json_data
        assert "messages" in json_data
        assert "temperature" in json_data


def assert_output(output_lines):
    expected_scores = {
        "creativity": "8.0 ± 0.0",
        "coherence": "7.0 ± 0.0",
        "impact": "9.0 ± 0.0",
        "final": "8.0 ± 0.2",
    }

    for category, score in expected_scores.items():
        line = [l for l in output_lines if score in l][0]
        if category == "final":
            assert f"Final Score: {score}" in line
        else:
            assert f"{category}: {score}" in line

    assert "Total Cost: $0.10" in output_lines
