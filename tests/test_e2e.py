import pytest
from unittest.mock import patch, AsyncMock, Mock


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
                        "content": (
                            "# creativity\n"
                            "Shows strong imagination.\n"
                            "creativity score: 80\n"
                            "---\n"
                            "# coherence\n"
                            "Clear structure.\n"
                            "coherence score: 70\n"
                            "---\n"
                            "# impact\n"
                            "Memorable message.\n"
                            "impact score: 90"
                        )
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
    mock_content.json = Mock(return_value=content_response)
    mock_content.raise_for_status = Mock()

    mock_judge = AsyncMock()
    mock_judge.json = Mock(return_value=judge_response)
    mock_judge.raise_for_status = Mock()

    mock_details = AsyncMock()
    mock_details.json = Mock(return_value=details_response)
    mock_details.raise_for_status = Mock()

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
    assert "creativity: 80.0 ± 0.0" in output_lines
    assert "coherence: 70.0 ± 0.0" in output_lines
    assert "impact: 90.0 ± 0.0" in output_lines
    assert "Final Score: 80.0 ± 2.5" in output_lines
    assert "Total Cost: $0.10" in output_lines
