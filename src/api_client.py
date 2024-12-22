import httpx
import os
import asyncio

from src.logger import Logger


class OpenRouterClient:
    def __init__(self):
        self.model_name = "anthropic/claude-3.5-haiku:beta"
        self.logger = Logger("log.yml")
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.max_retries = 3
        self.total_cost = 0

    async def request_chat_completion(self, messages, validator=None):
        params = {
            "model": self.model_name,
            "max_tokens": 2048,
            "temperature": 1.0,
            "messages": messages,
        }

        for attempt in range(self.max_retries):
            try:
                response, gen_id = await self._make_request(params)

                if validator and not await validator(response):
                    print(f"{gen_id} validation failed on attempt #{attempt + 1}")
                    continue

                return response

            except Exception as e:
                print(f"Error on attempt {attempt + 1}: {str(e)}")
                if attempt == self.max_retries - 1:
                    raise

        raise RuntimeError("Failed to get valid response after max retries")

    async def _make_request(self, params):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=params,
            )
            response.raise_for_status()
            body = response.json()

            if "error" in body:
                raise Exception(body["error"])

            content = body["choices"][0]["message"]["content"]
            details = await self._fetch_details(body["id"])

            cost = details["data"]["total_cost"]
            self.total_cost += cost

            self.logger.log(body["id"], cost, params, content)
            return content, body["id"]

    async def _fetch_details(self, generation_id: str):
        details_url = f"https://openrouter.ai/api/v1/generation?id={generation_id}"

        for _ in range(10):
            try:
                async with httpx.AsyncClient(timeout=3) as client:
                    response = await client.get(
                        details_url, headers={"Authorization": f"Bearer {self.api_key}"}
                    )
                    response.raise_for_status()
                    return response.json()
            except httpx.HTTPError:
                await asyncio.sleep(0.5)

        raise Exception("Details request timed out")
