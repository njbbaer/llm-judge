import httpx
import os
import asyncio

from src.logger import Logger
from src.config import Config


class OpenRouterClient:
    def __init__(self, config: Config):
        self.config = config
        self.logger = Logger("log.yml")
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.total_cost = 0

    async def request_chat_completion(self, messages, temperature, validator=None):
        params = {
            "model": self.config.model,
            "max_tokens": 2048,
            "temperature": temperature,
            "messages": messages,
        }

        for attempt in range(self.config.max_retries):
            try:
                response, gen_id = await self._make_request(params)

                if not response:
                    print(f"Empty response on attempt #{attempt + 1}: {gen_id}")
                    continue

                if validator and not await validator(response):
                    print(f"Validation failed on attempt #{attempt + 1}: {gen_id}")
                    continue

                return response

            except httpx.HTTPError as e:
                print(f"HTTP error on attempt {attempt + 1}: {str(e)}")
                if attempt == self.config.max_retries - 1:
                    raise

        raise RuntimeError("Failed to get a valid response after max retries")

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
