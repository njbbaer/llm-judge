import asyncio
import re
import statistics
import math
from tqdm import tqdm
from collections import defaultdict
from typing import List, Dict, Any

from src.api_client import OpenRouterClient
from src.config import Config


async def main(data: Dict = None):
    config = await Config.load(data)
    client = OpenRouterClient(config.model)

    with tqdm(total=config.total_calls, desc="Processing") as pbar:
        tasks = [
            process_variant(client, config, variant, pbar)
            for _ in range(config.iterations)
            for variant in config.content_variants
        ]

        if config.warm_cache and tasks:
            first_result = await tasks[0]
            remaining_results = await asyncio.gather(*tasks[1:])
            all_scores = [*remaining_results, first_result]
        else:
            all_scores = await asyncio.gather(*tasks)

    category_scores = group_scores_by_category(all_scores)
    category_stats, all_scores_flat = calculate_category_stats(category_scores)
    print_results(category_stats, all_scores_flat)
    print(f"Total Cost: ${client.total_cost:.2f}")


async def process_variant(
    client: OpenRouterClient, config: Config, variant: str, pbar: tqdm
) -> List[tuple]:
    content = await generate_content(client, config.content_prompt, variant, pbar)
    scores = await judge_content(
        client, config.judge_prompt, content, variant, config.judge_categories, pbar
    )
    return scores


def build_messages(system_prompt: str, user_prompt: str) -> List[Dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {
                        "type": "ephemeral",
                    },
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_prompt,
                }
            ],
        },
    ]


async def generate_content(
    client: OpenRouterClient, system_prompt: str, user_prompt: str, pbar: tqdm
) -> str:
    messages = build_messages(system_prompt, user_prompt)
    response_content = await client.request_chat_completion(messages, temperature=1.0)
    stripped_content = strip_tag(response_content, ["playwright", "think"])
    pbar.update(1)
    return stripped_content


def strip_tag(content: str, tags: List[str]) -> str:
    for tag in tags:
        content = re.sub(rf"<{tag}.*?>.*?</{tag}>", "", content, flags=re.DOTALL)
    content = re.sub(r"\n{3,}", "\n\n", content)
    return content.strip()


async def judge_content(
    client: OpenRouterClient,
    judge_prompt: str,
    content: str,
    variant: str,
    categories: List[str],
    pbar: tqdm,
) -> List[tuple]:
    async def validate_response(text: str) -> bool:
        scores = await validate_and_extract_scores(text, categories)
        return scores is not None

    messages = build_messages(judge_prompt, content)
    messages.insert(
        1, {"role": "assistant", "content": [{"type": "text", "text": variant}]}
    )

    response_content = await client.request_chat_completion(
        messages, temperature=0.0, validator=validate_response
    )
    scores = await validate_and_extract_scores(response_content, categories)
    pbar.update(1)
    return scores


async def validate_and_extract_scores(text: str, expected_categories: List[str]):
    pattern = r"<(\w+)>(.*?)<score>(\d+)</score>\s*</\1>"
    matches = re.findall(pattern, text, re.DOTALL)

    found_categories = {category for category, _, _ in matches}
    if found_categories != set(expected_categories):
        return None

    return [(category, int(score)) for category, _, score in matches]


def group_scores_by_category(all_scores: List[List[tuple]]) -> Dict[str, List[int]]:
    category_scores = defaultdict(list)
    for attempt_scores in all_scores:
        for category, score in attempt_scores:
            category_scores[category].append(score)
    return category_scores


def calculate_category_stats(category_scores: Dict[str, List[int]]):
    category_stats = {}
    all_scores_flat = []

    for category, scores in category_scores.items():
        mean, sem = calculate_stats(scores)
        category_stats[category] = (mean, sem)
        all_scores_flat.extend(scores)

    return category_stats, all_scores_flat


def calculate_stats(scores: List[int]) -> tuple[float, float]:
    mean = statistics.mean(scores)
    if len(scores) <= 1:
        return mean, 0
    std_dev = statistics.stdev(scores)
    sem = std_dev / math.sqrt(len(scores))
    return mean, sem


def print_results(category_stats: Dict[str, tuple], all_scores_flat: List[int]):
    for category, (mean, sem) in sorted(category_stats.items()):
        print(f"{category}: {mean:.1f} ± {sem:.1f}")

    overall_mean, overall_sem = calculate_stats(all_scores_flat)
    print(f"\nFinal Score: {overall_mean:.1f} ± {overall_sem:.1f}")


if __name__ == "__main__":
    asyncio.run(main())
