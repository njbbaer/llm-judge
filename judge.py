import asyncio
import httpx
import os
import re
import statistics
import math
from ruamel.yaml import YAML
from collections import defaultdict
from tqdm import tqdm

NUM_ITERATIONS = 2
MAX_RETRIES = 1


async def main():
    context = await load_context()
    all_scores = await gather_scores(context["messages"], context["categories"])

    category_scores = group_scores_by_category(all_scores)
    category_stats, all_scores_flat = calculate_category_stats(category_scores)
    print_results(category_stats, all_scores_flat)


async def load_context():
    yaml = YAML()
    with open("./context.yml", "r") as f:
        context = yaml.load(f)

    return {
        "messages": [
            {"role": "system", "content": context["prompt"]},
            {"role": "user", "content": context["content"]},
        ],
        "categories": context["categories"],
    }


async def gather_scores(messages, expected_categories):
    with tqdm(total=NUM_ITERATIONS, desc="Processing") as pbar:
        async with httpx.AsyncClient() as client:
            tasks = [
                process_iteration(client, messages, expected_categories, pbar)
                for _ in range(NUM_ITERATIONS)
            ]
            return await asyncio.gather(*tasks)


async def process_iteration(client, messages, expected_categories, pbar):
    return await request_completion(client, messages, expected_categories, pbar)


async def request_completion(client, messages, expected_categories, pbar):
    for _ in range(MAX_RETRIES):
        try:
            body = await make_api_request(client, messages)

            if body.get("choices"):
                content = body["choices"][0]["message"]["content"]
                scores = await validate_and_extract_scores(content, expected_categories)
                if scores:
                    pbar.update(1)
                    return scores
                print("Missing or invalid categories in response, retrying...")
            else:
                print(f"Response was empty on attempt {_ + 1}, retrying...")

        except Exception as e:
            print(f"Error on attempt {_ + 1}: {str(e)}")
            if _ == MAX_RETRIES - 1:
                raise

    raise RuntimeError("Failed to get valid response after retries")


async def make_api_request(client, messages):
    response = await client.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json",
        },
        json={
            "model": "anthropic/claude-3.5-haiku:beta",
            "messages": messages,
            "max_tokens": 2048,
            "temperature": 1.0,
        },
    )
    response.raise_for_status()
    return response.json()


async def validate_and_extract_scores(text, expected_categories):
    pattern = r"<(\w+)>(.*?)<score>(\d+)</score>\s*</\1>"
    matches = re.findall(pattern, text, re.DOTALL)

    found_categories = {category for category, _, _ in matches}
    if found_categories != set(expected_categories):
        return None

    return [(category, int(score)) for category, _, score in matches]


def group_scores_by_category(all_scores):
    category_scores = defaultdict(list)
    for attempt_scores in all_scores:
        for category, score in attempt_scores:
            category_scores[category].append(score)
    return category_scores


def calculate_category_stats(category_scores):
    category_stats = {}
    all_scores_flat = []

    for category, scores in category_scores.items():
        mean, sem = calculate_stats(scores)
        category_stats[category] = (mean, sem)
        all_scores_flat.extend(scores)

    return category_stats, all_scores_flat


def calculate_stats(scores):
    mean = statistics.mean(scores)
    std_dev = statistics.stdev(scores)
    sem = std_dev / math.sqrt(len(scores))
    return mean, sem


def print_results(category_stats, all_scores_flat):
    print("\nScores by Category:")
    for category, (mean, sem) in sorted(category_stats.items()):
        print(f"{category}: {mean:.1f} ± {sem:.1f}")

    overall_mean, overall_sem = calculate_stats(all_scores_flat)
    print(f"\nFinal Score: {overall_mean:.1f} ± {overall_sem:.1f}")


if __name__ == "__main__":
    asyncio.run(main())
