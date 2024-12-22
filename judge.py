import asyncio
import httpx
import os
import re
import statistics
import math
from ruamel.yaml import YAML
from collections import defaultdict
from tqdm import tqdm

MODEL_NAME = "anthropic/claude-3.5-haiku:beta"
MAX_RETRIES = 2


async def main():
    context = await load_context()
    total_calls = context["num_iterations"] * len(context["content_variants"]) * 2

    with tqdm(total=total_calls, desc="Processing") as pbar:
        async with httpx.AsyncClient() as client:
            tasks = []
            for _ in range(context["num_iterations"]):
                for variant in context["content_variants"]:
                    tasks.append(
                        process_variant(
                            client,
                            context["content_prompt"],
                            context["judge_prompt"],
                            variant,
                            context["judge_categories"],
                            pbar,
                        )
                    )
            all_scores = await asyncio.gather(*tasks)

    category_scores = group_scores_by_category(all_scores)
    category_stats, all_scores_flat = calculate_category_stats(category_scores)
    print_results(category_stats, all_scores_flat)


async def process_variant(
    client, content_prompt, judge_prompt, variant, categories, pbar
):
    content = await generate_content(client, content_prompt, variant, pbar)
    scores = await judge_content(client, judge_prompt, content, categories, pbar)
    return scores


async def load_context():
    yaml = YAML()
    with open("./context.yml", "r") as f:
        return yaml.load(f)


async def generate_content(client, system_prompt, user_prompt, pbar):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = await make_api_request(client, messages)
    pbar.update(1)
    return response["choices"][0]["message"]["content"]


async def judge_content(client, judge_prompt, content, categories, pbar):
    messages = [
        {"role": "system", "content": judge_prompt},
        {"role": "user", "content": content},
    ]

    for _ in range(MAX_RETRIES):
        try:
            response = await make_api_request(client, messages)
            response_content = response["choices"][0]["message"]["content"]
            scores = await validate_and_extract_scores(response_content, categories)
            if scores:
                pbar.update(1)
                return scores
            print(f"\n---\nINVALID JUDGING:\n{response_content}\n---\n")
            print("Missing or invalid categories in response, retrying...")
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
            "model": MODEL_NAME,
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
