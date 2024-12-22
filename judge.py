import asyncio
import re
import statistics
import math
from tqdm import tqdm
from collections import defaultdict

from src.yaml_config import yaml
from src.api_client import OpenRouterClient


async def main():
    context = await load_context()
    total_calls = context["num_iterations"] * len(context["content_variants"]) * 2
    client = OpenRouterClient()

    with tqdm(total=total_calls, desc="Processing") as pbar:
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
    print(f"Total Cost: ${client.total_cost:.2f}")


async def process_variant(
    client, content_prompt, judge_prompt, variant, categories, pbar
):
    content = await generate_content(client, content_prompt, variant, pbar)
    scores = await judge_content(client, judge_prompt, content, categories, pbar)
    return scores


async def load_context():
    with open("./context.yml", "r") as f:
        return yaml.load(f)


async def generate_content(client, system_prompt, user_prompt, pbar):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response_content = await client.request_chat_completion(messages)
    pbar.update(1)
    return response_content


async def judge_content(client, judge_prompt, content, categories, pbar):
    messages = [
        {"role": "system", "content": judge_prompt},
        {"role": "user", "content": content},
    ]

    async def validate_response(text):
        scores = await validate_and_extract_scores(text, categories)
        return scores is not None

    response_content = await client.request_chat_completion(messages, validate_response)
    scores = await validate_and_extract_scores(response_content, categories)
    pbar.update(1)
    return scores


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
    for category, (mean, sem) in sorted(category_stats.items()):
        print(f"{category}: {mean:.1f} ± {sem:.1f}")

    overall_mean, overall_sem = calculate_stats(all_scores_flat)
    print(f"\nFinal Score: {overall_mean:.1f} ± {overall_sem:.1f}")


if __name__ == "__main__":
    asyncio.run(main())
