from dataclasses import dataclass
from typing import List
import jinja2
import os

from .yaml_config import yaml


@dataclass
class Config:
    model: str
    iterations: int
    content_variants: List[str]
    content_prompt: str
    judge_prompt: str
    judge_categories: List[str]
    warm_cache: bool

    @classmethod
    async def load(cls, data) -> "Config":
        with open("./config/config.yml", "r") as f:
            if data is None:
                data = yaml.load(f)

        base_path = os.path.dirname(os.path.abspath("./config/config.yml"))
        resolved_data = cls._resolve_vars(data, base_path)

        return cls(
            model=resolved_data["model"],
            iterations=resolved_data["iterations"],
            content_variants=resolved_data["content_variants"],
            content_prompt=resolved_data["content_prompt"],
            judge_prompt=resolved_data["judge_prompt"],
            judge_categories=resolved_data["judge_categories"],
            warm_cache=resolved_data["warm_cache"],
        )

    @staticmethod
    def _resolve_vars(vars, base_path):
        MAX_ITERATIONS = 10
        for _ in range(MAX_ITERATIONS):
            resolved_vars = Config._resolve_vars_recursive(vars, base_path, vars)
            if resolved_vars == vars:
                return resolved_vars
            vars = resolved_vars
        raise RuntimeError("Too many iterations resolving vars. Circular reference?")

    @staticmethod
    def _resolve_vars_recursive(obj, base_path, context):
        if isinstance(obj, dict):
            return {
                key: Config._resolve_vars_recursive(value, base_path, context)
                for key, value in obj.items()
            }
        elif isinstance(obj, list):
            return [
                Config._resolve_vars_recursive(item, base_path, context) for item in obj
            ]
        elif isinstance(obj, str):
            if obj.startswith("file:"):
                file_path = obj.split("file:", 1)[1]
                full_path = os.path.join(base_path, file_path)
                with open(full_path) as file:
                    return file.read()
            return jinja2.Template(obj, trim_blocks=True, lstrip_blocks=True).render(
                context
            )
        else:
            return obj

    @property
    def total_calls(self) -> int:
        return self.iterations * len(self.content_variants) * 2
