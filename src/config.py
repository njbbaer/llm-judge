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
            warm_cache=resolved_data.get("warm_cache", False),
        )

    @staticmethod
    def _resolve_vars_recursive(obj, env):
        if isinstance(obj, dict):
            return {k: Config._resolve_vars_recursive(v, env) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [Config._resolve_vars_recursive(i, env) for i in obj]
        elif isinstance(obj, str):
            template = env.from_string(str(obj))
            return template.render()
        return obj

    @staticmethod
    def _resolve_vars(data, base_path):
        MAX_ITERATIONS = 10
        current = data

        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(base_path), autoescape=False
        )

        def load_file(filename):
            template = env.get_template(filename)
            return template.render()

        env.globals["load"] = load_file

        for _ in range(MAX_ITERATIONS):
            resolved = Config._resolve_vars_recursive(current, env)
            if resolved == current:
                return resolved
            current = resolved

        raise RuntimeError("Too many iterations resolving vars. Circular reference?")

    @property
    def total_calls(self) -> int:
        return self.iterations * len(self.content_variants) * 2
