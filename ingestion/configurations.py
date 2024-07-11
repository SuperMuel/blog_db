from typing import List, Any
from pydantic import BaseModel, Field


class Configuration(BaseModel):
    name: str
    region: str
    queries: List[str]


def read_configs(config_path: str) -> list[Configuration]:
    from json import loads

    configurations = loads(open(config_path, "r").read())["configurations"]

    return [Configuration(**config) for config in configurations]
