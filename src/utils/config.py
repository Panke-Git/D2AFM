"""
    @Project: UnderwaterImageEnhanced
    @Author: Panke
    @FileName: config.py
    @Time: 2025/5/20 22:29
    @Email: None
"""

import yaml


class Config:
    def __init__(self, data):
        for key, value in data.items():

            if isinstance(value, dict):
                setattr(self, key, Config(value))
            elif isinstance(value, list):
                setattr(self, key, [
                    Config(item) if isinstance(item, dict) else item for item in value
                ])
            else:
                setattr(self, key, value)

    @classmethod
    def load(cls, filepath: str, encoding: str = "utf-8"):
        with open(filepath, "r", encoding=encoding) as f:
            data = yaml.safe_load(f)
        return cls(data)
