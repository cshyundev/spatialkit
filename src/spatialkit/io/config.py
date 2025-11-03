"""
Configuration file I/O for JSON and YAML formats.

This module provides functions for reading and writing configuration files
in JSON and YAML formats.

Functions:
    - read_json: Read JSON files
    - write_json: Write JSON files with indentation
    - read_yaml: Read YAML files
    - write_yaml: Write YAML files

Author: Sehyun Cha
Email: cshyundev@gmail.com
License: MIT License
"""

import os
import os.path as osp
import json
from typing import Dict, Any

import yaml

from ..common.exceptions import (
    FileNotFoundError as CVFileNotFoundError,
    FileFormatError,
    ReadWriteError
)
def read_json(path: str) -> Dict[str, Any]:
    """
    Reads a JSON file.

    Args:
        path (str): Path to the JSON file.

    Returns:
        Dict[str, Any]: Data read from the JSON file.
        
    Raises:
        CVFileNotFoundError: If JSON file is not found.
        FileFormatError: If JSON file format is invalid.
        ReadWriteError: If reading JSON file fails.

    Example:
        data = read_json('example.json')
    """
    if not osp.exists(path):
        raise CVFileNotFoundError(
            f"JSON file not found: {path}. "
            f"Please check the file path and ensure the file exists."
        )
    
    if not osp.isfile(path):
        raise CVFileNotFoundError(
            f"Path is not a file: {path}. "
            f"Please provide a valid JSON file path."
        )
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            json_dict = json.load(f)
        return json_dict
    except json.JSONDecodeError as e:
        raise FileFormatError(
            f"Invalid JSON format in file {path}: {e}. "
            f"Please check JSON syntax and format."
        ) from e
    except Exception as e:
        raise ReadWriteError(
            f"Failed to read JSON file {path}: {e}. "
            f"Please check file permissions and format."
        ) from e


def write_json(path: str, json_dict: Dict[str, Any], indent: int = 1):
    """
    Writes a dictionary to a JSON file.

    Args:
        path (str): Path to save the JSON file.
        json_dict (Dict[str, Any]): Dictionary to write to the JSON file.
        indent (int): Indentation level for pretty-printing. Default is 1.
        
    Raises:
        ValueError: If json_dict is not serializable.
        ReadWriteError: If writing JSON file fails.

    Example:
        data = {'key': 'value'}
        write_json('output.json', data, indent=2)
    """
    if not isinstance(json_dict, dict):
        raise ValueError(
            f"json_dict must be a dictionary, got {type(json_dict)}. "
            f"Please provide a valid dictionary to write as JSON."
        )
    
    # Create directory if it doesn't exist
    dir_path = osp.dirname(path)
    if dir_path and not osp.exists(dir_path):
        try:
            os.makedirs(dir_path, exist_ok=True)
        except Exception as e:
            raise ReadWriteError(
                f"Failed to create directory {dir_path}: {e}. "
                f"Please check directory permissions."
            ) from e
    
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(json_dict, f, indent=indent)
    except TypeError as e:
        raise ValueError(
            f"Failed to serialize data to JSON: {e}. "
            f"Please ensure all data types are JSON serializable."
        ) from e
    except Exception as e:
        raise ReadWriteError(
            f"Failed to write JSON file {path}: {e}. "
            f"Please check file path and permissions."
        ) from e


def read_yaml(path: str) -> Dict[str, Any]:
    """
    Reads a YAML file.

    Args:
        path (str): Path to the YAML file.

    Returns:
        Dict[str, Any]: Data read from the YAML file.
        
    Raises:
        CVFileNotFoundError: If YAML file is not found.
        FileFormatError: If YAML file format is invalid.
        ReadWriteError: If reading YAML file fails.

    Example:
        data = read_yaml('example.yaml')
    """
    if not osp.exists(path):
        raise CVFileNotFoundError(
            f"YAML file not found: {path}. "
            f"Please check the file path and ensure the file exists."
        )
    
    if not osp.isfile(path):
        raise CVFileNotFoundError(
            f"Path is not a file: {path}. "
            f"Please provide a valid YAML file path."
        )
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            data_dict = yaml.load(f, Loader=yaml.FullLoader)
        return data_dict
    except yaml.YAMLError as e:
        raise FileFormatError(
            f"Invalid YAML format in file {path}: {e}. "
            f"Please check YAML syntax and format."
        ) from e
    except Exception as e:
        raise ReadWriteError(
            f"Failed to read YAML file {path}: {e}. "
            f"Please check file permissions and format."
        ) from e


def write_yaml(path: str, yaml_dict: Dict[str, Any]):
    """
    Writes a dictionary to a YAML file.

    Args:
        path (str): Path to save the YAML file.
        yaml_dict (Dict[str, Any]): Dictionary to write to the YAML file.
        
    Raises:
        ValueError: If yaml_dict is not a valid dictionary.
        ReadWriteError: If writing YAML file fails.

    Example:
        data = {'key': 'value'}
        write_yaml('output.yaml', data)
    """
    if not isinstance(yaml_dict, dict):
        raise ValueError(
            f"yaml_dict must be a dictionary, got {type(yaml_dict)}. "
            f"Please provide a valid dictionary to write as YAML."
        )
    
    # Create directory if it doesn't exist
    dir_path = osp.dirname(path)
    if dir_path and not osp.exists(dir_path):
        try:
            os.makedirs(dir_path, exist_ok=True)
        except Exception as e:
            raise ReadWriteError(
                f"Failed to create directory {dir_path}: {e}. "
                f"Please check directory permissions."
            ) from e
    
    try:
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(yaml_dict, f, default_flow_style=False, allow_unicode=True)
    except Exception as e:
        raise ReadWriteError(
            f"Failed to write YAML file {path}: {e}. "
            f"Please check file path and permissions."
        ) from e


