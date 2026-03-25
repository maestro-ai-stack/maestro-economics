"""Unit tests for mecon CLI (no real API calls)."""

import json
import os
import tempfile

import pytest
from click.testing import CliRunner

import cli.__main__ as mod


class TestSetup:
    def test_saves_config(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            old_dir, old_file = mod.CONFIG_DIR, mod.CONFIG_FILE
            mod.CONFIG_DIR = tmpdir
            mod.CONFIG_FILE = os.path.join(tmpdir, "config.json")
            try:
                result = runner.invoke(
                    mod.main,
                    ["setup"],
                    input="rc_live_test123\nhttps://test.com\n",
                )
                assert result.exit_code == 0
                with open(mod.CONFIG_FILE) as f:
                    cfg = json.load(f)
                assert cfg["api_key"] == "rc_live_test123"
                assert cfg["api_base"] == "https://test.com"
            finally:
                mod.CONFIG_DIR, mod.CONFIG_FILE = old_dir, old_file

    def test_default_api_base(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            old_dir, old_file = mod.CONFIG_DIR, mod.CONFIG_FILE
            mod.CONFIG_DIR = tmpdir
            mod.CONFIG_FILE = os.path.join(tmpdir, "config.json")
            try:
                result = runner.invoke(
                    mod.main,
                    ["setup"],
                    input="mykey\n\n",  # accept default base
                )
                assert result.exit_code == 0
                with open(mod.CONFIG_FILE) as f:
                    cfg = json.load(f)
                assert cfg["api_base"] == "https://ra.maestro.onl"
            finally:
                mod.CONFIG_DIR, mod.CONFIG_FILE = old_dir, old_file


class TestNoKey:
    def test_balance_without_key_errors(self):
        runner = CliRunner()
        old_file = mod.CONFIG_FILE
        old_env = os.environ.pop("MECON_API_KEY", None)
        old_base = os.environ.pop("MECON_API_BASE", None)
        mod.CONFIG_FILE = "/nonexistent/config.json"
        try:
            result = runner.invoke(mod.main, ["balance"])
            assert result.exit_code != 0
        finally:
            mod.CONFIG_FILE = old_file
            if old_env:
                os.environ["MECON_API_KEY"] = old_env
            if old_base:
                os.environ["MECON_API_BASE"] = old_base

    def test_status_without_key_errors(self):
        runner = CliRunner()
        old_file = mod.CONFIG_FILE
        old_env = os.environ.pop("MECON_API_KEY", None)
        old_base = os.environ.pop("MECON_API_BASE", None)
        mod.CONFIG_FILE = "/nonexistent/config.json"
        try:
            result = runner.invoke(mod.main, ["status", "abc123"])
            assert result.exit_code != 0
        finally:
            mod.CONFIG_FILE = old_file
            if old_env:
                os.environ["MECON_API_KEY"] = old_env
            if old_base:
                os.environ["MECON_API_BASE"] = old_base


class TestGetConfig:
    def test_env_var_takes_precedence(self):
        old_env_key = os.environ.get("MECON_API_KEY")
        old_env_base = os.environ.get("MECON_API_BASE")
        old_file = mod.CONFIG_FILE
        try:
            # Write a config file
            with tempfile.TemporaryDirectory() as tmpdir:
                mod.CONFIG_FILE = os.path.join(tmpdir, "config.json")
                with open(mod.CONFIG_FILE, "w") as f:
                    json.dump({"api_key": "file_key", "api_base": "https://file.com"}, f)
                # Set env vars
                os.environ["MECON_API_KEY"] = "env_key"
                os.environ["MECON_API_BASE"] = "https://env.com"
                cfg = mod.get_config()
                assert cfg["api_key"] == "env_key"
                assert cfg["api_base"] == "https://env.com"
        finally:
            mod.CONFIG_FILE = old_file
            if old_env_key:
                os.environ["MECON_API_KEY"] = old_env_key
            else:
                os.environ.pop("MECON_API_KEY", None)
            if old_env_base:
                os.environ["MECON_API_BASE"] = old_env_base
            else:
                os.environ.pop("MECON_API_BASE", None)

    def test_file_config_used_when_no_env(self):
        old_env_key = os.environ.pop("MECON_API_KEY", None)
        old_env_base = os.environ.pop("MECON_API_BASE", None)
        old_file = mod.CONFIG_FILE
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                mod.CONFIG_FILE = os.path.join(tmpdir, "config.json")
                with open(mod.CONFIG_FILE, "w") as f:
                    json.dump({"api_key": "file_key", "api_base": "https://file.com"}, f)
                cfg = mod.get_config()
                assert cfg["api_key"] == "file_key"
                assert cfg["api_base"] == "https://file.com"
        finally:
            mod.CONFIG_FILE = old_file
            if old_env_key:
                os.environ["MECON_API_KEY"] = old_env_key
            if old_env_base:
                os.environ["MECON_API_BASE"] = old_env_base

    def test_defaults_when_no_config(self):
        old_env_key = os.environ.pop("MECON_API_KEY", None)
        old_env_base = os.environ.pop("MECON_API_BASE", None)
        old_file = mod.CONFIG_FILE
        mod.CONFIG_FILE = "/nonexistent/config.json"
        try:
            cfg = mod.get_config()
            assert cfg["api_key"] == ""
            assert cfg["api_base"] == "https://ra.maestro.onl"
        finally:
            mod.CONFIG_FILE = old_file
            if old_env_key:
                os.environ["MECON_API_KEY"] = old_env_key
            if old_env_base:
                os.environ["MECON_API_BASE"] = old_env_base


class TestCliHelp:
    def test_main_help(self):
        runner = CliRunner()
        result = runner.invoke(mod.main, ["--help"])
        assert result.exit_code == 0
        assert "mecon" in result.output.lower() or "RA Compute" in result.output

    def test_submit_help(self):
        runner = CliRunner()
        result = runner.invoke(mod.main, ["submit", "--help"])
        assert result.exit_code == 0
        assert "--gpu" in result.output
        assert "--timeout" in result.output
