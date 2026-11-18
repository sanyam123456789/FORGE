"""
Tests for the shell-security layer (Fix 1): minimal child environment,
closed stdin, secret redaction, and secret-file blocking.

All commands are run with ``sys.executable`` so they work on Windows and
POSIX alike. No network, no real API keys.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from forge.builtin_tools import (
    EditFileTool,
    ListDirectoryTool,
    ReadFileTool,
    RunShellTool,
    WriteFileTool,
)
from forge.safety import (
    REDACTED,
    build_shell_env,
    collect_secret_values,
    is_secret_file,
    redact_secrets,
)

FAKE_KEY = "fake-service-key-1234567890"
FAKE_OPENCODE = "fake-opencode-key-abcdefghij"


def _py(code: str) -> str:
    """A run_shell command line running *code* with the current interpreter."""
    return f'"{sys.executable}" -c "{code}"'


@pytest.fixture
def secrets_env(monkeypatch):
    monkeypatch.setenv("FAKE_SERVICE_API_KEY", FAKE_KEY)
    monkeypatch.setenv("OPENCODE_API_KEY", FAKE_OPENCODE)
    monkeypatch.setenv("FORGE_LLM_API_KEY", "fake-forge-llm-key-xyz987")
    monkeypatch.setenv("FORGE_SOMETHING_PRIVATE", "forge-private-value-555")


# ---------------------------------------------------------------------------
# build_shell_env
# ---------------------------------------------------------------------------


class TestBuildShellEnv:
    def test_allowlisted_names_pass(self):
        env = build_shell_env(environ={"PATH": "/bin", "HOME": "/h", "LANG": "C"})
        assert env == {"PATH": "/bin", "HOME": "/h", "LANG": "C"}

    def test_case_insensitive_allowlist(self):
        env = build_shell_env(environ={"Path": "/bin", "SystemRoot": "C:\\Windows"})
        assert env == {"Path": "/bin", "SystemRoot": "C:\\Windows"}

    def test_unknown_and_secret_names_withheld(self):
        env = build_shell_env(
            environ={"PATH": "/bin", "OPENCODE_API_KEY": "x", "RANDOM_VAR": "y"}
        )
        assert env == {"PATH": "/bin"}

    def test_forge_vars_never_passed_even_if_opted_in(self):
        env = build_shell_env(
            ["FORGE_LLM_API_KEY", "forge_log_level"],
            environ={"FORGE_LLM_API_KEY": "k", "FORGE_LOG_LEVEL": "INFO"},
        )
        assert env == {}

    def test_opt_in_passes_secret_looking_name(self):
        env = build_shell_env(
            ["github_token"], environ={"GITHUB_TOKEN": "t", "OTHER_TOKEN": "u"}
        )
        assert env == {"GITHUB_TOKEN": "t"}


# ---------------------------------------------------------------------------
# run_shell child environment
# ---------------------------------------------------------------------------


class TestRunShellEnvironment:
    def test_fake_secrets_not_visible_to_child(self, tmp_path, secrets_env):
        code = (
            "import os; "
            "print(os.environ.get('FAKE_SERVICE_API_KEY', 'MISSING')); "
            "print(os.environ.get('OPENCODE_API_KEY', 'MISSING'))"
        )
        result = RunShellTool(workspace_root=tmp_path, env_allow=()).execute(
            command=_py(code)
        )
        assert result.success is True, result.output
        assert result.output.splitlines()[:2] == ["MISSING", "MISSING"]

    def test_forge_vars_not_visible_even_when_allowed(self, tmp_path, secrets_env):
        code = (
            "import os; "
            "print(sorted(k for k in os.environ if k.upper().startswith('FORGE_')))"
        )
        tool = RunShellTool(
            workspace_root=tmp_path,
            env_allow=("FORGE_LLM_API_KEY", "FORGE_SOMETHING_PRIVATE"),
        )
        result = tool.execute(command=_py(code))
        assert result.success is True, result.output
        assert result.output.splitlines()[0] == "[]"

    def test_path_present_and_python_runs(self, tmp_path):
        code = "import os; print(1); print(bool(os.environ.get('PATH')))"
        result = RunShellTool(workspace_root=tmp_path, env_allow=()).execute(
            command=_py(code)
        )
        assert result.success is True, result.output
        assert result.output.splitlines()[:2] == ["1", "True"]

    def test_opted_in_variable_is_passed_through(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MY_TOOL_HOME", "/opt/mytool")
        code = "import os; print(os.environ.get('MY_TOOL_HOME', 'MISSING'))"
        without = RunShellTool(workspace_root=tmp_path, env_allow=()).execute(
            command=_py(code)
        )
        with_allow = RunShellTool(
            workspace_root=tmp_path, env_allow=("my_tool_home",)
        ).execute(command=_py(code))
        assert without.output.splitlines()[0] == "MISSING"
        assert with_allow.output.splitlines()[0] == "/opt/mytool"

    def test_opted_in_secret_reaches_child_but_is_redacted(self, tmp_path, secrets_env):
        code = (
            "import os; v = os.environ.get('FAKE_SERVICE_API_KEY', ''); "
            "print(len(v)); print(v)"
        )
        result = RunShellTool(
            workspace_root=tmp_path, env_allow=("FAKE_SERVICE_API_KEY",)
        ).execute(command=_py(code))
        lines = result.output.splitlines()
        assert lines[0] == str(len(FAKE_KEY))  # the child did receive it
        assert lines[1] == REDACTED  # ... but the model never sees it
        assert FAKE_KEY not in result.output

    def test_env_allow_defaults_from_settings(self, tmp_path, monkeypatch):
        import forge.config as config

        fake = config.ForgeSettings(shell_env_allow=("my_default_var",))
        monkeypatch.setattr(config, "settings", fake)
        assert RunShellTool(workspace_root=tmp_path)._env_allow == ("my_default_var",)

    def test_env_allow_setting_parsed(self, clean_env, monkeypatch):
        from forge.config import load_settings

        monkeypatch.setenv("FORGE_SHELL_ENV_ALLOW", " GITHUB_TOKEN, my_var ,")
        assert load_settings().shell_env_allow == ("github_token", "my_var")

    def test_env_allow_default_empty(self, clean_env, monkeypatch):
        from forge.config import load_settings

        monkeypatch.delenv("FORGE_SHELL_ENV_ALLOW", raising=False)
        assert load_settings().shell_env_allow == ()


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------


class TestRedaction:
    def test_secret_printed_by_command_is_redacted(self, tmp_path, secrets_env):
        # The child cannot see the key, so the command prints the literal.
        code = f"import sys; print('key={FAKE_KEY}'); print('{FAKE_OPENCODE}', file=sys.stderr)"
        result = RunShellTool(workspace_root=tmp_path, env_allow=()).execute(
            command=_py(code)
        )
        assert FAKE_KEY not in result.output
        assert FAKE_OPENCODE not in result.output
        assert f"key={REDACTED}" in result.output
        assert result.output.endswith("[exit code 0]")

    def test_forge_private_value_redacted_but_known_settings_kept(self, monkeypatch):
        monkeypatch.setenv("FORGE_SOMETHING_PRIVATE", "forge-private-value-555")
        monkeypatch.setenv("FORGE_LLM_MODEL", "deepseek-v4.1-flash")
        monkeypatch.setenv("FORGE_WORKSPACE_ROOT", "/home/user/project")
        text = "forge-private-value-555 deepseek-v4.1-flash /home/user/project"
        assert redact_secrets(text) == (
            f"{REDACTED} deepseek-v4.1-flash /home/user/project"
        )

    def test_short_values_not_treated_as_secret(self):
        assert collect_secret_values({"SOME_TOKEN": "abc", "X_KEY": "12345678"}) == [
            "12345678"
        ]

    def test_read_file_output_is_redacted(self, tmp_path, secrets_env):
        (tmp_path / "notes.txt").write_text(f"token: {FAKE_KEY}\n", encoding="utf-8")
        result = ReadFileTool(workspace_root=tmp_path).execute(path="notes.txt")
        assert result.output == f"token: {REDACTED}\n"


# ---------------------------------------------------------------------------
# stdin
# ---------------------------------------------------------------------------


def test_stdin_command_returns_immediately(tmp_path):
    code = "import sys; data = sys.stdin.read(); print('got', repr(data))"
    result = RunShellTool(workspace_root=tmp_path, env_allow=()).execute(
        command=_py(code), timeout=20
    )
    assert result.success is True, result.output
    assert "got ''" in result.output


def test_subprocess_receives_devnull_stdin(tmp_path, monkeypatch):
    seen = {}

    def fake_run(*args, **kwargs):
        seen.update(kwargs)
        return subprocess.CompletedProcess(args[0], 0, stdout="", stderr="")

    monkeypatch.setattr("forge.builtin_tools.subprocess.run", fake_run)
    RunShellTool(workspace_root=tmp_path, env_allow=()).execute(command="echo hi")
    assert seen["stdin"] is subprocess.DEVNULL
    assert isinstance(seen["env"], dict)


# ---------------------------------------------------------------------------
# Secret files
# ---------------------------------------------------------------------------


class TestSecretFiles:
    @pytest.mark.parametrize(
        "name",
        [".env", ".env.local", ".ENV.production", "server.pem", "tls.key",
         "id_rsa", "id_rsa.pub", "id_ed25519"],
    )
    def test_secret_names(self, name):
        assert is_secret_file(name) is True

    @pytest.mark.parametrize(
        "name", [".env.example", "env.py", "keys.py", "app.envrc", "notes.txt"]
    )
    def test_non_secret_names(self, name):
        assert is_secret_file(name) is False

    def test_read_env_blocked(self, tmp_path):
        (tmp_path / ".env").write_text("FORGE_LLM_API_KEY=x\n", encoding="utf-8")
        result = ReadFileTool(workspace_root=tmp_path).execute(path=".env")
        assert result.success is False
        assert result.output == "ERROR: access to secret file '.env' is blocked by FORGE"

    def test_read_nested_env_blocked(self, tmp_path):
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / ".env.local").write_text("A=1\n", encoding="utf-8")
        result = ReadFileTool(workspace_root=tmp_path).execute(path="sub/.env.local")
        assert result.output.startswith("ERROR")

    def test_read_env_example_allowed(self, tmp_path):
        (tmp_path / ".env.example").write_text("FORGE_LLM_API_KEY=\n", encoding="utf-8")
        result = ReadFileTool(workspace_root=tmp_path).execute(path=".env.example")
        assert result.success is True
        assert result.output == "FORGE_LLM_API_KEY=\n"

    def test_write_and_edit_env_blocked(self, tmp_path):
        result = WriteFileTool(workspace_root=tmp_path).execute(path=".env", content="A=1")
        assert result.output.startswith("ERROR")
        assert not (tmp_path / ".env").exists()
        (tmp_path / "id_rsa").write_text("PRIVATE", encoding="utf-8")
        result = EditFileTool(workspace_root=tmp_path).execute(
            path="id_rsa", old_string="PRIVATE", new_string="X"
        )
        assert result.output.startswith("ERROR")
        assert (tmp_path / "id_rsa").read_text(encoding="utf-8") == "PRIVATE"

    def test_list_directory_still_shows_names(self, tmp_path):
        (tmp_path / ".env").write_text("A=1\n", encoding="utf-8")
        result = ListDirectoryTool(workspace_root=tmp_path).execute()
        assert ".env" in result.output
