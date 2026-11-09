"""Unit tests for the ``decode`` CLI entrypoint.

``CliRunner`` feeds an empty stdin, so the REPL hits EOF (Ctrl-D) immediately and exits
cleanly without ever issuing a model request — no network. The agent is *built* at startup
(construction is offline), which needs a Gemini key, so each test injects a dummy one via
the settings the factory reads.
"""

import subprocess
import sys
from datetime import UTC, datetime
from uuid import UUID

import pytest
from click.testing import CliRunner
from pydantic import SecretStr
from support.settings_env import hermetic_settings

from decode import cli as cli_mod
from decode.agent import context_window
from decode.cli import cli
from decode.context.session_log import SessionLog


@pytest.fixture(autouse=True)
def _dummy_provider_config(mocker):
    """Seed every LLM Provider's required config so startup construction succeeds (offline).

    Both the CLI's per-provider startup guard (``decode.cli.settings``) and the agent factory
    (``decode.agent.factory.settings``) read the same ``settings`` singleton. Seeding each
    provider's required config here lets the default test runs (provider ``gemini``) — and any test
    that flips ``llm_provider`` to a configured provider — reach ``run_app`` without tripping the
    guard (task 039). Individual tests use ``_select_provider`` to choose a provider and clear
    specific vars to exercise the failure cases. Modal defaults to the *unauthenticated* shape
    (neither proxy token), which is valid.
    """
    # Pin the provider to the gemini default so the suite is hermetic from a developer's local
    # `.env` (e.g. `LLM_PROVIDER=modal`); tests that exercise another provider override this via
    # `_select_provider`.
    mocker.patch.object(cli_mod.settings, "llm_provider", "gemini")
    mocker.patch(
        "decode.agent.factory.settings.gemini_api_key", SecretStr("test-key"), create=False
    )
    mocker.patch.object(cli_mod.settings, "gemini_api_key", SecretStr("test-key"))
    mocker.patch.object(cli_mod.settings, "openrouter_api_key", SecretStr("test-key"))
    mocker.patch.object(cli_mod.settings, "modal_endpoint_url", "https://decode--test.modal.run")
    mocker.patch.object(cli_mod.settings, "modal_endpoint_model", "openai/gpt-oss-120b")
    mocker.patch.object(cli_mod.settings, "modal_proxy_token_id", SecretStr(""))
    mocker.patch.object(cli_mod.settings, "modal_proxy_token_secret", SecretStr(""))


def _select_provider(mocker, provider, **overrides):
    """Select ``provider`` on the shared settings singleton and apply field overrides.

    The guard and the factory share one ``settings`` object, so patching it here is seen by both.
    ``overrides`` are attribute=value pairs (e.g. ``openrouter_api_key=SecretStr("")``).
    """
    mocker.patch.object(cli_mod.settings, "llm_provider", provider)
    for name, value in overrides.items():
        mocker.patch.object(cli_mod.settings, name, value)


def test_bare_decode_prints_help_and_exits_zero():
    result = CliRunner().invoke(cli, [])
    assert result.exit_code == 0
    assert "run" in result.output and "sessions" in result.output


def test_run_and_sessions_are_the_only_subcommands():
    """APICookbook fork (ADR-0024): ``run`` and ``sessions`` stay; the REPL and ``remote`` are gone."""
    assert set(cli.commands) == {"run", "sessions"}


def test_importing_the_cli_does_not_import_a_sandbox_backend():
    """The default (``none``) REPL pulls in no sandbox backend module at all (ADR-0011 §4).

    The backends are imported **lazily**, only when ``SANDBOX_MODE`` selects one — so importing
    ``decode.cli`` (the REPL path, ``none`` by default) must never pull docker/modal wiring in. A
    fresh interpreter keeps the check honest regardless of the rest of the suite.
    """

    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import decode.cli, sys; assert 'decode.sandbox.modal_backend' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr


# ADR-0021 §1,5: there is no Environment-Bucket guard any more, because there is no bucket. A remote
# DECODE_ENV reads the SAME chain as local (process env > .env > defaults) and names things after
# itself, so the REPL starts identically at every environment and never imports kitaru to do it.


# task 004 carryover: the no-key startup guard (friendly line, no traceback)


# the --agent startup flag (ADR-0003 §9, task 020)


# the --mode startup flag (ADR-0003 §9, task 022)


# task 039: the generalized per-provider startup guard (ADR-0005 §6)
#
# ``_provider_config_error()`` returns ONE friendly line (or ``None``) for the selected provider's
# required config. These first tests pin the helper's contract directly (decidable message text /
# which vars it names); the CLI-level tests below prove the exit code + no-traceback behaviour.


def test_provider_config_error_gemini_missing_key_returns_the_unchanged_message(mocker):
    # gemini keeps the verbatim task-004 message for backward-compat.
    _select_provider(mocker, "gemini", gemini_api_key=SecretStr(""))

    assert cli_mod._provider_config_error() == cli_mod._NO_KEY_MESSAGE


def test_provider_config_error_gemini_present_returns_none(mocker):
    _select_provider(mocker, "gemini", gemini_api_key=SecretStr("k"))

    assert cli_mod._provider_config_error() is None


def test_provider_config_error_openrouter_missing_key_names_var_and_provider(mocker):
    _select_provider(mocker, "openrouter", openrouter_api_key=SecretStr(""))

    msg = cli_mod._provider_config_error()

    assert msg is not None
    assert "OPENROUTER_API_KEY" in msg
    assert "openrouter" in msg


def test_provider_config_error_openrouter_present_returns_none(mocker):
    _select_provider(mocker, "openrouter", openrouter_api_key=SecretStr("k"))

    assert cli_mod._provider_config_error() is None


def test_provider_config_error_modal_missing_url_names_only_url(mocker):
    # model present, url absent → name only the absent var.
    _select_provider(mocker, "modal", modal_endpoint_url="", modal_endpoint_model="m")

    msg = cli_mod._provider_config_error()

    assert msg is not None
    assert "MODAL_ENDPOINT_URL" in msg
    assert "MODAL_ENDPOINT_MODEL" not in msg


def test_provider_config_error_modal_missing_model_names_only_model(mocker):
    _select_provider(
        mocker, "modal", modal_endpoint_url="https://x.modal.run", modal_endpoint_model=""
    )

    msg = cli_mod._provider_config_error()

    assert msg is not None
    assert "MODAL_ENDPOINT_MODEL" in msg
    assert "MODAL_ENDPOINT_URL" not in msg


def test_provider_config_error_modal_missing_both_names_both(mocker):
    _select_provider(mocker, "modal", modal_endpoint_url="", modal_endpoint_model="")

    msg = cli_mod._provider_config_error()

    assert msg is not None
    assert "MODAL_ENDPOINT_URL" in msg
    assert "MODAL_ENDPOINT_MODEL" in msg


def test_provider_config_error_modal_both_proxy_tokens_returns_none(mocker):
    # url + model present, both proxy tokens set → authenticated endpoint, valid.
    _select_provider(
        mocker,
        "modal",
        modal_endpoint_url="https://x.modal.run",
        modal_endpoint_model="m",
        modal_proxy_token_id=SecretStr("wk-1"),
        modal_proxy_token_secret=SecretStr("ws-1"),
    )

    assert cli_mod._provider_config_error() is None


def test_provider_config_error_modal_neither_proxy_token_returns_none(mocker):
    # url + model present, neither token set → --unauthenticated endpoint, valid.
    _select_provider(
        mocker,
        "modal",
        modal_endpoint_url="https://x.modal.run",
        modal_endpoint_model="m",
        modal_proxy_token_id=SecretStr(""),
        modal_proxy_token_secret=SecretStr(""),
    )

    assert cli_mod._provider_config_error() is None


def test_provider_config_error_modal_only_token_id_is_both_or_neither(mocker):
    _select_provider(
        mocker,
        "modal",
        modal_endpoint_url="https://x.modal.run",
        modal_endpoint_model="m",
        modal_proxy_token_id=SecretStr("wk-1"),
        modal_proxy_token_secret=SecretStr(""),
    )

    msg = cli_mod._provider_config_error()

    assert msg is not None
    assert "both-or-neither" in msg
    assert "MODAL_PROXY_TOKEN_ID" in msg
    assert "MODAL_PROXY_TOKEN_SECRET" in msg


def test_provider_config_error_modal_only_token_secret_is_both_or_neither(mocker):
    _select_provider(
        mocker,
        "modal",
        modal_endpoint_url="https://x.modal.run",
        modal_endpoint_model="m",
        modal_proxy_token_id=SecretStr(""),
        modal_proxy_token_secret=SecretStr("ws-1"),
    )

    msg = cli_mod._provider_config_error()

    assert msg is not None
    assert "both-or-neither" in msg
    assert "MODAL_PROXY_TOKEN_ID" in msg
    assert "MODAL_PROXY_TOKEN_SECRET" in msg


# task 039: CLI behaviour — friendly line + non-zero exit, no traceback


# task 071: the sandbox backend-availability guard — the helper contract (ADR-0011 §1)
#
# ``_sandbox_config_error()`` returns ONE friendly line (or ``None``) for the selected Sandbox Mode
# when its backend is unavailable. Presence/reachability only, like the provider-key guards — a
# present-but-wrong value is NOT rejected here. The probes are PATCHED: no real docker daemon is
# contacted and no modal credentials/import/network are touched.


def test_sandbox_config_error_none_returns_none_and_runs_no_probe(mocker):
    mocker.patch.object(cli_mod.settings, "sandbox_mode", "none")
    modal_probe = mocker.patch("decode.cli._modal_credentials_present")

    assert cli_mod._sandbox_config_error() is None
    modal_probe.assert_not_called()


def test_sandbox_config_error_modal_missing_creds_returns_the_modal_line(mocker):
    mocker.patch.object(cli_mod.settings, "sandbox_mode", "modal")
    mocker.patch("decode.cli._modal_credentials_present", return_value=False)

    msg = cli_mod._sandbox_config_error()

    assert msg is not None
    assert "SANDBOX_MODE=modal" in msg
    assert "modal token set" in msg
    assert ".env.example" in msg


def test_sandbox_config_error_modal_present_creds_returns_none(mocker):
    """Presence, not correctness: present-but-fake modal creds pass the guard (AC)."""
    mocker.patch.object(cli_mod.settings, "sandbox_mode", "modal")
    mocker.patch("decode.cli._modal_credentials_present", return_value=True)

    assert cli_mod._sandbox_config_error() is None


# task 071: the docker daemon-reachability probe (missing binary / non-zero / timeout)


# task 071: the modal credential-presence probe (no network, no modal import)


def test_modal_credentials_present_true_with_env_token_pair(monkeypatch, mocker, tmp_path):
    """Both MODAL_TOKEN_ID + MODAL_TOKEN_SECRET in env → present (the modal CLI's own contract)."""
    monkeypatch.setenv("MODAL_TOKEN_ID", "ak-1")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "as-1")
    mocker.patch("decode.cli.Path.home", return_value=tmp_path)  # ~/.modal.toml absent regardless

    assert cli_mod._modal_credentials_present() is True


def test_modal_credentials_present_true_with_modal_toml(monkeypatch, mocker, tmp_path):
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)
    (tmp_path / ".modal.toml").write_text("[default]\n")
    mocker.patch("decode.cli.Path.home", return_value=tmp_path)

    assert cli_mod._modal_credentials_present() is True


def test_modal_credentials_present_inside_a_modal_container(monkeypatch, mocker, tmp_path):
    """task 142 / ADR-0020 §3: a Modal container has neither the token pair nor ~/.modal.toml — it
    carries an ambient identity, marked by modal's own MODAL_IS_REMOTE. Without this branch a
    nested-sandbox run on the Modal Headless App is rejected by its own guard."""
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)
    monkeypatch.setenv("MODAL_IS_REMOTE", "1")
    mocker.patch("decode.cli.Path.home", return_value=tmp_path)

    assert cli_mod._modal_credentials_present() is True


def test_modal_credentials_absent_with_no_env_and_no_toml(monkeypatch, mocker, tmp_path):
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)
    monkeypatch.delenv("MODAL_IS_REMOTE", raising=False)
    mocker.patch("decode.cli.Path.home", return_value=tmp_path)  # empty tmp dir → no ~/.modal.toml

    assert cli_mod._modal_credentials_present() is False


def test_modal_credentials_absent_with_only_one_env_token(monkeypatch, mocker, tmp_path):
    monkeypatch.setenv("MODAL_TOKEN_ID", "ak-1")
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)
    mocker.patch("decode.cli.Path.home", return_value=tmp_path)

    assert cli_mod._modal_credentials_present() is False


# task 071: the sandbox guard in the REPL startup chain (friendly line, no traceback)


# task 082: the Workspace repo resolution + the none-mode guard (ADR-0012 §3)
#
# ``_resolve_sandbox_repo`` (--repo > SANDBOX_REPO > None) and ``_sandbox_repo_config_error`` (a repo
# requested while SANDBOX_MODE=none is a config error) are the decidable helpers; the CLI-level tests
# below prove the exit code + no-traceback behaviour and the flag threading into ``run_app``.


def test_resolve_sandbox_repo_flag_wins_over_the_setting(mocker):
    mocker.patch.object(cli_mod.settings, "sandbox_repo", "https://from.env/repo.git")

    assert cli_mod._resolve_sandbox_repo("/flag/repo") == "/flag/repo"


def test_resolve_sandbox_repo_falls_back_to_the_setting(mocker):
    mocker.patch.object(cli_mod.settings, "sandbox_repo", "https://from.env/repo.git")

    assert cli_mod._resolve_sandbox_repo(None) == "https://from.env/repo.git"


def test_resolve_sandbox_repo_none_when_neither_is_set(mocker):
    mocker.patch.object(cli_mod.settings, "sandbox_repo", "")  # the default (unset)

    assert cli_mod._resolve_sandbox_repo(None) is None


def test_sandbox_repo_config_error_repo_in_none_mode_returns_the_message(mocker):
    mocker.patch.object(cli_mod.settings, "sandbox_mode", "none")

    msg = cli_mod._sandbox_repo_config_error("/some/repo")

    assert msg is not None
    assert "--repo/SANDBOX_REPO" in msg
    assert "SANDBOX_MODE=modal" in msg
    assert ".env.example" in msg


def test_sandbox_repo_config_error_repo_in_a_sandbox_mode_returns_none(mocker):
    mocker.patch.object(cli_mod.settings, "sandbox_mode", "modal")

    assert cli_mod._sandbox_repo_config_error("/some/repo") is None


def test_sandbox_repo_config_error_no_repo_returns_none(mocker):
    mocker.patch.object(cli_mod.settings, "sandbox_mode", "none")

    assert cli_mod._sandbox_repo_config_error(None) is None


def test_cli_repo_and_local_flags_are_documented():
    result = CliRunner().invoke(cli, ["run", "--help"])

    assert result.exit_code == 0
    assert "--repo" in result.output
    assert "--local" in result.output


# --version flag


def test_cli_version_option_exists():
    """`--version` is registered on the CLI group and exits cleanly."""
    result = CliRunner().invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "decode" in result.output
    assert "0.1.0" in result.output


def test_version_option_does_not_trigger_startups():
    """--version exits before any startup guards or env checks — pure metadata."""
    # Even with no provider configured, --version must succeed (no real provider key needed).
    result = CliRunner().invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "GEMINI_API_KEY" not in result.output
    assert "Traceback" not in result.output


# Context-window notice + the no-inference paths (task 123)


def test_help_performs_no_provider_probe(mocker):
    """``decode --help`` must never cold-start a GPU to print usage text."""
    probe = mocker.patch.object(context_window, "_probe")

    result = CliRunner().invoke(cli, ["--help"])

    assert result.exit_code == 0
    probe.assert_not_called()


def test_version_performs_no_provider_probe(mocker):
    """Same for ``--version`` — pure metadata, no inference, no network."""
    probe = mocker.patch.object(context_window, "_probe")

    result = CliRunner().invoke(cli, ["--version"])

    assert result.exit_code == 0
    probe.assert_not_called()


def test_run_help_performs_no_provider_probe(mocker):
    """The ``run`` subcommand's own help is a no-inference path too."""
    probe = mocker.patch.object(context_window, "_probe")

    result = CliRunner().invoke(cli, ["run", "--help"])

    assert result.exit_code == 0
    probe.assert_not_called()


@pytest.fixture
def notice_settings(monkeypatch, mocker):
    """Point BOTH the resolver and the cli at a purpose-built Settings for the notice tests.

    Not the shared singleton: by the time the whole suite has run, another test has assigned to it
    directly and permanently marked ``compaction_context_window_tokens`` as explicitly set, which
    is precisely the signal the notice branches on (see ``support.settings_env``).
    """

    def _use(**overrides):
        config = hermetic_settings(monkeypatch, **overrides)
        mocker.patch.object(context_window, "settings", config)
        mocker.patch.object(cli_mod, "settings", config)
        return config

    return _use


def test_context_window_notice_warns_only_when_nothing_could_resolve_it(notice_settings, mocker):
    """An unresolvable model gets the one assumed-window line, naming the fallback."""
    mocker.patch.object(context_window, "_probe", return_value=None)
    notice_settings(llm_provider="gemini", gemini_model="acme/unlisted-model-v1")

    notice = cli_mod._context_window_notice()

    assert notice is not None
    assert "acme/unlisted-model-v1" in notice
    assert "200,000" in notice


def test_context_window_notice_is_silent_after_a_successful_probe(notice_settings, mocker):
    """A probed window is KNOWN — claiming "assuming" would be a lie the operator learns to skip."""
    mocker.patch.object(context_window, "_probe", return_value=262_144)
    notice_settings(llm_provider="gemini", gemini_model="acme/unlisted-model-v1")

    assert cli_mod._context_window_notice() is None


def test_context_window_notice_is_silent_for_a_table_model(notice_settings, mocker):
    mocker.patch.object(context_window, "_probe", return_value=None)
    notice_settings(llm_provider="gemini", gemini_model="gemini-3.5-flash")

    assert cli_mod._context_window_notice() is None


def test_context_window_notice_is_silent_when_the_operator_set_the_window(notice_settings, mocker):
    """An explicit setting is owned by the operator — never warned about, and never probed."""
    probe = mocker.patch.object(context_window, "_probe")
    notice_settings(
        llm_provider="gemini",
        gemini_model="acme/unlisted-model-v1",
        compaction_context_window_tokens=8192,
    )

    assert cli_mod._context_window_notice() is None
    probe.assert_not_called()


def test_context_window_notice_describes_the_model_override(notice_settings, mocker):
    """``decode run --model <unknown>`` warns about the OVERRIDE, not the configured model."""
    mocker.patch.object(context_window, "_probe", return_value=None)
    notice_settings(llm_provider="gemini", gemini_model="gemini-3.5-flash")

    notice = cli_mod._context_window_notice("acme/unlisted-model-v1")

    assert notice is not None
    assert "acme/unlisted-model-v1" in notice


# ``decode sessions`` — the local session listing


def _seed_sessions(tmp_path, mocker, hours=(9, 11)):
    """Open one session per hour under a temp sessions dir the CLI command will read."""
    sessions_dir = tmp_path / "sessions"
    mocker.patch.object(cli_mod.settings, "sessions_dir", sessions_dir)
    for index, hour in enumerate(hours, start=1):
        SessionLog.create(
            sessions_dir,
            cwd=tmp_path,
            now=datetime(2026, 6, 19, hour, 30, 45, tzinfo=UTC),
            session_id=UUID(f"0000000{index}-0000-0000-0000-000000000000"),
        )
    return sessions_dir


def test_sessions_lists_most_recent_first(tmp_path, mocker):
    _seed_sessions(tmp_path, mocker)

    result = CliRunner().invoke(cli, ["sessions"])

    assert result.exit_code == 0
    lines = result.output.splitlines()
    assert [line.split()[1] for line in lines] == ["11:30:45Z", "09:30:45Z"]
    # The id column is exactly what ``--resume`` takes.
    assert lines[0].split()[2] == "00000002-0000-0000-0000-000000000000"


def test_sessions_reports_an_empty_directory_without_failing(tmp_path, mocker):
    mocker.patch.object(cli_mod.settings, "sessions_dir", tmp_path / "sessions")

    result = CliRunner().invoke(cli, ["sessions"])

    assert result.exit_code == 0  # no sessions yet is the first-run state, not an error
    assert "no sessions yet" in result.output


def test_sessions_needs_no_provider_key(tmp_path, mocker, monkeypatch):
    """Listing local files must not run the provider startup guard."""
    _seed_sessions(tmp_path, mocker, hours=(9,))
    mocker.patch.object(cli_mod.settings, "gemini_api_key", SecretStr(""))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    result = CliRunner().invoke(cli, ["sessions"])

    assert result.exit_code == 0
    assert "00000001-0000-0000-0000-000000000000" in result.output


def test_provider_config_error_anthropic_missing_key_names_the_var(mocker):
    _select_provider(mocker, "anthropic", anthropic_api_key=SecretStr(""))
    assert "ANTHROPIC_API_KEY" in cli_mod._provider_config_error()


def test_provider_config_error_anthropic_present_returns_none(mocker):
    _select_provider(mocker, "anthropic", anthropic_api_key=SecretStr("test-key"))
    assert cli_mod._provider_config_error() is None
