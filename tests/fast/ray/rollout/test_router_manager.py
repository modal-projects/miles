from __future__ import annotations

from unittest.mock import patch

import pytest
from tests.fast.ray.rollout.conftest import make_args

from miles.ray.rollout.router_manager import (
    _resolve_session_server_ports,
    _wait_for_session_server_pool_ready,
    start_router,
    start_session_server,
)


class TestStartRouter:
    def test_returns_existing_when_already_configured(self):
        """Happy path: ``sglang_router_ip`` and ``sglang_router_port`` are
        already set and ``force_new=False`` → skip subprocess launch entirely
        and return the existing tuple."""
        args = make_args(
            use_miles_router=False,
            sglang_router_ip="10.1.2.3",
            sglang_router_port=4567,
        )
        # No mocks needed — the function returns before touching anything.
        ip, port = start_router(args, force_new=False)
        assert (ip, port) == ("10.1.2.3", 4567)

    def test_pd_disagg_with_miles_router_asserts(self):
        args = make_args(use_miles_router=True, sglang_router_ip=None, sglang_router_port=None)
        with patch("miles.ray.rollout.router_manager.get_host_info", return_value=("h", "127.0.0.1")), patch(
            "miles.ray.rollout.router_manager.find_available_port", return_value=20000
        ):
            with pytest.raises(AssertionError, match="miles router does not support PD"):
                start_router(args, has_pd_disaggregation=True, force_new=False)

    def test_port_conflict_raises_runtime_error(self):
        args = make_args(use_miles_router=False, sglang_router_ip=None, sglang_router_port=None)
        with patch("miles.ray.rollout.router_manager.get_host_info", return_value=("h", "127.0.0.1")), patch(
            "miles.ray.rollout.router_manager.find_available_port", return_value=20000
        ), patch("miles.ray.rollout.router_manager.is_port_available", return_value=False):
            with pytest.raises(RuntimeError, match="already in use"):
                start_router(args)


class TestStartSessionServer:
    def test_disabled_returns_silently(self):
        """Happy no-op: ``use_session_server=False`` → return without raising,
        without touching any other config."""
        args = make_args(use_session_server=False)
        start_session_server(args)

    def test_enabled_without_hf_checkpoint_raises(self):
        args = make_args(use_session_server=True, hf_checkpoint=None)
        with pytest.raises(ValueError, match="hf-checkpoint"):
            start_session_server(args)

    def test_enabled_port_conflict_raises_runtime_error(self):
        """When a configured ``session_server_port`` is already bound, fail
        loud rather than silently re-using the stale process."""
        args = make_args(
            use_session_server=True,
            hf_checkpoint="/fake/model",
            sglang_router_ip="127.0.0.1",
            sglang_router_port=20000,
            session_server_ip="127.0.0.1",
            session_server_port=[20001],
        )
        with patch("miles.ray.rollout.router_manager.is_port_available", return_value=False):
            with pytest.raises(RuntimeError, match="already in use"):
                start_session_server(args)


class TestResolveSessionServerPorts:
    def test_none_auto_allocates_one_port(self):
        with patch("miles.ray.rollout.router_manager.find_available_port", return_value=20002):
            assert _resolve_session_server_ports(None) == [20002]

    def test_single_value_is_a_single_server(self):
        assert _resolve_session_server_ports([30000]) == [30000]

    def test_two_values_expand_to_half_open_range(self):
        assert _resolve_session_server_ports([30000, 30004]) == [30000, 30001, 30002, 30003]

    def test_empty_range_raises(self):
        with pytest.raises(ValueError, match="empty"):
            _resolve_session_server_ports([30004, 30000])

    def test_more_than_two_values_raises(self):
        with pytest.raises(ValueError, match="one port or a start/end range"):
            _resolve_session_server_ports([30000, 30001, 30002])


class TestWaitForSessionServerPoolReady:
    def test_polls_all_ports_against_one_deadline(self):
        class Process:
            def is_alive(self):
                return True

        attempts = {30000: 0, 30001: 0}

        class Connection:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

        def create_connection(address, timeout):
            del timeout
            assert address[0] == "::1"
            port = address[1]
            attempts[port] += 1
            if attempts[port] < (3 if port == 30000 else 2):
                raise ConnectionRefusedError
            return Connection()

        with (
            patch(
                "miles.ray.rollout.router_manager.socket.create_connection",
                side_effect=create_connection,
            ),
            patch("miles.ray.rollout.router_manager.time.sleep"),
        ):
            ready = _wait_for_session_server_pool_ready(
                "[::1]",
                [(30000, Process()), (30001, Process())],
                timeout=10,
            )

        assert set(ready) == {30000, 30001}
        assert attempts == {30000: 3, 30001: 2}

    def test_reports_dead_shard_with_pool_progress(self):
        class Process:
            def __init__(self, alive):
                self.alive = alive

            def is_alive(self):
                return self.alive

        with pytest.raises(RuntimeError, match=r"port 30001 died.*0/2 ready"):
            _wait_for_session_server_pool_ready(
                "127.0.0.1",
                [(30000, Process(True)), (30001, Process(False))],
                timeout=1,
            )
