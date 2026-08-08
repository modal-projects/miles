from types import SimpleNamespace

import miles.rollout.session.server as session_server


def test_run_session_server_disables_uvicorn_access_log(monkeypatch):
    app = object()
    uvicorn_call = {}

    class FakeSessionServer:
        def __init__(self, args, backend_url):
            assert backend_url == "http://backend"
            self.app = app

    monkeypatch.setattr(session_server, "configure_logger_raw", lambda *_: None)
    monkeypatch.setattr(session_server.setproctitle, "setproctitle", lambda *_: None)
    monkeypatch.setattr(session_server, "SessionServer", FakeSessionServer)

    def fake_uvicorn_run(received_app, **kwargs):
        uvicorn_call["app"] = received_app
        uvicorn_call.update(kwargs)

    monkeypatch.setattr(session_server.uvicorn, "run", fake_uvicorn_run)

    args = SimpleNamespace(session_server_ip="127.0.0.1", session_server_port=31001)
    session_server.run_session_server(args, "http://backend")

    assert uvicorn_call == {
        "app": app,
        "host": "127.0.0.1",
        "port": 31001,
        "log_level": "info",
        "access_log": False,
    }
