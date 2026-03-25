from packages.core.db import session as session_module


def test_get_db_yields_session_and_closes_it(monkeypatch):
    events = []

    class FakeSession:
        def close(self):
            events.append("closed")

    fake_session = FakeSession()
    monkeypatch.setattr(session_module, "SessionLocal", lambda: fake_session)

    generator = session_module.get_db()

    assert next(generator) is fake_session

    try:
        next(generator)
    except StopIteration:
        pass

    assert events == ["closed"]
