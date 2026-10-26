from cohesion_taxonomy import naming


def test_parse_name_strips_quotes_and_label_prefix():
    assert naming.parse_name('"Governing Law"\n') == "Governing Law"
    assert naming.parse_name("Name: Notices") == "Notices"
    assert naming.parse_name("") == ""
    assert len(naming.parse_name("x" * 100)) == 60


def test_fallback_name_is_deterministic_and_marked():
    texts = ["Notices. All notices shall be in writing.", "Notices: any notice must be written.",
             "Section 1.2 Notices. Each party shall notify the other."]
    a = naming.fallback_name(texts)
    assert a == naming.fallback_name(texts)
    assert a.endswith("(auto)") and a.startswith("Notices")
    words = naming.fallback_name(["the agreement shall terminate upon notice", "termination of the agreement"])
    assert words.endswith("(auto)")


def test_name_cluster_without_key_makes_no_call():
    assert naming.name_cluster(["Governing Law. Delaware law applies."]) == ("[no api key]", False)


def test_name_cluster_unreachable_endpoint_falls_back(monkeypatch):
    monkeypatch.setenv("NAMING_API_KEY", "test-key-not-a-secret")
    monkeypatch.setattr(naming, "API", "http://127.0.0.1:9/chat/completions")   # nothing listens on tcp/9
    name, ok = naming.name_cluster(["Confidentiality. Each party shall keep the terms confidential."])
    assert ok is False and name.endswith("(auto)")
