from app.agents.symptom_agent import DatasetTextIndex, RuleBasedSymptomAgent


def make(lexicon_path, texts=None):
    return RuleBasedSymptomAgent(lexicon_path, DatasetTextIndex(texts) if texts is not None else None)


def canon(mentions):
    return [m.canonical for m in mentions]


def test_extracts_known_symptoms_with_offsets(lexicon_path):
    r = make(lexicon_path).analyze("I have a dry cough and shortness of breath.")
    assert canon(r.symptom_mentions) == ["cough", "shortness_of_breath"]
    for m in r.symptom_mentions:
        assert r.original_text[m.start:m.end] == m.text
    assert r.method == "rule_based_lexicon_v1" and r.confidence is None


def test_unseen_term_is_preserved_not_dropped(lexicon_path):
    r = make(lexicon_path).analyze("A strange tingling in my fingers and a metallic taste.")
    terms = [u.text for u in r.unrecognized_terms]
    assert "strange tingling in my fingers" in terms and "metallic taste" in terms
    assert r.symptom_mentions == [] and r.negated_mentions == []
    assert all(u.in_dataset_text is None for u in r.unrecognized_terms)  # no dataset to check against


def test_unseen_term_checked_against_dataset_text(lexicon_path):
    agent = make(lexicon_path, ["Patient reports metallic taste after medication."])
    r = agent.analyze("metallic taste and fizzing ears")
    by_text = {u.text: u.in_dataset_text for u in r.unrecognized_terms}
    assert by_text["metallic taste"] is True
    assert by_text["fizzing ears"] is False


def test_negation_and_scope_termination(lexicon_path):
    r = make(lexicon_path).analyze("No fever or chills, but I have a cough.")
    assert canon(r.negated_mentions) == ["fever", "chills"]
    assert canon(r.symptom_mentions) == ["cough"]
    r = make(lexicon_path).analyze("Patient denies chest pain.")
    assert r.negated_mentions[0].negation_cue == "denies"


def test_post_negation(lexicon_path):
    r = make(lexicon_path).analyze("The fever has resolved.")
    assert canon(r.negated_mentions) == ["fever"]


def test_spelling_variation_uses_fuzzy_match(lexicon_path):
    r = make(lexicon_path).analyze("feverr and shortness of brething, wheezng")
    got = {m.canonical: m for m in r.symptom_mentions}
    assert set(got) == {"fever", "shortness_of_breath", "wheezing"}
    assert all(m.match_type == "fuzzy" and 0.85 <= m.match_similarity < 1 for m in got.values())


def test_empty_input_is_not_reported_as_absent_symptoms(lexicon_path):
    r = make(lexicon_path).analyze("   ")
    assert r.symptom_mentions == [] and r.normalized_text == ""
    assert "not evidence that symptoms are absent" in r.limitations[0]


def test_contradictory_description_keeps_both(lexicon_path):
    r = make(lexicon_path).analyze("I have a cough. Actually I do not have a cough.")
    assert canon(r.symptom_mentions) == ["cough"] and canon(r.negated_mentions) == ["cough"]
    assert r.contradictions and "cough" in r.contradictions[0]


def test_uncertainty_cue(lexicon_path):
    r = make(lexicon_path).analyze("maybe some wheezing at night")
    m = r.symptom_mentions[0]
    assert m.uncertain and m.uncertainty_cue == "maybe"
    assert any(c.kind == "pattern" and c.text == "at night" for c in r.context_mentions)


def test_duration_and_context_fields(lexicon_path):
    r = make(lexicon_path).analyze("cough for 3 days", context="smoker", duration="since last week")
    durations = [d.text for d in r.duration_mentions]
    assert "for 3 days" in durations and "since last week" in durations
    assert any(c.kind == "history" for c in r.context_mentions)


def test_red_flag_notice_only_when_affirmed(lexicon_path):
    agent = make(lexicon_path)
    assert agent.analyze("crushing chest pain").safety_notice
    assert agent.analyze("no chest pain").safety_notice is None
    notice = agent.analyze("coughing up blood").safety_notice
    assert "cannot assess urgency" in notice and "diagnos" not in notice.lower()
