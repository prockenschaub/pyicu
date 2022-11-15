import pytest

from pyicu.concepts.load import config_paths
from pyicu.concepts import ConceptDict, Concept
from pyicu.concepts.load import read_dictionary, combine_concepts


@pytest.fixture
def unparsed_sofa_concept():
    return {"sofa": read_dictionary()["sofa"]}


@pytest.fixture
def unparsed_hr_concept():
    return {"hr": read_dictionary()["hr"]}


def test_read_dictionary_cfg_dirs():
    dict0 = read_dictionary()
    dict1 = read_dictionary(cfg_dirs=config_paths()[0])
    dict2 = read_dictionary(cfg_dirs=config_paths())
    assert dict0 == dict1
    assert dict1 == dict2


def test_loading_default_dict(default_dict):
    raw = read_dictionary()
    assert [r for r in raw.keys()] == [p for p in default_dict.concepts.keys()]
    assert all([isinstance(p, Concept) for p in default_dict.concepts.values()])


def test_loading_two_dicts(unparsed_hr_concept):
    alt = {"hr": {"sources": {"mimic": [{"ids": 42, "table": "chartevents", "sub_var": "itemid"}]}}}
    fin = combine_concepts(unparsed_hr_concept, alt)
    assert fin["hr"]["sources"]["mimic"][0]["ids"] == 42


def test_loading_two_dicts_no_overlap(unparsed_hr_concept, unparsed_sofa_concept):
    combine_concepts(unparsed_sofa_concept, unparsed_hr_concept)


def test_loading_two_dicts_rec(unparsed_sofa_concept):
    alt = {"sofa": {"sources": {}}}
    with pytest.raises(ValueError) as e_info:
        combine_concepts(unparsed_sofa_concept, alt)
    assert e_info.match("cannot merge recursive")


def test_loading_two_dicts_malformed(unparsed_hr_concept):
    alt = {"hr": {"sources": [1]}}
    with pytest.raises(ValueError) as e_info:
        combine_concepts(unparsed_hr_concept, alt)
    assert e_info.match("cannot merge .* due to malformed")


def test_loading_two_dicts_non_sources(unparsed_hr_concept):
    alt = {"hr": {"max": 5000, "sources": {}}}
    with pytest.raises(ValueError) as e_info:
        combine_concepts(unparsed_hr_concept, alt)
    assert e_info.match("cannot merge .* due to non-`sources`")


def test_getting_single_concept_from_dict(default_dict):
    assert default_dict["hr"] == default_dict.concepts["hr"]
    assert isinstance(default_dict["hr"], Concept)


def test_getting_multiple_concepts_from_dict(default_dict):
    assert isinstance(default_dict[["hr", "resp"]], ConceptDict)
    assert list(default_dict[["hr", "resp"]].concepts.keys()) == ["hr", "resp"]


def test_dict_index_type_error(default_dict):
    with pytest.raises(TypeError) as e_info:
        default_dict["hr", "resp"]
    assert e_info.match("cannot index `ConceptDict` with <class 'tuple'>")


def test_merging_dicts(default_dict):
    dict1 = default_dict[["hr"]]
    dict2 = default_dict[["resp"]]
    merged = dict1.merge(dict2)
    assert list(merged.concepts.keys()) == ["hr", "resp"]


def test_merging_dicts_overwrite(default_dict):
    dict1 = default_dict[["hr", "resp"]]
    dict2 = default_dict[["resp"]]
    merged = dict1.merge(dict2, overwrite=True)
    assert merged["resp"] == dict2["resp"]


def test_merging_dicts_error(default_dict):
    dict1 = default_dict[["hr", "resp"]]
    dict2 = default_dict[["resp"]]
    with pytest.raises(ValueError) as e_info:
        dict1.merge(dict2)
    assert e_info.match("duplicate concepts found when merging")
