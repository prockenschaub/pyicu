import pytest

from pyicu.concepts import ConceptDict, Concept
from pyicu.concepts.load import read_dictionary


def test_loading_default_dict(default_dict):
    raw = read_dictionary()
    assert [r for r in raw.keys()] == [p for p in default_dict.concepts.keys()]
    assert all([isinstance(p, Concept) for p in default_dict.concepts.values()])


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
