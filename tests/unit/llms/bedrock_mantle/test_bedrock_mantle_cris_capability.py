"""CRIS inference-profile ids (global./us./eu./...) keep the model's Responses capability.

Pure functions of (model, model_cost): synthetic catalogue, no AWS, no network.
"""

import pytest

from litellm.llms.bedrock_mantle.common_utils import mantle_base_segment, mantle_supports_responses

MODEL = "openai.gpt-6-astra"
CATALOG = {
    f"bedrock_mantle/{MODEL}": {"mode": "responses", "use_openai_responses_path": True},
    "bedrock_mantle/openai.gpt-oss-120b": {"supported_endpoints": ["/v1/chat/completions", "/v1/responses"]},
    "bedrock_mantle/openai.gpt-oss-safeguard-120b": {"mode": "chat"},
}


@pytest.mark.parametrize("prefix", ["global", "us", "eu", "apac", "us-gov"])
def test_profile_qualified_model_keeps_declared_capability(prefix):
    model = f"{prefix}.{MODEL}"
    assert mantle_supports_responses(model, CATALOG) is True
    assert mantle_base_segment(model, CATALOG) == "openai/v1"


@pytest.mark.parametrize("region", ["us-east-1", "us-west-2", "eu-central-1"])
@pytest.mark.parametrize("prefix", ["", "global.", "us.", "eu.", "apac.", "us-gov."])
def test_region_prefix_is_normalized_before_cris_lookup(region, prefix):
    model = f"{region}/{prefix}{MODEL}"
    assert mantle_supports_responses(model, CATALOG) is True
    assert mantle_base_segment(model, CATALOG) == "openai/v1"


def test_explicit_profile_entry_takes_precedence():
    catalog = {**CATALOG, f"bedrock_mantle/global.{MODEL}": {"mode": "chat", "use_openai_responses_path": False}}
    for model in (f"global.{MODEL}", f"us-west-2/global.{MODEL}"):
        assert mantle_supports_responses(model, catalog) is False
        assert mantle_base_segment(model, catalog) == "v1"


def test_capability_is_per_model_not_per_family():
    assert mantle_supports_responses("openai.gpt-oss-120b", CATALOG) is True
    assert mantle_supports_responses("global.openai.gpt-oss-120b", CATALOG) is True
    assert mantle_supports_responses("openai.gpt-oss-safeguard-120b", CATALOG) is False
    assert mantle_supports_responses("global.openai.gpt-oss-safeguard-120b", CATALOG) is False


@pytest.mark.parametrize(
    "model",
    [None, "", "global.", "global.nonexistent", f"notaprefix.{MODEL}", f"global.global.{MODEL}",
     f"not-a-region/{MODEL}", f"us-east-1/global.global.{MODEL}", "us-east-1/global."],
)
def test_unknown_and_malformed_names_do_not_gain_capability(model):
    assert mantle_supports_responses(model, CATALOG) is False
    assert mantle_base_segment(model, CATALOG) == "v1"


def test_lookup_does_not_mutate_catalog():
    before = {k: dict(v) for k, v in CATALOG.items()}
    mantle_supports_responses(f"global.{MODEL}", CATALOG)
    mantle_base_segment(f"global.{MODEL}", CATALOG)
    assert CATALOG == before


# register_model() can canonicalize a bedrock_mantle registration onto an existing
# bare Runtime catalog key (e.g. global.openai.gpt-6.1-sol). Accept that spelling
# only when it explicitly belongs to bedrock_mantle: otherwise Responses routes
# through OpenAILikeResponsesConfig, which keeps the AWS URL but drops SigV4.
NATIVE = {"litellm_provider": "bedrock_mantle", "mode": "responses", "use_openai_responses_path": True}


@pytest.mark.parametrize("wire", ["global.openai.gpt-6.1-sol", "global.xai.grok-4.7", "openai.gpt-6-astra-minor"])
@pytest.mark.parametrize("region", ["", "us-east-1/", "eu-central-1/"])
def test_canonicalized_bare_entry_keeps_native_responses(wire, region):
    catalog = {wire: dict(NATIVE)}
    assert mantle_supports_responses(f"{region}{wire}", catalog) is True
    assert mantle_base_segment(f"{region}{wire}", catalog) == "openai/v1"
    assert catalog == {wire: NATIVE}


@pytest.mark.parametrize("prefix", ["global", "us", "eu", "apac", "us-gov"])
def test_canonicalized_unqualified_fallback(prefix):
    catalog = {MODEL: {"litellm_provider": "bedrock_mantle", "supported_endpoints": ["/v1/responses"], "use_openai_responses_path": True}}
    assert mantle_supports_responses(f"{prefix}.{MODEL}", catalog) is True
    assert mantle_base_segment(f"{prefix}.{MODEL}", catalog) == "openai/v1"
    assert mantle_supports_responses(f"global.global.{MODEL}", catalog) is False
    assert mantle_supports_responses(f"not-a-region/{MODEL}", catalog) is False


@pytest.mark.parametrize("provider", [None, "bedrock", "bedrock_converse", "openai", "xai", ""])
def test_bare_entries_of_other_providers_are_ignored(provider):
    entry = {**NATIVE, "litellm_provider": provider}
    catalog = {f"global.{MODEL}": entry, MODEL: entry}
    assert mantle_supports_responses(f"global.{MODEL}", catalog) is False
    assert mantle_base_segment(f"global.{MODEL}", catalog) == "v1"


def test_exact_canonicalized_negative_beats_unqualified_positive():
    catalog = {
        f"global.{MODEL}": {"litellm_provider": "bedrock_mantle", "mode": "chat", "use_openai_responses_path": False},
        f"bedrock_mantle/{MODEL}": {"mode": "responses", "use_openai_responses_path": True},
    }
    assert mantle_supports_responses(f"global.{MODEL}", catalog) is False
    assert mantle_base_segment(f"global.{MODEL}", catalog) == "v1"
    # The provider-prefixed exact declaration is still the most specific.
    catalog[f"bedrock_mantle/global.{MODEL}"] = {"mode": "responses", "use_openai_responses_path": True}
    assert mantle_supports_responses(f"global.{MODEL}", catalog) is True
    assert mantle_base_segment(f"global.{MODEL}", catalog) == "openai/v1"
