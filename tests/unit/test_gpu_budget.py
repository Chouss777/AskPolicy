"""Offline 8 GiB configuration contract; cannot approve a stack change."""

from dataclasses import FrozenInstanceError
from pathlib import Path
import runpy

import pytest


ROOT = Path(__file__).resolve().parents[2]
module = runpy.run_path(str(ROOT / 'src/askpolicy/generation/gpu_budget.py'))
profile = module['EIGHT_GIB_PROPOSAL']
ApprovalError = module['GpuBudgetApprovalError']


def test_current_stack_unchanged_and_candidate_explicit():
    assert profile.production_model == 'qwen3:4b-instruct'
    assert profile.proposed_generator == 'qwen3.5:4b'
    assert profile.active_embedding_backend == 'e5_cpu'
    assert profile.active_embedding_model == 'intfloat/multilingual-e5-small'
    assert profile.proposed_ollama_embedder == 'qwen3-embedding:0.6b'
    assert profile.reranker_device == 'cpu'
    assert profile.reranker_enabled is False


def test_8gib_daemon_limits_are_explicit_and_local_only():
    assert profile.daemon_environment() == {
        'OLLAMA_HOST': '127.0.0.1:11434',
        'OLLAMA_CONTEXT_LENGTH': '4096',
        'OLLAMA_KEEP_ALIVE': '5m',
        'OLLAMA_MAX_LOADED_MODELS': '1',
        'OLLAMA_NUM_PARALLEL': '1',
        'OLLAMA_MAX_QUEUE': '8',
        'OLLAMA_NO_CLOUD': '1',
    }


def test_generator_limits_are_bounded_non_thinking_and_nonstreaming():
    assert profile.candidate_chat_options() == {
        'model': 'qwen3.5:4b', 'stream': False, 'think': False,
        'keep_alive': '5m',
        'options': {'num_ctx': 4096, 'num_predict': 512, 'temperature': 0},
    }


def test_experimental_embedder_unloads_and_disallows_truncation():
    assert profile.candidate_embedding_options() == {
        'model': 'qwen3-embedding:0.6b', 'truncate': False, 'keep_alive': '0',
    }


def test_distinct_calls_do_not_share_mutable_options():
    before = profile.candidate_chat_options()
    before['options']['num_ctx'] = 99999
    assert profile.candidate_chat_options()['options']['num_ctx'] == 4096
    e = profile.daemon_environment()
    e['OLLAMA_MAX_LOADED_MODELS'] = '10'
    assert profile.daemon_environment()['OLLAMA_MAX_LOADED_MODELS'] == '1'


def test_profile_is_frozen():
    with pytest.raises(FrozenInstanceError):
        profile.generation_num_ctx = 8192


@pytest.mark.parametrize('unapproved', [None, '', 'd8b0f5e9760c', 'X'*64, '0'*64, 1, False])
def test_unapproved_or_partial_digest_rejected(unapproved):
    with pytest.raises(ApprovalError, match='^GPU_PROFILE_NOT_APPROVED$'):
        profile.check_proposed_generator_digest(
            installed_digest=profile.observed_generator_digest,
            owner_approved_digest=unapproved,
        )


@pytest.mark.parametrize('installed', ['', 'd8b0f5e9760c', '0'*64, True, 123, None])
def test_installed_digest_mismatch_rejected(installed):
    with pytest.raises(ApprovalError, match='^GPU_PROFILE_NOT_APPROVED$'):
        profile.check_proposed_generator_digest(
            installed_digest=installed,
            owner_approved_digest=profile.observed_generator_digest,
        )


def test_exact_digest_pair_matches_only_as_format_and_identity_check():
    assert profile.check_proposed_generator_digest(
        installed_digest=profile.observed_generator_digest,
        owner_approved_digest=profile.observed_generator_digest,
    ) is None
    # This tests only digest equality, not documented §18 authorization.
