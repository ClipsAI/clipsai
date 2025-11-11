#!/usr/bin/env python3
"""
Test script for council voting system.

Tests the CouncilVoter class and integration with VVSA.
"""
import sys
import os
import asyncio

# Add project to path
sys.path.insert(0, '/home/user/clipsai')


def test_council_imports():
    """Test that all council modules can be imported."""
    print("=" * 70)
    print("TEST 1: Module Imports")
    print("=" * 70)

    try:
        from adlab.council import CouncilVoter, CouncilVote, LocalFallbackModel
        print("✓ adlab.council imports successfully")

        from adlab.vvsa import HybridScorer, create_hybrid_scorer
        print("✓ adlab.vvsa HybridScorer imports successfully")

        from adlab.config import Config
        print("✓ adlab.config imports successfully")

        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_council_initialization():
    """Test council voter initialization."""
    print("\n" + "=" * 70)
    print("TEST 2: Council Voter Initialization")
    print("=" * 70)

    try:
        from adlab.council import CouncilVoter

        # Test with no API keys (should only have local fallback)
        voter = CouncilVoter(
            anthropic_api_key=None,
            openai_api_key=None,
            google_api_key=None
        )

        print(f"✓ CouncilVoter initialized")
        print(f"  Active models: {list(voter.models.keys())}")
        print(f"  Voting method: {voter.voting_method}")
        print(f"  Weights: {voter.weights}")

        stats = voter.get_voting_stats()
        print(f"  Stats: {stats}")

        return True
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_local_fallback_scoring():
    """Test local fallback model scoring."""
    print("\n" + "=" * 70)
    print("TEST 3: Local Fallback Model Scoring")
    print("=" * 70)

    try:
        from adlab.council import LocalFallbackModel

        model = LocalFallbackModel()

        # Test various hook texts
        test_hooks = [
            "What's the secret to viral videos?",
            "This hack changed everything",
            "You won't believe what happened next",
            "Simple tutorial",
            "Hi there this is a very long transcript that goes on and on and on without much substance or interesting content at all"
        ]

        for hook in test_hooks:
            result = model.score_hook(hook)
            print(f"\nHook: '{hook[:50]}...'")
            print(f"  Score: {result['score']:.2f}")
            print(f"  Reasoning: {result['reasoning']}")

        return True
    except Exception as e:
        print(f"✗ Scoring failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_council_voting():
    """Test council voting on sample clips."""
    print("\n" + "=" * 70)
    print("TEST 4: Council Voting")
    print("=" * 70)

    try:
        from adlab.council import CouncilVoter

        voter = CouncilVoter(
            voting_method="weighted_average",
            enable_parallel=False  # Sequential for testing
        )

        # Sample clips with hook text
        sample_clips = [
            ("What's the secret to making $10k a month?", "clip_1"),
            ("Today I'll show you how to cook pasta.", "clip_2"),
            ("You won't believe this insane trick!", "clip_3"),
            ("This is how you fix your credit score fast.", "clip_4"),
            ("Random boring content here nothing special.", "clip_5"),
        ]

        print(f"\nVoting on {len(sample_clips)} sample clips...")

        for hook, clip_id in sample_clips:
            vote = voter.vote_on_clip(
                transcript=hook,
                clip_id=clip_id
            )

            print(f"\nClip: {clip_id}")
            print(f"  Hook: '{hook}'")
            print(f"  Consensus Score: {vote.consensus_score:.2f}")
            print(f"  Agreement Level: {vote.agreement_level:.3f}")
            print(f"  Individual Scores: {vote.individual_scores}")

        return True
    except Exception as e:
        print(f"✗ Voting failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hybrid_scorer():
    """Test hybrid VVSA + Council scorer."""
    print("\n" + "=" * 70)
    print("TEST 5: Hybrid Scorer (VVSA + Council)")
    print("=" * 70)

    try:
        from adlab.vvsa import create_hybrid_scorer
        from adlab.config import Config

        # Create config
        config = Config()

        # Override some settings for testing
        config.data["council"] = {
            "enabled": True,
            "voting_method": "weighted_average",
            "enable_parallel": False
        }

        # Create hybrid scorer
        hybrid = create_hybrid_scorer(config)

        print(f"✓ Hybrid scorer created")
        print(f"  VVSA scorer: {hybrid.vvsa_scorer}")
        print(f"  Council voter: {hybrid.council_voter}")
        print(f"  Council enabled: {hybrid.use_council}")

        # Test with mock clips (we can't easily test full pipeline without video)
        print("\n  Note: Full pipeline test requires video file")
        print("  Use orchestrator for end-to-end testing")

        return True
    except Exception as e:
        print(f"✗ Hybrid scorer failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_orchestrator_integration():
    """Test orchestrator integration."""
    print("\n" + "=" * 70)
    print("TEST 6: Orchestrator Integration")
    print("=" * 70)

    try:
        from clipfactory.processing.orchestrator import ClipFactoryOrchestrator

        config = {
            "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
        }

        orchestrator = ClipFactoryOrchestrator(config)

        print("✓ Orchestrator initialized")
        print("  Note: phase1_council_deliberation requires video file")
        print("  Check orchestrator.py lines 145-310 for implementation")

        # Verify method exists
        assert hasattr(orchestrator, 'phase1_council_deliberation')
        assert hasattr(orchestrator, '_extract_hook_text')
        print("✓ Required methods exist")

        return True
    except Exception as e:
        print(f"✗ Orchestrator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("COUNCIL VOTING SYSTEM TEST SUITE")
    print("=" * 70)

    results = []

    # Run tests
    results.append(("Module Imports", test_council_imports()))
    results.append(("Council Initialization", test_council_initialization()))
    results.append(("Local Fallback Scoring", test_local_fallback_scoring()))
    results.append(("Council Voting", test_council_voting()))
    results.append(("Hybrid Scorer", test_hybrid_scorer()))
    results.append(("Orchestrator Integration", test_orchestrator_integration()))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
