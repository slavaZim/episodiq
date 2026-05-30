"""Tests for annotation prompt constants."""

from episodiq.clustering.annotator.constants import get_prompt


class TestGetPrompt:

    def test_solo_observation_text(self):
        prompt = get_prompt("observation", "text", contrastive=False)
        assert "user messages" in prompt
        assert "user ..." in prompt

    def test_contrastive_action_text(self):
        prompt = get_prompt("action", "text", contrastive=True)
        assert "TARGET" in prompt
        assert "SIMILAR" in prompt
        assert "agent ..." in prompt

    def test_tool_category_in_prompt(self):
        prompt = get_prompt("observation", "bash", contrastive=False)
        assert "tool bash" in prompt

    def test_action_tool_category(self):
        prompt = get_prompt("action", "editor", contrastive=False)
        assert "tool editor" in prompt

    def test_contrastive_has_examples_section(self):
        prompt = get_prompt("observation", "text", contrastive=True)
        assert "GOOD:" in prompt
        assert "BAD:" in prompt
