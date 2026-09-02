from types import SimpleNamespace
from unittest.mock import Mock, patch

from frappe.tests.utils import FrappeTestCase

from raven.ai.agents_integration import get_model_settings
from raven.ai.ai import handle_ai_thread_message, handle_bot_dm


class TestAIRouting(FrappeTestCase):
	def test_reasoning_model_omits_sampling_parameters(self):
		bot = SimpleNamespace(
			model_provider="OpenAI",
			model="gpt-5.6-terra",
			reasoning_effort="medium",
			temperature=0.7,
			top_p=0.9,
		)

		settings = get_model_settings(bot)

		self.assertIsNone(settings.temperature)
		self.assertIsNone(settings.top_p)
		self.assertEqual(settings.reasoning_effort, "medium")

	def test_non_reasoning_model_keeps_sampling_parameters(self):
		bot = SimpleNamespace(
			model_provider="OpenAI",
			model="gpt-4o",
			reasoning_effort=None,
			temperature=0.7,
			top_p=0.9,
		)

		settings = get_model_settings(bot)

		self.assertEqual(settings.temperature, 0.7)
		self.assertEqual(settings.top_p, 0.9)
		self.assertIsNone(settings.reasoning_effort)

	def test_local_model_keeps_sampling_parameters(self):
		bot = SimpleNamespace(
			model_provider="Local LLM",
			model="gpt-5-compatible-local-model",
			reasoning_effort=None,
			temperature=0.5,
			top_p=0.8,
		)

		settings = get_model_settings(bot)

		self.assertEqual(settings.temperature, 0.5)
		self.assertEqual(settings.top_p, 0.8)

	@patch("raven.ai.ai.handle_bot_dm_with_agents")
	def test_direct_message_uses_agents_for_legacy_bot(self, agents_handler: Mock):
		bot = SimpleNamespace(model_provider="OpenAI", openai_assistant_id="asst_legacy")
		message = Mock()

		handle_bot_dm(message, bot)

		agents_handler.assert_called_once_with(message, bot)

	@patch("raven.ai.ai.handle_ai_thread_message_with_agents")
	@patch("raven.ai.ai.frappe.get_cached_doc")
	def test_thread_message_uses_agents_for_legacy_thread(
		self, get_cached_doc: Mock, agents_handler: Mock
	):
		bot = SimpleNamespace(model_provider="OpenAI", openai_assistant_id="asst_legacy")
		channel = SimpleNamespace(thread_bot="P.E.R.I.")
		message = Mock()
		get_cached_doc.return_value = bot

		handle_ai_thread_message(message, channel)

		get_cached_doc.assert_called_once_with("Raven Bot", "P.E.R.I.")
		agents_handler.assert_called_once_with(message, channel, bot)
