from types import SimpleNamespace
from unittest.mock import Mock, patch

from frappe.tests.utils import FrappeTestCase

from raven.ai.ai import handle_ai_thread_message, handle_bot_dm


class TestAIRouting(FrappeTestCase):
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
