import base64
from io import BytesIO
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

import frappe
from PIL import Image

from raven.ai.agents_integration import handle_ai_request_sync
from raven.ai.ai import handle_ai_thread_message, process_message_with_agent
from raven.raven_messaging.doctype.raven_message.raven_message import RavenMessage


class TestChatImageInput(TestCase):
	def setUp(self):
		self.bot = SimpleNamespace(
			name="PERI",
			model_provider="OpenAI",
			model="gpt-4o",
			debug_mode=False,
			send_message=Mock(),
		)
		self.image_message = SimpleNamespace(
			name="photo-message",
			owner="worker@example.test",
			message_type="Image",
			file="/private/files/photo.jpg?fid=FILE-PHOTO",
			text=None,
			content=None,
		)
		image = BytesIO()
		Image.new("RGB", (16, 16), "blue").save(image, "JPEG")
		self.photo = image.getvalue()
		self.file = Mock(is_remote_file=False, file_size=len(self.photo))
		self.file.get_content.return_value = self.photo
		self.get_doc = self.enterContext(patch("raven.ai.ai.frappe.get_doc", return_value=self.file))
		self.get_value = self.enterContext(patch("raven.ai.ai.frappe.db"))
		self.get_value.get_value.return_value = "FILE-PHOTO"
		self.publish = self.enterContext(patch("raven.ai.ai.frappe.publish_realtime"))
		self.enterContext(patch("raven.ai.ai.frappe._", side_effect=lambda message: message))
		self.enterContext(patch("raven.ai.ai.extract_file_content_for_agent", return_value=""))
		self.enterContext(patch("raven.ai.conversation_file_handler.ConversationFileHandler"))

	def test_upload_sends_image_with_or_without_a_caption(self):
		with (
			patch.object(frappe, "get_cached_doc", return_value=self.bot),
			patch.object(frappe, "get_all", return_value=[]),
			patch(
				"raven.ai.ai.handle_ai_request_sync",
				return_value={"success": True, "response": "The photo is blue."},
			) as run,
		):
			for caption in (None, "What colour is the photo?"):
				with self.subTest(caption=caption):
					run.reset_mock()
					self.image_message.text = caption
					self.image_message.content = caption or "photo.jpg"
					if caption:
						process_message_with_agent(self.image_message, self.bot, "photo-thread", True)
					else:
						channel = SimpleNamespace(name="photo-thread", thread_bot="PERI")
						handle_ai_thread_message(self.image_message, channel)

					run.assert_called_once()
					request = run.call_args.kwargs
					self.assertIn(caption or "Analyse this photo.", request["message"])
					prefix, encoded_photo = request["image_url"].split(",", 1)
					self.assertEqual(prefix, "data:image/jpeg;base64")
					self.assertEqual(base64.b64decode(encoded_photo), self.photo)
					self.bot.send_message.assert_called_with(channel_id="photo-thread", text="The photo is blue.")
		self.file.has_permission.assert_called_with("read", user=self.image_message.owner)
		self.get_doc.assert_called_with("File", "FILE-PHOTO")

	def test_follow_up_can_use_the_photo_that_started_a_direct_message_thread(self):
		message = SimpleNamespace(
			owner=self.image_message.owner,
			creation="2026-11-14 10:00:00",
			message_type="Text",
			text="Is there a label?",
		)
		channel = SimpleNamespace(name=self.image_message.name, is_dm_thread=True)
		self.get_value.get_value.side_effect = [None, self.image_message, "FILE-PHOTO"]
		with (
			patch("raven.ai.ai.frappe.get_all", return_value=[]),
			patch(
				"raven.ai.ai.handle_ai_request_sync",
				return_value={"success": True, "response": "Done"},
			) as run,
		):
			process_message_with_agent(message, self.bot, channel.name, False, channel)

		self.assertEqual(run.call_args.kwargs["message"], message.text)
		self.assertTrue(run.call_args.kwargs["image_url"].startswith("data:image/jpeg;base64,"))

	def test_attachment_without_read_permission_does_not_reach_the_model(self):
		self.file.has_permission.return_value = False
		with patch("raven.ai.ai.handle_ai_request_sync") as run:
			process_message_with_agent(self.image_message, self.bot, "photo-thread", True)

		run.assert_not_called()
		self.file.get_content.assert_not_called()
		self.assertIn("couldn't open the photo", self.bot.send_message.call_args.kwargs["text"])

	def test_agent_runner_accepts_text_and_photo_inputs(self):
		with (
			patch("raven.ai.agents_integration.RavenAgentManager") as manager,
			patch("raven.ai.agents_integration.Runner.run") as run,
		):
			run.return_value = SimpleNamespace(final_output="Hello")
			agent = manager.return_value.create_agent.return_value
			self.assertTrue(handle_ai_request_sync(self.bot, "Hi", "text-thread")["success"])
			run.assert_awaited_once_with(agent, "Hi", max_turns=5, run_config=None)

			run.reset_mock()
			image_url = "data:image/jpeg;base64," + base64.b64encode(self.photo).decode("ascii")
			result = handle_ai_request_sync(
				self.bot, "Analyse this photo.", "photo-thread", image_url=image_url
			)
			self.assertTrue(result["success"])
			self.assertEqual(
				run.await_args.args[1],
				[
					{
						"role": "user",
						"content": [
							{"type": "input_text", "text": "Analyse this photo."},
							{"type": "input_image", "image_url": image_url, "detail": "high"},
						],
					}
				],
			)
			self.assertFalse(run.await_args.kwargs["run_config"].trace_include_sensitive_data)

	def test_raven_queues_ai_jobs_after_the_message_and_upload_commit(self):
		for is_thread in (False, True):
			with self.subTest(is_thread=is_thread):
				message = SimpleNamespace(
					channel_id="photo-channel", owner=self.image_message.owner, is_bot_message=0
				)
				channel = SimpleNamespace(is_ai_thread=is_thread, is_direct_message=not is_thread)
				settings = SimpleNamespace(enable_ai_integration=1)
				bot_user = SimpleNamespace(type="Bot", bot="PERI")
				bot = SimpleNamespace(is_ai_bot=1)
				with (
					patch.object(frappe, "get_cached_doc", side_effect=[settings, channel, bot]),
					patch.object(frappe, "get_cached_value", return_value=bot_user),
					patch.object(frappe, "enqueue") as enqueue,
					patch(
						"raven.raven_messaging.doctype.raven_message.raven_message.get_peer_user_id_from_dm_users",
						return_value="PERI",
					),
				):
					RavenMessage.handle_ai_message(message)

				self.assertTrue(enqueue.call_args.kwargs["enqueue_after_commit"])
				self.assertIs(enqueue.call_args.kwargs["message"], message)
