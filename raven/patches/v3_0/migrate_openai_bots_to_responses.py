import frappe


def execute():
	"""Stop existing bots from retaining references to retired Assistants."""
	frappe.db.set_value(
		"Raven Bot",
		{"openai_assistant_id": ["is", "set"]},
		"openai_assistant_id",
		None,
		update_modified=False,
	)
