import os
import unittest
from unittest.mock import MagicMock, patch
from PIL import Image
import tempfile
import shutil

# Make sure we can import from lmms_eval
import sys

sys.path.append(os.getcwd())

from lmms_eval.models.chat.qwen_agent import QwenAgent
from lmms_eval.api.instance import Instance


class TestQwenAgent(unittest.TestCase):
    def setUp(self):
        self.mock_agent_patcher = patch("lmms_eval.models.chat.qwen_agent.Assistant")
        self.mock_assistant_cls = self.mock_agent_patcher.start()
        self.mock_agent_instance = MagicMock()
        self.mock_assistant_cls.return_value = self.mock_agent_instance

        self.agent_model = QwenAgent()

    def tearDown(self):
        self.mock_agent_patcher.stop()

    def test_init(self):
        """Test initialization of QwenAgent"""
        self.mock_assistant_cls.assert_called_once()
        self.assertEqual(self.agent_model.tools, ["image_zoom_in_tool"])

    def test_generate_until_mock(self):
        """Test generate_until with mocked agent response"""

        # Mock the run method to yield a response
        mock_response = [{"role": "assistant", "content": "The answer is 42."}]
        self.mock_agent_instance.run.return_value = [mock_response]  # Mocking run to return a list (as it iterates)

        # Create a dummy request
        image = Image.new("RGB", (60, 30), color="red")
        instance = Instance(
            request_type="generate_until",
            arguments=("What color is this?",),
            idx=0,
            metadata={"task": "test", "doc_id": 0, "repeats": 1},
        )
        instance.visuals = [image]
        responses = self.agent_model.generate_until([instance])

        self.assertEqual(len(responses), 1)
        self.assertEqual(responses[0], "The answer is 42.")

        # Verify the call to agent.run
        input_args = self.mock_agent_instance.run.call_args
        self.assertIsNotNone(input_args)
        messages = input_args.kwargs.get("messages")
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        content = messages[0]["content"]
        # content should be list of dicts: image, text
        self.assertEqual(len(content), 2)
        self.assertTrue("image" in content[0])
        self.assertEqual(content[1]["text"], "What color is this?")


if __name__ == "__main__":
    unittest.main()
