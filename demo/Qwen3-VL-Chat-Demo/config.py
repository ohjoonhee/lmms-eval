import os

from modelscope_studio.components.pro.chatbot import (
    ChatbotActionConfig,
    ChatbotBotConfig,
    ChatbotMarkdownConfig,
    ChatbotUserConfig,
    ChatbotWelcomeConfig,
)
from modelscope_studio.components.pro.multimodal_input import (
    MultimodalInputUploadConfig,
)

# Env
is_cn = False
api_key = os.getenv("API_KEY", "EMPTY")
base_url = "http://localhost:8000/v1"


def get_text(text: str, cn_text: str):
    if is_cn:
        return cn_text
    return text


# Save history in browser
save_history = True
MODEL = "Qwen/Qwen3-VL-32B-Thinking-FP8"
THINKING_MODEL = "Qwen/Qwen3-VL-32B-Thinking-FP8"


# Chatbot Config
def markdown_config():
    return ChatbotMarkdownConfig()


def user_config(disabled_actions=None):
    return ChatbotUserConfig(
        class_names=dict(content="user-message-content"),
        actions=[
            "copy",
            "edit",
            ChatbotActionConfig(
                action="delete",
                popconfirm=dict(title=get_text("Delete the message", "删除消息"), description=get_text("Are you sure to delete this message?", "确认删除该消息？"), okButtonProps=dict(danger=True)),
            ),
        ],
        disabled_actions=disabled_actions,
    )


def bot_config(disabled_actions=None):
    return ChatbotBotConfig(
        actions=[
            "copy",
            "edit",
            ChatbotActionConfig(
                action="retry",
                popconfirm=dict(
                    title=get_text("Regenerate the message", "重新生成消息"),
                    description=get_text("Regenerate the message will also delete all subsequent messages.", "重新生成消息会删除所有后续消息。"),
                    okButtonProps=dict(danger=True),
                ),
            ),
            ChatbotActionConfig(
                action="delete",
                popconfirm=dict(title=get_text("Delete the message", "删除消息"), description=get_text("Are you sure to delete this message?", "确认删除该消息？"), okButtonProps=dict(danger=True)),
            ),
        ],
        avatar="./assets/qwen.png",
        disabled_actions=disabled_actions,
    )


def welcome_config():
    return ChatbotWelcomeConfig(
        variant="borderless",
        icon="./assets/qwen.png",
        title=get_text("Hello, I'm Qwen3-VL", "你好，我是 Qwen3-VL"),
        description=get_text("Enter text and upload images or videos to get started.", "输入文本并上传图片或视频，开始对话吧。"),
        prompts=dict(
            title=get_text("How can I help you today?", "有什么我能帮助您的吗?"),
            styles={
                "list": {
                    "width": "100%",
                },
                "item": {
                    "flex": 1,
                },
            },
            items=[
                {
                    "label": get_text("🤔 Logic Reasoning", "🤔 逻辑推理"),
                    "children": [
                        {
                            "urls": [
                                "https://misc-assets.oss-cn-beijing.aliyuncs.com/Qwen/Qwen3-VL-Demo/r-1-1.png",
                                "https://misc-assets.oss-cn-beijing.aliyuncs.com/Qwen/Qwen3-VL-Demo/r-1-2.png",
                                "https://misc-assets.oss-cn-beijing.aliyuncs.com/Qwen/Qwen3-VL-Demo/r-1-3.png",
                            ],
                            "description": get_text("Which one of these does the kitty seem to want to try first?", "这只猫看起来要尝试先做什么？"),
                        },
                        {
                            "urls": [
                                "https://misc-assets.oss-cn-beijing.aliyuncs.com/Qwen/Qwen3-VL-Demo/r-2.png",
                            ],
                            "description": get_text(
                                "In the circuit, the diodes are ideal and the voltage source is Vs = 4 sin(ωt) V. Find the value measured on the ammeter.",
                                "电路中的 diodes 是理想的，电压源为 Vs = 4 sin(ωt) V。求电流表测量的数值。",
                            ),
                        },
                        {
                            "urls": ["https://misc-assets.oss-cn-beijing.aliyuncs.com/Qwen/Qwen3-VL-Demo/r-3.png"],
                            "description": get_text(
                                "Which is the most popular Friday drink in Boston?\nAnswer the question using a single word or phrase.",
                                " Boston 的星期五饮料中最受欢迎的是什么？\n请用一个单词或短语回答该问题。",
                            ),
                        },
                    ],
                },
                {
                    "label": get_text("👨‍💻 Coding", "👨‍💻 编程"),
                    "children": [
                        {
                            "urls": ["https://misc-assets.oss-cn-beijing.aliyuncs.com/Qwen/Qwen3-VL-Demo/c-1.png"],
                            "description": get_text("Create the webpage using HTML and CSS based on my sketch design. Color it in dark mode.", "基于我的草图设计，用 HTML 和 CSS 创建网页，并暗色模式下颜色。"),
                        },
                        {
                            "urls": ["https://misc-assets.oss-cn-beijing.aliyuncs.com/Qwen/Qwen3-VL-Demo/c-2.png"],
                            "description": get_text(
                                "Solve the problem using C++. Starter code:\nclass Solution {\npublic:\n    int countStableSubsequences(vector<int>& nums) {\n\n    }\n};",
                                "使用 C++ 解决问题。起始代码：\nclass Solution {\npublic:\n    int countStableSubsequences(vector<int>& nums) {\n\n    }\n};",
                            ),
                        },
                        {
                            "urls": ["https://misc-assets.oss-cn-beijing.aliyuncs.com/Qwen/Qwen3-VL-Demo/c-3.png"],
                            "description": get_text("How to draw this plot using matplotlib?", "如何使用 matplotlib 绘制这张图？"),
                        },
                    ],
                },
            ],
        ),
    )


def upload_config():
    return MultimodalInputUploadConfig(
        accept="image/*,video/*",
        placeholder={
            "inline": {"title": "Upload files", "description": "Click or drag files to this area to upload images or videos"},
            "drop": {
                "title": "Drop files here",
            },
        },
    )


DEFAULT_SYS_PROMPT = "You are a helpful and harmless assistant."

DEFAULT_THEME = {
    "token": {
        "colorPrimary": "#6A57FF",
    }
}
