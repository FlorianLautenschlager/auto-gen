from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage, ToolCallSummaryMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core import MessageContext, RoutedAgent, message_handler, TopicId, type_subscription
from autogen_core.tools import FunctionTool
from .messages import ImageGenerationAgentMessage, ImagePathMessage
from . import gpt_image

# Define a simple function tool that the agent can use.
# For this example, we use a fake weather tool for demonstration purposes.
def generate_image(description: str, filename: str) -> str:
    """Generate an image for a given prompt."""
    print("[tool] -> Generate Image: Image generation started for image description: " + description)
    return gpt_image.generate_image(description,filename)

stock_price_tool = FunctionTool(generate_image, description="Generate an image for a given prompt. Provide a name for the image to save it (without file type ending).", name="generate_image")


@type_subscription(topic_type="image-critique")
class ImageGenerationAgent(RoutedAgent):
    def __init__(
        self, name: str, model_client: OpenAIChatCompletionClient, system_message: str, description: str,
    ) -> None:
        super().__init__(name)
        self._delegate = AssistantAgent(
            name, model_client=model_client,
            system_message=system_message,
            description=description,
            tools=[generate_image],
        )

    @message_handler
    async def handle_my_message_type(
        self, message: ImageGenerationAgentMessage, ctx: MessageContext
    ) -> None:
        print(f"[{self.id.type}] -> received message: {message.content}")
        response = await self._delegate.on_messages(
            [TextMessage(content=message.content, source="assistant")],
            ctx.cancellation_token,
        )

        print(f"[{self.id.type}] -> responded: {response.chat_message.content}")

        if (isinstance(response.chat_message,ToolCallSummaryMessage)):
            print(f"[{self.id.type}] -> publish message type: {type(response.chat_message.content)}")
            await self.publish_message(
                ImagePathMessage(imagePath=response.chat_message.content),
                topic_id=TopicId(type="image-publish", source=self.id.key),
            )
        else:
            print(f"[{self.id.type}] -> No tool call: {response.chat_message}")
            ctx.cancellation_token.cancel()

       

