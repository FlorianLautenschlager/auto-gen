from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage, ToolCallSummaryMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core import MessageContext, RoutedAgent, message_handler, TopicId, type_subscription
from autogen_core.tools import FunctionTool
from .messages import ImageGenerationAgentMessage, ImagePathMessage
from . import gpt_image
from colorama import Fore, Back, Style, init

# Initialize colorama (required on Windows)
init()

def generate_image(description: str, filename: str) -> str:
    """Generate an image for a given description and filename."""
    print("[tool] -> Generate Image: Image generation started for image description: " + description)
    return gpt_image.generate_image(description,filename)



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
        print(Back.GREEN)
        print(f"[{self.id.type}] -> received message from {message.source}")

        response = await self._delegate.on_messages(
            [TextMessage(content=message.content, source="assistant")],
            ctx.cancellation_token,
        )

        if (isinstance(response.chat_message,ToolCallSummaryMessage)):
            print(f"[{self.id.type}] <- respond publish message: {response.chat_message.content}")
            print(Style.RESET_ALL)

            await self.publish_message(
                ImagePathMessage(imagePath=response.chat_message.content, source=self.id.type),
                topic_id=TopicId(type="image-publish", source=self.id.key),
            )
        else:
            print(f"[{self.id.type}] <- respond: No image generated. Call cancellation token.")
            print(Style.RESET_ALL)
            ctx.cancellation_token.cancel()

       

