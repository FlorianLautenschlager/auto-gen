from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core import (
    MessageContext,
    RoutedAgent,
    message_handler,
    type_subscription,
    Image,
    TopicId,
)
from .messages import ImagePathMessage, ImageGenerationAgentMessage


@type_subscription(topic_type="image-publish")
class ImageCriticAgent(RoutedAgent):
    def __init__(
        self,
        name: str,
        model_client: OpenAIChatCompletionClient,
        system_message: str,
        description: str,
    ) -> None:
        super().__init__(name)
        self.model_client = model_client
        self.system_message = system_message
        self._delegate = AssistantAgent(
            name,
            model_client=model_client,
            system_message=system_message,
            description=description,
        )

    @message_handler
    async def handle_my_message_type(
        self, message: ImagePathMessage, ctx: MessageContext
    ) -> None:
        print(f"[{self.id.type}] -> received message from {message.source}")

        print(f"[{self.id.type}] -> starting to critique image: {message.imagePath}")

        image = Image.from_file(message.imagePath)

        messages = [
            UserMessage(content=[image], source="user"),
            UserMessage(
                content=self.system_message,
                source="assistant",
            ),
        ]

        resp = await self.model_client.create(messages=messages)

        print(f"[{self.id.type}] <- respond critique of image: {resp.content}")

        await self.publish_message(
            ImageGenerationAgentMessage(
                content=resp.content, source=self.id.type
            ),
            topic_id=TopicId(type="image-critique", source=self.id.key),
        )
