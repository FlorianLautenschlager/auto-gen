import asyncio

from dotenv import load_dotenv
import os

from autogen_core import SingleThreadedAgentRuntime, AgentId
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from core import ImageGenerationAgent, ImageGenerationAgentMessage, ImageCriticAgent
from colorama import Fore, Back, Style, init

# Initialize colorama (required for Windows/PowerShell)
init()

load_dotenv(override=True)

async def main() -> None:

    model_client = AzureOpenAIChatCompletionClient(
        model=os.getenv("GPT_41_DEPLOYMENT_NAME"),
        api_key=os.getenv("GPT_41_AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("GPT_41_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("GPT_41_AZURE_OPENAI_ENDPOINT"),
    )

    runtime = SingleThreadedAgentRuntime()

    await ImageGenerationAgent.register(
        runtime,
        "image_generation_agent",
        lambda: ImageGenerationAgent(
            name="image_generation_agent",
            model_client=model_client,
            system_message="""
          You are a creative assistant.
          You can generate images using GPT-1. 
          
          Accept critics and improve the image generation process until the image is 'APPROVED'
          """,
            description="Creates and displays an image",
        ),
    )

    await ImageCriticAgent.register(
        runtime,
        "image_critic_agent",
        lambda: ImageCriticAgent(
            name="image_critic_agent",
            model_client=model_client,
            system_message="""
        You need to improve the prompt of the figures you saw.
        How to create a figure that is better in terms of color, shape, text (clarity), and other things.

        Respond with 'APPROVE' to when your feedbacks are almost addressed.""",
            description="Provides constructive feedback and critics an image",
        ),
    )

    print(Fore.YELLOW + Back.BLUE + "Starting runtime and agents...")
    print(Style.RESET_ALL)
    runtime.start()

    user_input = input("Press Enter to start image generation or type 'exit' to quit, e.g. \n\n'Generate a flyer for my eye clinic that shows a hyperrealistic image of a human blue eye and the text 'Eye Clinic' in the center of the image. The flyer should be colorful and eye-catching'.\n\n What do you want to generate? ")
    if user_input.lower == "exit":
        runtime.stop()
        print("Exiting...")
        return

    await runtime.send_message(
        ImageGenerationAgentMessage(
            content=user_input,
              source="User"
        ),
        AgentId("image_generation_agent", "default"),
    )

    await runtime.stop_when_idle()
    print(Fore.YELLOW + Back.BLUE + "Runtime and agents stopped...")
    print(Style.RESET_ALL)


asyncio.run(main())
