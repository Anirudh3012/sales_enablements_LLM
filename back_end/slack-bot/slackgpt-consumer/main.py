import os
import json
import asyncio
import logging
from slack_bolt.async_app import AsyncApp
from slack_sdk.web.async_client import AsyncWebClient
from dotenv import load_dotenv
from fastapi import FastAPI
import aio_pika
from qa_chain import update_conversation_history, get_llm_responses

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Initialize the Slack app with your bot token
app = AsyncApp(
    token=os.environ.get("SLACK_BOT_TOKEN"),
    signing_secret=os.environ.get("SLACK_SIGNING_SECRET")
)

# Initialize FastAPI app
fastapi_app = FastAPI()

# Initialize Slack async client
slack_client = AsyncWebClient(token=os.environ.get("SLACK_BOT_TOKEN"))

async def post_response_to_slack(slack_message, slack_channel, thread_ts):
    """
    Asynchronously posts a response to Slack.
    """
    message = f"\n{slack_message}\n"
    try:
        await slack_client.chat_postMessage(
            channel=slack_channel,
            text=message,
            thread_ts=thread_ts
        )
        logger.info("Message posted to Slack successfully.")
    except Exception as e:
        logger.error(f"Failed to post message to Slack: {e}")

async def on_message(message: aio_pika.IncomingMessage):
    """
    Asynchronous callback to handle messages from RabbitMQ.
    """
    async with message.process():
        data = json.loads(message.body.decode('utf-8'))
        chatgpt_prompt = data.get("prompt")
        slack_channel = data.get("channel")
        thread_ts = data.get("thread_ts")

        # Update and maintain conversation history
        conversation_history = update_conversation_history([], chatgpt_prompt)
        responses = await get_llm_responses([chatgpt_prompt], conversation_history)

        for response in responses:
            message_text = f"Response: {response['result']} | Confidence: {response['confidence_score']:.2f} ({response['confidence_level']})"
            await post_response_to_slack(message_text, slack_channel, thread_ts)


async def consume_messages():
    """
    Sets up RabbitMQ consumer.
    """
    connection = await aio_pika.connect_robust(os.getenv("CLOUDAMQP_URL"))
    channel = await connection.channel()  # Ensure the channel is opened
    queue = await channel.declare_queue('slack_bot_queue')
    await queue.consume(on_message, no_ack=False)
    logger.info("Started consuming messages.")

@fastapi_app.on_event("startup")
async def startup_event():
    """
    Starts up the consumer on application startup.
    """
    logger.info("Starting message consumer...")
    task = asyncio.create_task(consume_messages())
    await task

@fastapi_app.on_event("shutdown")
async def shutdown_event():
    """
    Handles tasks to perform on application shutdown.
    """
    logger.info("Shutting down application...")

# Uncomment the following if you want to run using Uvicorn directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:fastapi_app", host="0.0.0.0", port=4000, reload=True)
