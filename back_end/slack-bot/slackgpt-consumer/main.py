import os
import json
from slack_bolt import App
from dotenv import load_dotenv
from fastapi import FastAPI
import time

# from qa_chain import get_llm_responses
from broker import cloudamqp
# from chatgpt_helper import chagptify_text
# from chatgpt_helper_new import get_llm_response

# Load environment variables from .env file
load_dotenv()

# Initializes your app with your bot token
app = App(
    token=os.environ.get("SLACK_BOT_TOKEN"),
    signing_secret=os.environ.get("SLACK_SIGNING_SECRET")
)
import sys
fastapi_app = FastAPI()


great_grandparent_dir = os.path.abspath(os.path.join(__file__, '../../../..'))

# Add this directory to sys.path
sys.path.insert(0, great_grandparent_dir)
from qa_chain import update_conversation_history
from qa_chain import get_llm_responses

def post_response_to_slack(slack_message, slack_channel, thread_ts):
    message = f"\n {slack_message} \n"
    
    app.client.chat_postMessage(
        channel=slack_channel,
        text=message,
        thread_ts=thread_ts
    )


def callback(ch, method, properties, body):
    """Handle message from CloudAMQP and respond in Slack."""
    body = json.loads(body.decode('utf-8'))
    chatgpt_prompt = body.get("prompt")
    slack_channel = body.get("channel")
    thread_ts = body.get("thread_ts")

    # Simulating the processing of the prompt and maintaining conversation history
    conversation_history = update_conversation_history([], chatgpt_prompt)  # Assuming a function to manage history
    queries = [chatgpt_prompt]

    start_time = time.time()
    responses = get_llm_responses(queries, conversation_history)  # This should be an async function
    response_time = time.time() - start_time

    for i, response in enumerate(responses):
        card_text = response["result"]
        confidence_score = response["confidence_score"]
        confidence_level = response["confidence_level"]

        # Update conversation history with the response
        conversation_history = update_conversation_history(conversation_history, card_text)

        # Send the response back to Slack
        message = f"Response Time: {response_time:.2f}s, Score: {confidence_score:.2f} ({confidence_level})"
        post_response_to_slack(message, slack_channel, thread_ts)


def main():
    cloudamqp.consume_message(callback=callback)
    print('message consumed')
    

# Run the app with a FastAPI server
@fastapi_app.on_event("startup")
def startup_event():
    """ Code to run during startup """
    main()


@fastapi_app.on_event("shutdown")
def shutdown_event():
    """ Code to run during shutdown """
    pass

# if __name__ == "__main__":
#     app.start(port=int(os.environ.get("PORT", 4000)))