import datetime
from datetime import timedelta
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import os
from document_processing import enhance_metadata_with_bertopic_and_advanced_ner, Document, embed_documents_in_batches
from langchain_community.embeddings import OpenAIEmbeddings  # Correct import
from sentence_transformers import SentenceTransformer
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")

CHANNEL_IDS = ["C06HD8ADMC5"]  # Replace with actual channel IDs
client = WebClient(token=SLACK_BOT_TOKEN)

import asyncio

async def fetch_messages(channel_id):
    messages = []
    try:
        result = await client.conversations_history(channel=channel_id)
        for message in result['messages']:
            user_id = message.get('user', 'unknown')
            text = message.get('text', '')
            timestamp = datetime.fromtimestamp(float(message.get('ts', 0)))
            channel_info = await client.conversations_info(channel=channel_id)
            channel_name = channel_info['channel']['name']
            user_info = await client.users_info(user=user_id)
            user_name = user_info['user']['name']
            messages.append({
                "channel_id": channel_id,
                "channel_name": channel_name,
                "user_id": user_id,
                "user_name": user_name,
                "text": text,
                "timestamp": timestamp
            })
    except SlackApiError as e:
        print(f"Error fetching messages from channel {channel_id}: {e.response['error']}")
    return messages

async def fetch_messages_from_slack():
    all_messages = []
    tasks = [fetch_messages(channel_id) for channel_id in CHANNEL_IDS]
    results = await asyncio.gather(*tasks)
    for result in results:
        all_messages.extend(result)
    return all_messages

def group_messages_into_threads(messages, time_threshold=timedelta(minutes=5)):
    """
    Group consecutive messages by the same user within a short time frame or by thread.
    """
    threads = []
    current_thread = []
    previous_message = None

    for message in messages:
        if previous_message and (
            message["user_id"] == previous_message["user_id"]
            and message["timestamp"] - previous_message["timestamp"] <= time_threshold
        ):
            current_thread.append(message)
        else:
            if current_thread:
                threads.append(current_thread)
            current_thread = [message]
        previous_message = message

    if current_thread:
        threads.append(current_thread)
    
    return threads

def flatten_metadata(metadata):
    """
    Convert complex metadata to a format that is compatible with MongoDB.
    """
    flattened_metadata = {}
    for key, value in metadata.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flattened_metadata[f"{key}_{sub_key}"] = ', '.join(map(str, sub_value)) if isinstance(sub_value, list) else str(sub_value)
        elif isinstance(value, list):
            flattened_metadata[key] = ', '.join(map(str, value))
        else:
            flattened_metadata[key] = str(value)
    return flattened_metadata

async def process_threads_and_enhance_metadata(threads):
    """
    Process each thread to combine messages and enhance metadata.
    """
    enhanced_documents = []

    for thread in threads:
        combined_content = " ".join([msg["text"] for msg in thread])
        combined_metadata = {
            "source": "slack_thread",
            "user_ids": [msg["user_id"] for msg in thread],
            "user_names": [msg["user_name"] for msg in thread],
            "timestamps": [msg["timestamp"].strftime("%Y-%m-%d %H:%M:%S") for msg in thread],
            "channel_id": thread[0]["channel_id"],
            "channel_name": thread[0]["channel_name"],
        }

        doc = Document(page_content=combined_content, metadata=combined_metadata)
        enhanced_doc = await enhance_metadata_with_bertopic_and_advanced_ner(doc)
        enhanced_documents.append(enhanced_doc)

    return enhanced_documents

async def fetch_and_process_slack_messages():
    """
    Fetch Slack messages, group them into threads, and enhance metadata.
    """
    messages = await fetch_messages_from_slack()
    threads = group_messages_into_threads(messages)
    enhanced_documents = await process_threads_and_enhance_metadata(threads)
    
    slack_embeddings = await embed_documents_in_batches(enhanced_documents, sentence_model, batch_size=10, use_sentence_transformer=True)
    slack_re_embeddings = await embed_documents_in_batches(
        [Document(page_content=str(embedding)) for embedding in slack_embeddings], 
        OpenAIEmbeddings(model="text-embedding-ada-002"), 
        batch_size=10
    )

    return enhanced_documents, slack_re_embeddings
