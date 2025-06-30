# # src/utils/llm_interface.py


# import os
# import json
# import time
# from typing import Dict, Any, Optional
# from anthropic import Anthropic, BadRequestError, APIError

# def call_llm(
#     prompt: str, 
#     system_message: str,
#     results_dir,
#     data_entry,
#     cached_message: Optional[str] = None,
#     retry_time: int = 5,
# ) -> Dict[str, Any]:
#     """
#     Sends a prompt to the LLM using Anthropic's direct API with message caching.

#     Args:
#         prompt (str): The prompt to send to the LLM.
#         system_message (str): The system message to set the context for the LLM.
#         cached_message (Optional[str]): Previous message content to cache.
#         retry_time (int): Time to wait before retrying in case of failure.

#     Returns:
#         Dict[str, Any]: The JSON response from the LLM.
#     """
#     client = Anthropic()
#     max_retries = 6

#     messages = [{"role": "user", "content": prompt}]

#     system_messages = [{
#                     "type": "text",
#                     "text": system_message
#                 }]

#     if cached_message:
#         system_messages.insert(1, {
#             "type": "text",
#             "text": cached_message,
#             "cache_control": {"type": "ephemeral"}
#         })

#     for attempt in range(1, max_retries + 1):
#         try:
#             response = client.beta.prompt_caching.messages.create(
#                 model="claude-3-5-sonnet-20240620",
#                 max_tokens=8024,
#                 system=system_messages,
#                 messages=messages
#             )

#             content = response.content
#             if not content:
#                 raise ValueError("No content found in the LLM response.")

#             json_response = json.loads(content[0].text)
#             return json_response

#         except (APIError, ValueError, json.JSONDecodeError, BadRequestError) as e:
#             print(f"Attempt {attempt} - Error during LLM call: {e}")
#             if attempt < max_retries:
#                 print(f"Retrying in {retry_time} seconds...")
#                 time.sleep(retry_time)
#             else:
#                 # Save error only after max retries exceeded
#                 error_data = {
#                     "text": data_entry["doc_text"],
#                     "location": data_entry["doc_location"], 
#                     "file": data_entry["document_file"],
#                     "error": str(e)
#                 }
           
#                 error_file = os.path.join(results_dir, "errors.json")
#                 try:
#                     with open(error_file, 'r') as f:
#                         errors = json.load(f)
#                 except FileNotFoundError:
#                     errors = {"errors": []}
                    
#                 errors["errors"].append(error_data)
                
#                 with open(error_file, 'w') as f:
#                     json.dump(errors, f, indent=2)

#                 print("Max retries reached. Raising exception.")
#                 raise Exception("Failed to get a valid response from LLM after multiple attempts.") from e

import os
import json
import time
from typing import Dict, Any, Optional
from anthropic import Anthropic, BadRequestError, APIError
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file
# openai_key = os.getenv('OPENAI_API_KEY')

def call_llm(
    prompt: str, 
    system_message: str,
    results_dir: str,
    data_entry: Dict,
    cached_message: Optional[str] = None,
    retry_time: int = 15,
) -> Dict[str, Any]:
    """
    Sends a prompt to the LLM (either Anthropic or OpenAI) with message caching support.

    Args:
        prompt (str): The prompt to send to the LLM.
        system_message (str): The system message to set the context for the LLM.
        results_dir (str): Directory to save error logs.
        data_entry (Dict): Entry containing document info for error logging.
        cached_message (Optional[str]): Previous message content to cache (used for Anthropic).
        retry_time (int): Time to wait before retrying in case of failure.

    Returns:
        Dict[str, Any]: The JSON response from the LLM.
    """
    max_retries = 50
    use_openai = os.getenv('USE_OPENAI', 'false').lower() == 'true'

    def save_error(error: str):
        error_data = {
            "text": data_entry["doc_text"],
            "location": data_entry["doc_location"], 
            "file": data_entry["document_file"],
            "error": error
        }
        
        error_file = os.path.join(results_dir, "errors.json")
        try:
            with open(error_file, 'r') as f:
                errors = json.load(f)
        except FileNotFoundError:
            errors = {"errors": []}
        
        errors["errors"].append(error_data)
        
        with open(error_file, 'w') as f:
            json.dump(errors, f, indent=2)

    if use_openai:
        print("USING OPENAI")
        client = OpenAI()
        print(client)
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": cached_message + "\n" + prompt}
        ]

        for attempt in range(1, max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model="o3-mini-2025-01-31",
                    reasoning_effort="high",
                    messages=messages,
                    response_format={"type": "json_object"}
                )
                
                content = response.choices[0].message.content
                if not content:
                    raise ValueError("No content found in the LLM response.")

                return json.loads(content)

            except (OpenAIError, ValueError, json.JSONDecodeError) as e:
                print(f"Attempt {attempt} - Error during OpenAI LLM call: {e}")
                if attempt < max_retries:
                    print(f"Retrying in {retry_time} seconds...")
                    time.sleep(retry_time)
                else:
                    save_error(str(e))
                    print("Max retries reached. Raising exception.")
                    raise Exception("Failed to get a valid response from OpenAI LLM after multiple attempts.") from e

    else:
        client = Anthropic()
        messages = [{"role": "user", "content": prompt}]
        system_messages = [{"type": "text", "text": system_message}]

        if cached_message:
            system_messages.insert(1, {
                "type": "text",
                "text": cached_message,
                "cache_control": {"type": "ephemeral"}
            })

        for attempt in range(1, max_retries + 1):
            try:
                response = client.beta.prompt_caching.messages.create(
                    model="claude-3-5-sonnet-20240620",
                    max_tokens=8024,
                    system=system_messages,
                    messages=messages
                )

                content = response.content
                if not content:
                    raise ValueError("No content found in the LLM response.")

                return json.loads(content[0].text)

            except (APIError, ValueError, json.JSONDecodeError, BadRequestError) as e:
                print(f"Attempt {attempt} - Error during Anthropic LLM call: {e}")
                if attempt < max_retries:
                    print(f"Retrying in {retry_time} seconds...")
                    time.sleep(retry_time)
                else:
                    save_error(str(e))
                    print("Max retries reached. Raising exception.")
                    raise Exception("Failed to get a valid response from Anthropic LLM after multiple attempts.") from e