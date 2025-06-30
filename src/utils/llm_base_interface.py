import os
import json
import time
from typing import Dict, Any, Optional
from anthropic import Anthropic, BadRequestError, APIError

def call_base_llm(
    prompt: str, 
    system_message: str,
    cached_message: Optional[str] = None,
    max_retries: int = 50,
    retry_time: int = 5,
    max_tokens: int = 8024
) -> Dict[str, Any]:
    """
    Generic LLM interface using Anthropic's API with message caching.
    
    Args:
        prompt (str): The prompt to send to the LLM.
        system_message (str): The system message for context.
        cached_message (Optional[str]): Previous message to cache.
        max_retries (int): Maximum number of retry attempts.
        retry_time (int): Time to wait between retries in seconds.
        max_tokens (int): Maximum tokens in response.
    
    Returns:
        Dict[str, Any]: The JSON response from the LLM.
        
    Raises:
        Exception: If unable to get valid response after retries.
    """
    client = Anthropic()
    messages = [{"role": "user", "content": prompt}]

    system_messages = [{
        "type": "text",
        "text": system_message
    }]

    if cached_message:
        system_messages.insert(1, {
            "type": "text",
            "text": cached_message,
            "cache_control": {"type": "ephemeral"}
        })

    for attempt in range(1, max_retries + 1):
        try:
            response = client.beta.prompt_caching.messages.create(
                model="claude-3-5-sonnet-latest",
                max_tokens=max_tokens,
                system=system_messages,
                messages=messages
            )

            content = response.content
            if not content:
                raise ValueError("Empty response from LLM")

            return json.loads(content[0].text)

        except (APIError, ValueError, json.JSONDecodeError, BadRequestError) as e:
            print(f"Attempt {attempt} - Error: {str(e)}")
            
            if attempt < max_retries:
                print(f"Retrying in {retry_time} seconds...")
                time.sleep(retry_time)
                continue
            
            raise Exception(f"Failed to get valid response after {max_retries} attempts: {str(e)}")