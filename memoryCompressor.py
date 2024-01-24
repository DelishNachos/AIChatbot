import requests
import json

systemContext: str = "Summarize"

# Headers
headers = {
    "Content-Type": "application/json",
}

def get_api_response(prompt: str):
    text: str | None = None
    try:
        response: dict = requests.post(URL, headers=headers, data=json.dumps(create_prompt_format(prompt)))
        choices: dict = response.json()['choices'][0]
        text = choices['message']['content']

    except Exception as e:
        #print('Error:', e)
        print("error")

    return text

def create_prompt_format(prompt: str):
    payload = {
    "messages": [
        {"role": "system", "content": systemContext},
        {"role": "user", "content": prompt},
    ],
    "model": "openchat-3.5-7b",
    "stream": False,
    "max_tokens": 4096,
    "stop": ['Human:', 'AI:'],
    "frequency_penalty": .2,
    "presence_penalty": .1,
    "temperature": .7,
    "top_p": 0.95
    }
    return payload

