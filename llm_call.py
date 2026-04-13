from portkey_ai import Portkey

import os
from dotenv import load_dotenv

load_dotenv()
portkey_api_key = os.getenv("PORTKEY_API_KEY")

portkey = Portkey(
    api_key= portkey_api_key,
    provider="@topicosdeengenhariadesoftwarre/z-ai/glm-4.5-air:free"
)

chat_complete = portkey.chat.completions.create(
    model="z-ai/glm-4.5-air:free",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello chatgpt, how are you?"}
    ]
)
print(chat_complete.choices[0].message.content)