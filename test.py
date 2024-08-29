import openai
from openai import OpenAI

api_base = "https://api.nextapi.fun"
api_key = "ak-Lfu504S5OrzjNvivYYdY6E8xvn1hiTY42texx7WvTIojB9MC"

client = OpenAI(base_url=api_base,api_key=api_key)


response = client.chat.completions.create(
                model='gpt-4o-mini',
                temperature=0.0,
                messages=[{
                    'role': 'user',
                    'content': 'ping!',
                }],
            )
content = response.choices[0].message.content
print(content)