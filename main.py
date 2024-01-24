from typing import Final
import os
from dotenv import load_dotenv
from discord import Intents,Client,Message,FFmpegPCMAudio
from responses import get_responses
#import nacl
import asyncio
import chatbot
import pyttsx3

load_dotenv()
TOKEN: Final[str] = os.getenv('DISCORD_TOKEN')

intents: Intents = Intents.default()
intents.message_content = True
client: Client = Client(intents = intents)

voice = None
engine = None
fullPath = ''

async def send_message(message: Message, user_message: str):
    if message.channel.name != "aibotchannel":
        return
   
    if not user_message:
        print('Message was empty')
        return

    try:
        responses: list[str] = get_responses(user_message)
        for res in responses:
            await message.reply(res)

        if voice == None:
            return
                   
        #engine.save_to_file(responses[0], fullPath)
        #engine.runAndWait()
        
    except Exception as e:
        print(e)

@client.event
async def on_ready():
    print(f'{client.user} is now running')


@client.event
async def on_message(message: Message):
    if message.author == client.user:
        return
    
    if message.content.lower() == 'join':
        await join(message)
        return

    if message.content.lower() == 'leave':
        await leave(message)
        return

    username: str = str(message.author)
    user_message: str = message.content
    channel: str = str(message.channel)

    print(f'[{channel}] {username}: "{user_message}"')
    await send_message(message, user_message)

async def join(message: Message):
    if (message.author.voice):
        channel = message.author.voice.channel
        voice = await channel.connect()
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[1].id)
        fullPath = os.path.join(os.getcwd(), "test.wav")
        voice.play(FFmpegPCMAudio(source=fullPath))
    else:
        await message.channel.send("You are not in a voice channel, you must be in a voice channel to run this command!")

# async def leave(message: Message):
#     if message.author.voice:
#         await message.author.voice.channel.disconnect()
#     else:
#         await ctx.send("You must be in the Voice Channel or I'm not in a Voice Channel")

def main():
    client.run(token=TOKEN)

if __name__ == '__main__':
    main()
