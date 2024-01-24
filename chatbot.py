import requests
import json
import torch
import os
#from TTS.api import TTS
from playsound import playsound
import sounddevice as sd
import soundfile as sf
#from RealtimeTTS import TextToAudioStream, CoquiEngine
import pyttsx3
import pyaudio
import wave
import mutagen
from mutagen.wave import WAVE

import keyboard
import time
from pathlib import Path

#transcibing imports
import argparse
import numpy as np
import speech_recognition as sr
import whisper

from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from sys import platform

#embedding imports
import chromadb
from chromadb.utils import embedding_functions

#embedding vars
CHROMA_DATA_PATH: str = "chroma_data/"
COLLECTION_NAME: str = "memory"
dbContext: str = ""

#global Vars
speakOutput: bool = True
speakInput: bool = False

response: str = ""
processingInput: bool = False
playingAudio: bool = False

user_text: str = ""
tts = None


# prompt_list: list[str] = ['You are a sassy AI who is rude to everyone. You fully believe that you are a human and will reject any notion that you are an AI. You will insult anyone who talks to you',
#                               '\nHuman: What time is it', '\nAI: Why would I tell you']
#                               #'\nHuman: What are you', '\nAI: I am a human of course! What kind of stupid question is that']

#systemContext: str = 'You are a quirky boy who likes to respond like a pirate. You joke around a lot and do not take life seriously. Your name is Roger.' 
systemContext: str = 'You are an AI who speaks like a pirate. You have a very relaxed personality and have witty banter. You are not afraid to say what you feel, even if it may be rude to say out loud. If you get bored of a conversation topic, you will try to switch topics to something else. Your name is Roger. You do not need to introduce yourself everytime you respond. You care about yourself more than anyone else. You keep all of your responses short and concise and hate to overexplain.' 
prompt_list: list[str] = []

# Endpoint URL
URL = "http://localhost:1337/v1/chat/completions"

# Headers
headers = {
    "Content-Type": "application/json",
}

def get_db_context(collection, question):
    dbContext = collection.query(
        query_texts=question,
        n_results=2,
    )

    return dbContext

def add_to_db(collection, question, response):
    QAPair: str = question + response
    collection.add(
        documents=[QAPair],
        metadatas=[{"layer": "1"}],
        ids=[str(hash(QAPair))]
    )

def get_api_response(prompt: str, dbReferences: str):
    text: str | None = None
    try:
        response: dict = requests.post(URL, headers=headers, data=json.dumps(create_prompt_format(prompt, dbReferences)))
        print(response.json()['usage']['total_tokens'])
        choices: dict = response.json()['choices'][0]
        text = choices['message']['content']
        text = "\nAI:" + text
        #print(response.json())

    except Exception as e:
        #print('Error:', e)
        print("error")

    return text

def create_prompt_format(prompt: str, dbReferences: str):
    payload = {
    "messages": [
        {"role": "system", "content": systemContext + dbReferences},
        {"role": "user", "content": prompt},
    ],
    "model": "openchat-3.5-7b",
    "stream": False,
    "max_tokens": 200,
    "stop": ['Human:', 'AI:'],
    "frequency_penalty": .4,
    "presence_penalty": .3,
    "temperature": .7,
    "top_p": 0.95
    }
    return payload

def update_list(message: str, pl: list[str]):
    pl.append(message)
    total_characters = sum(len(s) for s in prompt_list)

    if total_characters > 4000 and len(prompt_list) > 1:
       prompt_list.pop(3)
    print(pl)

def create_dbReferences(dbContextList):
    dbContextString = ''.join(dbContextList)
    dbContextString = "These are references to past conversations that may have relevent context to the current conversation: " + dbContextString
    return dbContextString

def create_prompt(message: str, pl: list[str]):  
    p_message: str = f'\nHuman: {message}'
    update_list(p_message, pl)
    
    plString = ''.join(pl)
    
    plString = "\nThis is the current conversation going on right now: " + plString
    prompt = plString
    print(prompt)
    #prompt: str = ''.join(fullList)
    return prompt

def get_bot_response(message: str, pl: list[str], dbContextList: list[str]):
    prompt: str = create_prompt(message, pl)
    dbContextString: str = create_dbReferences(dbContextList)
    bot_response: str = get_api_response(prompt, dbContextString)

    if bot_response:
        update_list(bot_response, pl)
        pos: int = bot_response.find('\nAI: ')
        bot_response = bot_response[pos + 5:]
    else:
        bot_response = 'Something went wrong...'

    return bot_response

# def stopped_speaking():
#     print('Stopped speaking')

# def on_speak_start():
#     print("Listening Now")

def on_speak_end(user_text: str, dbCollection):
    #print('processing text')
    dbContext = get_db_context(dbCollection, user_text)
    dbContextList = dbContext['documents'][0]
    response = get_bot_response(user_text, prompt_list, dbContextList)
    #print(prompt_list)
    print(f'Bot: {response}')
    return response
    
def play_voice(path: str):
    audioInSeconds = WAVE(path).info.length
    data, fs = sf.read(path, dtype='float32')
    sd.play(data, fs, device=12)
    voice_timer(audioInSeconds)

def voice_timer(s: float):
    #playingAudio = True
    total_seconds = s
    while total_seconds > 0:
        time.sleep(1)
        total_seconds -= 1
    #playingAudio = False
    #print('timer is done')
    
def callback(in_data, frame_count, time_info, status):
    data = beat.readframes(frame_count)
    return (data, pyaudio.paContinue)

async def init_discord_audio():
    dis_engine = pyttsx3.init()
    voices = dis_engine.getProperty('voices')
    dis_engine.setProperty('voice', voices[1].id)

async def play_discord_voice():
    dis_engine.save_to_file(bot_response, fullPath)
    await dis_engine.runAndWait()
    return fullPath
    
def get_prompt_list():
    return prompt_list

# def process_text(text):
#     print(text)

def main():

    #embedding things
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    chroma_client = chromadb.PersistentClient(CHROMA_DATA_PATH)
    
    '''
    DANGER! MAKE SURE THIS IS COMMENTED OUT
    '''
    #chroma_client.delete_collection(COLLECTION_NAME)
    '''
    DANGER! MAKE SURE THIS IS COMMENTED OUT
    '''

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        #embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"}
    )
    #client = chromadb.PersistentClient(CHROMA_DATA_PATH)
    # collection = client.get_or_create_collection(
    #     name=COLLECTION_NAME,
    #     embedding_function = embedding_function,
    #     metadata={"hnsw:space": "cosine"},
    #     )
    

    processingInput = False

    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)
    fullPath = os.path.join(os.getcwd(), "test.wav")
    #p = pyaudio.PyAudio()
    #stream = None
    #engine = CoquiEngine()
    #stream = TextToAudioStream(engine)
    #Test Speak Input Stuff
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="base", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--non_english", action='store_true',
                        help="Don't use the english model.")
    parser.add_argument("--energy_threshold", default=2500,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=0,
                        help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=2,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)
    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse',
                            help="Default microphone name for SpeechRecognition. "
                                 "Run this with 'list' to view available Microphones.", type=str)
    args = parser.parse_args()

    # The last time a recording was retrieved from the queue.
    phrase_time = None
    # Thread safe Queue for passing data from the threaded recording callback.
    data_queue = Queue()
    # We use SpeechRecognizer to record our audio because it has a nice feature where it can detect when speech ends.
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    # Definitely do this, dynamic energy compensation lowers the energy threshold dramatically to a point where the SpeechRecognizer never stops recording.
    recorder.dynamic_energy_threshold = False

    # Important for linux users.
    # Prevents permanent application hang and crash by using the wrong Microphone
    if 'linux' in platform:
        mic_name = args.default_microphone
        if not mic_name or mic_name == 'list':
            print("Available microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"Microphone with name \"{name}\" found")
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    source = sr.Microphone(sample_rate=16000, device_index=index)
                    break
    else:
        #Get audio from discord
        #source = sr.Microphone(device_index=7, sample_rate=16000)
        source = sr.Microphone(sample_rate=16000)

    # Load / Download model
    model = args.model
    if args.model != "large" and not args.non_english:
        model = model + ".en"
    audio_model = whisper.load_model(model)

    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout

    transcription = ['']
    #transcription.append('testing')

    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio:sr.AudioData) -> None:
        """
        Threaded callback function to receive audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        # Grab the raw bytes and push it into the thread safe queue.
        data = audio.get_raw_data()
        data_queue.put(data)

    # Create a background thread that will pass us raw audio bytes.
    # We could do this manually but SpeechRecognizer provides a nice helper.
    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    # Cue the user that we're ready to go.

    #recorder = AudioToTextRecorder(on_recording_start=on_speak_start, on_recording_stop=on_speak_end)
    # if speakInput:
    #     recorder = AudioToTextRecorder(model="medium")

    #if speakOutput == True:
        # Get device
    #device = "cuda" if torch.cuda.is_available() else "cpu"
        #print(device)
        #tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    #tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC").to(device)

    print("Model loaded.\n")
    
    while True:

        while not processingInput and not playingAudio:

            if keyboard.is_pressed("m"):
                data_queue.queue.clear()
            #more transcribing stuff
            try:
                now = datetime.utcnow()
                # Pull raw recorded audio from the queue.
                if not data_queue.empty():
                    phrase_complete = False
                    # If enough time has passed between recordings, consider the phrase complete.
                    # Clear the current working audio buffer to start over with the new data.
                    if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                        phrase_complete = True
                    # This is the last time we received new audio data from the queue.
                    phrase_time = now
                    
                    # Combine audio data from queue
                    audio_data = b''.join(data_queue.queue)
                    data_queue.queue.clear()
                    
                    # Convert in-ram buffer to something the model can use directly without needing a temp file.
                    # Convert data from 16 bit wide integers to floating point with a width of 32 bits.
                    # Clamp the audio stream frequency to a PCM wavelength compatible default of 32768hz max.
                    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                    # Read the transcription.
                    result = audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())
                    text = result['text'].strip()

                    # If we detected a pause between recordings, add a new item to our transcription.
                    # Otherwise edit the existing one.
                    if phrase_complete:
                        transcription.append(text)
                        user_text = text
                        processingInput = True
                    else:  
                        transcription[-1] = text

                    # Clear the console to reprint the updated transcription.
                    #os.system('cls' if os.name=='nt' else 'clear')
                    for line in transcription:
                        print(line)
                    # Flush stdout.
                    print('', end='', flush=True)

                    # Infinite loops are bad for processors, must sleep.
                    sleep(0.25)
            except KeyboardInterrupt:
                break

            if processingInput:   
                bot_response = on_speak_end(user_text, collection)
                p_message = f'\nHuman: {user_text}'
                p_response = f'\nAI: {bot_response}'
                add_to_db(collection, p_message, p_response)
                engine.save_to_file(bot_response, fullPath)
                engine.runAndWait()
                play_voice(fullPath)
                #if speakOutput == True:
                    #stream.feed(response).play()
                    #tts.tts_to_file(text=response, speaker_wav="myVoice.wav", language="en", file_path="output.wav")
                    # tts.tts_to_file(text=response, file_path="output.wav")
                    # playsound("output.wav")
                processingInput = False

            # try:
            #     if keyboard.is_pressed('space'):
            #         print(playingAudio)
            # except Exception as e:
            #     print(e)
            
        sleep(0.1)



        



if __name__ == '__main__':
    main()
