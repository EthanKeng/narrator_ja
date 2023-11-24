import os
from openai import OpenAI
import base64
import time
import errno
from elevenlabs import generate, play, set_api_key, voices
from threading import Thread, Semaphore


client = OpenAI()

set_api_key(os.environ.get("ELEVENLABS_API_KEY"))
# Semaphore to limit the number of concurrent threads
thread_semaphore = Semaphore(2)

def encode_image(image_path):
    while True:
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except IOError as e:
            if e.errno != errno.EACCES:
                # Not a "file in use" error, re-raise
                raise
            # File is being written to, wait a bit and retry
            time.sleep(0.1)


def play_audio(text):
    audio_stream = generate(text, model="eleven_multilingual_v2",voice=os.environ.get("ELEVENLABS_VOICE_ID"), stream=True, latency= 3)
    chunks_to_collect = 50
    collected_chunks = []

    for i, chunk in enumerate(audio_stream):
        collected_chunks.append(chunk)

        if (i + 1) % chunks_to_collect == 0:
            # Concatenate the collected chunks
            concatenated_chunks = b''.join(collected_chunks)

            # Play the concatenated chunks
            play(concatenated_chunks)

            # Reset the collected chunks
            collected_chunks = []

    # Play any remaining chunks if the total number of chunks is not a multiple of 5
    if collected_chunks:
        concatenated_chunks = b''.join(collected_chunks)
        play(concatenated_chunks)


def generate_new_line(base64_image):
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in Japanese"},
                {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{base64_image}",
                },
            ],
        },
    ]


def analyze_image(base64_image, script):
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "system",
                "content": """
                You are Sir David Attenborough. Narrate the picture of the human as if it is a nature documentary.
                Make it snarky and funny. Don't repeat yourself. Make it short. If I do anything remotely interesting, make a big deal about it! 
                """,
            },
        ]
        + script
        + generate_new_line(base64_image),
        max_tokens=500,
    )
    response_text = response.choices[0].message.content
    return response_text


def getImgAnalyze(script):
    # path to your image
    image_path = os.path.join(os.getcwd(), "./frames/frame.jpg")

    # getting the base64 encoding
    base64_image = encode_image(image_path)

    # analyze posture
    print("👀 David is watching...")
    analysis = analyze_image(base64_image, script=script)

    print("🎙️ David says:")
    print(analysis)
    return analysis

def main():
    script = []
    analysis = getImgAnalyze(script)

    while True:

        thread = Thread(target=play_audio, args=(analysis,))
        thread.start()
        time.sleep(5)

        analysis = getImgAnalyze(script)
        thread.join()

        # play_audio(analysis)

        script = script + [{"role": "assistant", "content": analysis}]

        # wait for 5 seconds
        time.sleep(0.1)


if __name__ == "__main__":
    main()
