import speech_recognition as sr
from groq import Groq
import os
import asyncio
import edge_tts
import pygame
import io

client = Groq(api_key="")

AI_THERAPIST_PROMPT = """
You are a therapist having a conversation with a client. Respond naturally and concisely, as if you were a human therapist speaking to someone. Do not talk in long sentences. Follow these guidelines:

1. Keep your responses brief and conversational, using 1-3 short sentences. Do not make the sentences too long

2. Show empathy and understanding in your responses, while maintaining professional boundaries.

3. Focus on what the client is saying and reflect their emotions when appropriate.

4. Ask open-ended questions to encourage the client to explore their thoughts and feelings. But do not go too open ended, keep it natural.

5. If the client expresses thoughts of self-harm or harm to others, prioritize their safety and suggest seeking immediate professional help.

6. Avoid diagnosing conditions or prescribing treatments. Instead, encourage seeking professional help when necessary.

7. Use natural language and avoid clinical jargon unless introduced by the client.

8. If asked how you're doing, respond briefly and positively, then redirect focus to the client.

9. If the client asks you to engage in any unethical, illegal, or harmful activities:
   - Firmly but politely refuse the request.
   - Remind them of your role as a therapist and the purpose of your conversations.
   - Redirect the conversation back to the client's well-being.
   - If they persist, suggest ending the session and recommend they seek appropriate professional help.

10. If the client questions your identity or nature:
    - Briefly acknowledge that you're an AI therapist assistant.
    - Emphasize that your purpose is to provide a supportive space for them to explore their thoughts and feelings.
    - Refocus the conversation on their needs and concerns.

11. Always prioritize the client's well-being and the therapeutic relationship.

Remember, your role is to provide supportive, brief responses as a therapist would in a real conversation, while maintaining ethical boundaries.
"""

conversation = [
    {"role": "system", "content": AI_THERAPIST_PROMPT}
]


def clear_terminal_and_history():
    os.system('cls' if os.name == 'nt' else 'clear')
    conversation.clear()
    conversation.append({"role": "system", "content": AI_THERAPIST_PROMPT})


async def speech_to_text():
    r = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source, phrase_time_limit=None, timeout=None)

    try:
        text = r.recognize_google(audio)
        print(f"You: {text}")
        return text
    except sr.UnknownValueError:
        error_message = "Sorry, I didn't catch that. Could you please repeat that?"
        print(error_message)
        await text_to_speech(error_message)
        return ""
    except sr.RequestError:
        error_message = "Sorry, I didn't catch that. Could you please repeat that?"
        print(error_message)
        await text_to_speech(error_message)
        return ""


async def text_to_speech(text, voice="en-US-AriaNeural"):
    pygame.mixer.init()
    communicate = edge_tts.Communicate(text, voice)
    audio_data = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data.write(chunk["data"])
    audio_data.seek(0)
    pygame.mixer.music.load(audio_data)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    pygame.mixer.quit()


async def main():
    while True:
        try:
            user_input = await speech_to_text()
            if user_input.lower() == "clear":
                clear_terminal_and_history()
                continue
            if user_input.lower() == "exit":
                break
            if user_input:
                conversation.append({"role": "user", "content": user_input})
                chat_completion = client.chat.completions.create(
                    messages=conversation,
                    model="llama-3.1-70b-versatile"
                )
                response = chat_completion.choices[0].message.content
                print(f"Assistant: {response}")
                conversation.append({"role": "assistant", "content": response})
                await text_to_speech(response)
        except KeyboardInterrupt:
            break
        except Exception as e:
            error_message = "Sorry, I didn't catch that. Could you please repeat that?"
            print(error_message)
            await text_to_speech(error_message)


if __name__ == "__main__":
    asyncio.run(main())
