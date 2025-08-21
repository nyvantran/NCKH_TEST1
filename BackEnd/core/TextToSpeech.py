import asyncio
import subprocess
import os
from edge_tts import Communicate


class TextToSpeech:
    def __init__(self, voice: str, **kwargs):
        self.text = ""
        self.voice = voice
        self.player_command = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", "-"]
        self.player_process = subprocess.Popen(
            self.player_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.kwargs = kwargs

    def generateCommunicate(self, text: str = None):
        """Create a Communicate instance for text-to-speech."""
        if not text is None:
            self.text = text

        return Communicate(self.text, self.voice, **self.kwargs)

    async def play_audio(self, text: str = None) -> None:
        """Play the generated audio file using ffplay."""
        if text is not None:
            self.text = text
        communicate = self.generateCommunicate(self.text)
        try:
            # Stream audio data to ffplay
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    self.player_process.stdin.write(chunk["data"])
                    self.player_process.stdin.flush()
        except BrokenPipeError:
            print("ℹ️ Audio playback was interrupted.")
        except Exception as e:
            print(f"Error during audio playback: {e}")

    async def saveAudio(self, text: str = None, load: str = None):
        """Save the generated audio to a file."""
        if text is not None:
            self.text = text
        communicate = self.generateCommunicate(self.text)
        if load is not None:
            try:
                await communicate.save(load)
                print(f"Audio saved to {load}")
            except Exception as e:
                print(f"Error saving audio: {e}")
        else:
            print("No file path provided to save audio.")

    def play(self, text: str = None, load: str = None):
        """Start the audio playback."""
        if load is not None:
            if not os.path.exists(load):
                asyncio.run(self.saveAudio(text, load))

            with open(load, 'rb') as f:
                audio_data = f.read()

            try:
                self.player_process.stdin.write(audio_data)
                self.player_process.stdin.flush()
            except BrokenPipeError:
                print("ℹ️ Audio playback was interrupted.")
            except Exception as e:
                print(f"Error during audio playback: {e}")

        else:
            asyncio.run(self.play_audio(text))

    def stop(self):
        """Stop the audio playback."""
        if self.player_process.stdin:
            self.player_process.stdin.close()
        self.player_process.wait()
        print("✅ Audio playback stopped.")


# example usage
def main():
    text1 = "987654321"
    config = r"D:\Project\NCKH\NCKH_YOLOv5_social_distancing\BackEnd\speech"
    # text2 = "123456789"
    voice = "vi-VN-NamMinhNeural"
    texttospeech = TextToSpeech(voice=voice, rate="+50%", pitch="+50Hz")
    texttospeech.play(text1, load=os.path.join(config, "test1.mp3"))
    # texttospeech.play(text2)
    #
    texttospeech.stop()
    pass


if __name__ == "__main__":
    main()
