"""
Handles the extraction and transcription of content from YouTube videos.

This module provides a primary function, get_video_transcript, which first
attempts to fetch a pre-existing transcript for a given YouTube video URL.
If no transcript is available, it gracefully falls back to downloading the
video's audio, transcribing it using OpenAI's Whisper model, and then
cleaning up the audio file.
"""

import os
import logging
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound
from faster_whisper import WhisperModel
import yt_dlp

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s -%(filename)s - %(message)s')


def get_video_transcript(youtube_url: str) -> str | None:
    """
    Retrieves the transcript for a YouTube video.

    Tries to fetch the pre-generated transcript first. If that fails,
    it downloads the audio and transcribes it using Whisper.

    Args:
        youtube_url (str): The URL of the YouTube video.

    Returns:
        str | None: The full transcript as a single string, or None if an error occurs.
    """
    try:
        # Extract the video ID from the URL
        video_id = youtube_url.split("v=")[1].split("&")[0]
        logging.info(f"Attempting to fetch transcript for video ID: {video_id}")
        
        # Fetch transcript using the API
        ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.fetch(video_id)
        
        # Format the transcript list into a single string
        transcript = " ".join([d.text for d in transcript_list])
        logging.info(f"Successfully fetched pre-existing transcript for video ID: {video_id}")
        return transcript
        
    except NoTranscriptFound:
        logging.warning(f"No pre-existing transcript found for {youtube_url}. Falling back to audio transcription.")
        return _transcribe_audio_with_whisper(youtube_url)
    except Exception as e:
        logging.error(f"An unexpected error occurred while processing {youtube_url}: {e}")
        return None


def _transcribe_audio_with_whisper(youtube_url: str) -> str | None:
    """
    Downloads audio from a YouTube URL and transcribes it using Whisper.

    This is a fallback method used when no pre-existing transcript is available.
    
    Args:
        youtube_url (str): The URL of the YouTube video.

    Returns:
        str | None: The transcribed text, or None if an error occurs.
    """
    audio_file = "temp_audio.mp4"
    
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': audio_file,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }

    try:
        
        logging.info(f"Downloading audio for: {youtube_url}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        
        
        audio_file_mp3 = "temp_audio.mp3"
        if not os.path.exists(audio_file_mp3):
            raise FileNotFoundError("Audio file was not created after download.")

        
        logging.info("Loading Whisper model and starting transcription...")
        model = WhisperModel("base", device="cpu", compute_type="int8")
        segments, _ = model.transcribe(audio_file_mp3, beam_size=5)
        
        transcription = "".join(segment.text for segment in segments)
        logging.info("Transcription complete.")
        return transcription

    except Exception as e:
        logging.error(f"Failed during audio download or transcription: {e}")
        return None
    finally:
        
        audio_file_mp3 = "temp_audio.mp3"
        if os.path.exists(audio_file_mp3):
            os.remove(audio_file_mp3)
            logging.info(f"Cleaned up audio file: {audio_file_mp3}")