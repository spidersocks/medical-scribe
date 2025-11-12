import asyncio
import json
import logging
import os
from typing import Any, Dict, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult
from starlette.websockets import WebSocketState

logger = logging.getLogger(__name__)
router = APIRouter()

# Enable debug logging for DashScope
logging.getLogger('dashscope').setLevel(logging.DEBUG)

class ParaformerCallback(RecognitionCallback):
    def __init__(self, loop, queue):
        self.loop = loop
        self.queue = queue

    def on_open(self):
        logger.info("[DashScope] Connection opened")

    def on_error(self, message: str):
        logger.error("[DashScope] Error: %s", message)
        self.loop.call_soon_threadsafe(
            self.queue.put_nowait,
            {"_error": True, "message": message}
        )

    def on_close(self):
        logger.info("[DashScope] Connection closed")

    def on_complete(self):
        logger.info("[DashScope] Recognition completed")
        self.loop.call_soon_threadsafe(
            self.queue.put_nowait,
            {"_complete": True}
        )

    def on_event(self, message: Any):
        try:
            if isinstance(message, RecognitionResult):
                sentence = message.get_sentence()
                if isinstance(sentence, dict) and 'text' in sentence:
                    is_final = message.is_sentence_end(sentence)
                    logger.debug(
                        "[DashScope] Received %s: %s",
                        "final" if is_final else "partial",
                        sentence['text']
                    )
                    
                    # Transform to AWS-like format
                    transformed = {
                        "Transcript": {
                            "Results": [{
                                "ResultId": f"alibaba-{message.get_request_id()}",
                                "IsPartial": not is_final,
                                "Alternatives": [{
                                    "Transcript": sentence['text'],
                                    "Items": []
                                }],
                                "LanguageCode": "en"  # Will need to extract from message
                            }]
                        },
                        "DisplayText": sentence['text'],
                        "TranslatedText": None,
                        "ComprehendEntities": []
                    }
                    
                    self.loop.call_soon_threadsafe(
                        self.queue.put_nowait,
                        transformed
                    )
        except Exception as e:
            logger.error("Error processing DashScope event: %s", e)

@router.websocket("/alibaba")
async def transcribe_alibaba(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established")

    # Initialize DashScope recognition
    loop = asyncio.get_running_loop()
    queue = asyncio.Queue()
    callback = ParaformerCallback(loop, queue)
    
    try:
        recognizer = Recognition(
            model="paraformer-realtime-v2",
            format="pcm",
            sample_rate=16000,
            diarization_enabled=True,
            language_hints=['en', 'yue', 'zh'],
            callback=callback,
            semantic_punctuation_enabled=False,  # Important for real-time
            punctuation_prediction_enabled=True,
            inverse_text_normalization_enabled=True
        )
        
        recognizer.start()
        logger.info("DashScope recognition started")

        # Start forwarding events to WebSocket
        async def forward_events():
            while True:
                msg = await queue.get()
                if isinstance(msg, dict):
                    if msg.get("_error"):
                        await websocket.close(code=1011, reason=msg["message"])
                        break
                    if msg.get("_complete"):
                        await websocket.close()
                        break
                    if websocket.client_state == WebSocketState.CONNECTED:
                        await websocket.send_text(json.dumps(msg))

        forward_task = asyncio.create_task(forward_events())

        # Process incoming audio
        while True:
            try:
                chunk = await websocket.receive_bytes()
                if not chunk:
                    logger.info("Received empty chunk - sending end flag")
                    recognizer.send_end_flag()
                    break
                recognizer.send_audio_frame(chunk)
            except WebSocketDisconnect:
                logger.info("Client disconnected")
                break

        await forward_task
    except Exception as e:
        logger.error("Transcription error: %s", e)
        await websocket.close(code=1011, reason=str(e))
    finally:
        try:
            recognizer.stop()
        except:
            pass
        logger.info("Transcription session ended")