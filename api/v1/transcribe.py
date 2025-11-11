import asyncio
import json
import logging
from typing import List

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from dashscope.audio.asr import Recognition

from config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/alibaba")
async def transcribe_alibaba(ws: WebSocket):
    """Proxy to Alibaba Paraformer real-time transcription service."""
    await ws.accept()

    # We will always instruct Alibaba to detect between our target languages,
    # ignoring any query parameters from the client.
    language_hints: List[str] = ['en', 'yue', 'zh']
    
    logger.info(
        "Browser connected to Alibaba endpoint. Auto-detecting languages: %s", 
        language_hints
    )

    if not settings.dashscope_api_key:
        logger.error("DASHSCOPE_API_KEY is not set.")
        await ws.close(code=1011, reason="Server not configured for this transcription service.")
        return

    try:
        class WsCallback:
            def __init__(self, websocket: WebSocket):
                self.websocket = websocket
                self.task_id = None

            async def on_open(self) -> None:
                logger.info("Alibaba Recognizer connection opened.")

            async def on_error(self, message: str) -> None:
                logger.error("Alibaba Recognizer error: %s", message)
                if self.websocket.client_state == WebSocketState.CONNECTED:
                    await self.websocket.close(code=1011, reason="Transcription service error.")

            async def on_close(self) -> None:
                logger.info("Alibaba Recognizer connection closed.")
                if self.websocket.client_state == WebSocketState.CONNECTED:
                     await self.websocket.close()

            async def on_event(self, message: str) -> None:
                try:
                    data = json.loads(message)
                    logger.debug("Received from Alibaba: %s", data)

                    header = data.get("header", {})
                    payload = data.get("payload", {})
                    
                    if not header or not payload or not payload.get("result"):
                        return

                    is_partial = header.get("name") == "TranscriptionResultChanged"
                    transcript_text = payload.get("result")
                    speaker_id = payload.get("speaker_id")

                    aws_compatible_payload = {
                        "Transcript": {
                            "Results": [
                                {
                                    "ResultId": f"alibaba-seg-{payload.get('sentence_id', 0)}",
                                    "IsPartial": is_partial,
                                    "Alternatives": [
                                        {
                                            "Transcript": transcript_text,
                                            "Items": [
                                                {
                                                    "Type": "pronunciation",
                                                    "Speaker": speaker_id
                                                }
                                            ] if speaker_id else []
                                        }
                                    ],
                                    "LanguageCode": payload.get("language")
                                }
                            ]
                        },
                        "DisplayText": transcript_text,
                        "TranslatedText": None,
                        "ComprehendEntities": []
                    }

                    if self.websocket.client_state == WebSocketState.CONNECTED:
                        await self.websocket.send_text(json.dumps(aws_compatible_payload))

                except Exception as e:
                    logger.error("Error processing or transforming message from Alibaba: %s", e)

        # Initialize the recognizer with our fixed language hints
        callback = WsCallback(ws)
        recognizer = Recognition(
            model='paraformer-realtime-v2',
            format='pcm',
            sample_rate=16000,
            callback=callback,
            diarization_enabled=True, 
            language_hints=language_hints, # <--- TYPO REMOVED HERE
        )

        recognizer.start()

        while True:
            try:
                audio_chunk = await ws.receive_bytes()
                if audio_chunk:
                    recognizer.send_audio_frame(audio_chunk)
            except WebSocketDisconnect:
                logger.info("Client disconnected. Closing Alibaba recognizer.")
                break
    
    except Exception as exc:
        logger.error("Alibaba WebSocket proxy failed: %s", exc)
        if ws.client_state == WebSocketState.CONNECTED:
            await ws.close(code=1011, reason="Internal server error.")
    finally:
        if 'recognizer' in locals() and recognizer.is_running():
            recognizer.stop()
        logger.info("Alibaba browser session ended.")