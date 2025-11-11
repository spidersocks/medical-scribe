import asyncio
import json
import logging

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

    language_code = ws.query_params.get("language_code", "en-US")
    logger.info("Browser connected to Alibaba endpoint. Selected language=%s", language_code)

    if not settings.dashscope_api_key:
        logger.error("DASHSCOPE_API_KEY is not set.")
        await ws.close(code=1011, reason="Server not configured for this transcription service.")
        return

    try:
        # The Dashscope SDK's real-time recognition uses a callback structure.
        # We'll create a callback class to forward results to our client WebSocket.
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
                # This is where we receive transcription results from Alibaba.
                # We need to transform this into the format our frontend expects.
                try:
                    data = json.loads(message)
                    logger.debug("Received from Alibaba: %s", data)

                    header = data.get("header", {})
                    payload = data.get("payload", {})
                    
                    if not header or not payload or not payload.get("result"):
                        return

                    # 1. Determine if the result is partial or final
                    is_partial = header.get("name") == "TranscriptionResultChanged"
                    
                    # 2. Extract the text and speaker
                    transcript_text = payload.get("result")
                    # Use speaker_id for final results, otherwise it will be null
                    speaker_id = payload.get("speaker_id")

                    # 3. Build the AWS-compatible structure
                    aws_compatible_payload = {
                        "Transcript": {
                            "Results": [
                                {
                                    # Use the sentence_id from Alibaba for a stable ID
                                    "ResultId": f"alibaba-seg-{payload.get('sentence_id', 0)}",
                                    "IsPartial": is_partial,
                                    "Alternatives": [
                                        {
                                            "Transcript": transcript_text,
                                            # Create a minimal Items array with the speaker info
                                            # The frontend uses this to determine the speaker
                                            "Items": [
                                                {
                                                    "Type": "pronunciation",
                                                    "Speaker": speaker_id
                                                }
                                            ] if speaker_id else []
                                        }
                                    ],
                                    # Paraformer sends language with each sentence on final results
                                    "LanguageCode": payload.get("language")
                                }
                            ]
                        },
                        # For now, we are not doing Comprehend or Translation.
                        # We can add these later.
                        "DisplayText": transcript_text,
                        "TranslatedText": None,
                        "ComprehendEntities": []
                    }

                    # 4. Send the transformed payload to the client
                    if self.websocket.client_state == WebSocketState.CONNECTED:
                        await self.websocket.send_text(json.dumps(aws_compatible_payload))

                except Exception as e:
                    logger.error("Error processing or transforming message from Alibaba: %s", e)

        # Initialize the recognizer
        callback = WsCallback(ws)
        recognizer = Recognition(
            model='paraformer-realtime-v2',
            format='pcm',
            sample_rate=16000,
            callback=callback,
            diarization_enabled=True, # Enable speaker diarization
            language_hints=['zh', 'en', 'yue'], # For Mandarin, English, Cantonese
        )

        # Start the recognition process (this opens the WebSocket to Alibaba)
        recognizer.start()

        # Forward audio from our client to the recognizer
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
