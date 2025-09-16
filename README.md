# Medical Scribe Backend

FastAPI backend for real-time medical transcription using Amazon Transcribe Medical.

## Features
- Real-time audio transcription
- WebSocket connections
- Clinical note generation
- AWS Transcribe Medical integration

## Environment Variables
- `AWS_ACCESS_KEY_ID`: AWS access key
- `AWS_SECRET_ACCESS_KEY`: AWS secret key  
- `AWS_REGION`: AWS region (default: us-east-1)
- `ALLOWED_ORIGINS`: Comma-separated list of allowed CORS origins
- `PORT`: Server port (default: 8000)

## Local Development
```bash
pip install -r requirements.txt
python main.py