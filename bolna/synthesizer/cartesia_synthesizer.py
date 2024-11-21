import asyncio
import copy
import os
# import json
import traceback
# import base64

from cartesia import AsyncCartesia

from bolna.memory.cache.inmemory_scalar_cache import InmemoryScalarCache
from .base_synthesizer import BaseSynthesizer
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import convert_audio_to_wav, create_ws_data_packet, resample

logger = configure_logger(__name__)

class CartesiaSynthesizer(BaseSynthesizer):
    def __init__(self, voice_id, voice, model="sonic-english", audio_format="mp3", sampling_rate="16000",
                 stream=False, buffer_size=400, synthesizer_key=None, caching=True, **kwargs):
        """
        Initialize Cartesia Synthesizer
        
        Args:
            voice_id (str): Unique identifier for the voice
            voice (str): Voice name or description
            model (str, optional): TTS model to use. Defaults to "sonic-english".
            audio_format (str, optional): Output audio format. Defaults to "mp3".
            sampling_rate (str, optional): Sampling rate. Defaults to "16000".
            stream (bool, optional): Enable streaming mode. Defaults to False.
            buffer_size (int, optional): Buffer size for streaming. Defaults to 400.
            synthesizer_key (str, optional): Cartesia API key. Defaults to None.
            caching (bool, optional): Enable caching of synthesized audio. Defaults to True.
        """
        super().__init__(stream)
        
        # API Key Management
        self.api_key = os.environ.get("CARTESIA_API_KEY", synthesizer_key)
        if not self.api_key:
            raise ValueError("Cartesia API key is required. Set CARTESIA_API_KEY env var or pass synthesizer_key.")
        
        # Cartesia Client Initialization
        self.client = AsyncCartesia(api_key=self.api_key)
        
        # Voice and Model Configuration
        self.voice_id = voice_id
        self.model = model
        
        # Streaming Configuration
        self.stream = True
        self.sampling_rate = sampling_rate
        self.use_mulaw = True
        
        # State Management
        self.first_chunk_generated = False
        self.last_text_sent = False
        
        # Caching
        self.caching = caching
        self.cache = InmemoryScalarCache() if caching else None
        
        # Tracking
        self.synthesized_characters = 0
        self.context_id = None
        
        # WebSocket Components
        self.websocket = None
        self.context = None
        
        # Output Format Configuration
        self.output_format = {
            "container": "raw",
            "encoding": "pcm_mulaw",
            "sample_rate": 8000
        }

    def get_engine(self):
        """Return the current TTS model"""
        return self.model

    async def establish_connection(self):
        """Establish WebSocket connection with Cartesia"""
        while self.context is None:
            logger.info("Waiting for WebSocket context to be established...")
            await asyncio.sleep(1)
            try:
                self.websocket = await self.client.tts.websocket()
                self.context = self.websocket.context()
                logger.info("Established Cartesia WebSocket connection")
                return True
            except Exception as e:
                logger.error(f"Failed to establish connection: {e}")
                return None

    async def monitor_connection(self):
        while True:
            if self.websocket is None:
                logger.info("Re-establishing connection...")
                await self.establish_connection()
            await asyncio.sleep(50)

    async def sender(self, text, end_of_llm_stream=False):
        """
        Send text to Cartesia for synthesis
        
        Args:
            text (str): Text to synthesize
            end_of_llm_stream (bool): Flag to indicate end of input stream
        """
        try:
            # Ensure connection is established
            while self.context is None:
                logger.info("Waiting for WebSocket context to be established...")
                await asyncio.sleep(1)

            # Send text if not empty
            if text:
                logger.info(f"Sending text chunk: {text}")
                await self.context.send(
                    model_id=self.model,
                    transcript=text,
                    voice_id=self.voice_id,
                    continue_=True,
                    output_format=self.output_format
                )

            # If end of stream, signal no more inputs
            if end_of_llm_stream:
                self.last_text_sent = True
                await self.context.no_more_inputs()

        except Exception as e:
            logger.error(f"Error in sender: {e}")
            traceback.print_exc()

    async def receiver(self):
        """Receive audio chunks from Cartesia WebSocket"""
        try:
            async for output in self.context.receive():
                if 'audio' in output:
                    chunk = output['audio']
                    yield chunk
                else:
                    logger.info("No audio data in the response")
        except Exception as e:
            logger.error(f"Error in receiver: {e}")
            traceback.print_exc()

    async def generate(self):
        """
        Generate audio from text, supporting streaming
        
        Yields:
            Audio packets with metadata
        """
        try:
            if self.stream:
                async for message in self.receiver():
                    logger.info("Received message from server")
                    
                    meta_info = self.meta_info if hasattr(self, 'meta_info') else {}
                    meta_info['format'] = 'mulaw'
                    audio = message

                    yield create_ws_data_packet(audio, meta_info)
                    
                    if not self.first_chunk_generated:
                        meta_info["is_first_chunk"] = True
                        self.first_chunk_generated = True

                    if self.last_text_sent:
                        self.first_chunk_generated = False
                        self.last_text_sent = False
                        meta_info["end_of_synthesizer_stream"] = True
        except Exception as e:
            logger.error(f"Error in Cartesia generate: {e}")
            traceback.print_exc()


    async def push(self, message):
        """
        Push a message to the synthesizer queue
        
        Args:
            message (dict): Message containing text and metadata
        """
        logger.info(f"Pushed message to internal queue {message}")
        if self.stream:
            # Establish connection if not already done
            if self.context is None:
                await self.establish_connection()

            meta_info, text = message.get("meta_info"), message.get("data")
            self.synthesized_characters += len(text) if text is not None else 0
            
            end_of_llm_stream = "end_of_llm_stream" in meta_info and meta_info["end_of_llm_stream"]
            self.meta_info = copy.deepcopy(meta_info)
            meta_info["text"] = text
            
            # Use context ID from meta info
            self.context_id = meta_info["request_id"]
            
            # Create sender task
            self.sender_task = asyncio.create_task(self.sender(text, end_of_llm_stream))


    # def get_synthesized_characters(self):
    #     """Get total number of synthesized characters"""
    #     return self.synthesized_characters

    # async def close(self):
    #     """Close the Cartesia WebSocket connection"""
    #     try:
    #         if self.websocket:
    #             await self.websocket.close()
    #         if self.client:
    #             await self.client.close()
    #     except Exception as e:
    #         logger.error(f"Error closing Cartesia connection: {e}")

