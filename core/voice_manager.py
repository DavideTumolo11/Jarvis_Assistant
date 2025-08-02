"""
JARVIS AI ASSISTANT - VOICE MANAGER DEFINITIVO (MASTER PLAN)
==========================================================

Questo file gestisce tutto il sistema vocale di Jarvis secondo il Master Plan:
- STT (Speech-to-Text) con OpenAI Whisper DIRETTO (no SpeechRecognition)
- TTS (Text-to-Speech) con Piper Neural TTS nativo
- Audio I/O con sistema nativo Windows/macOS/Linux (NO PyAudio)
- Wake word detection con fuzzy matching
- Eventi real-time per integrazione sistema

STACK TECNOLOGICO MASTER PLAN:
- Whisper: Accesso diretto alle API per STT
- Piper: TTS neurale completamente locale  
- Audio: Sistema nativo (Windows WASAPI, macOS Core Audio, Linux ALSA)
- Threading: AsyncIO per processing non-bloccante
"""

import asyncio
import logging
import threading
import time
import os
import tempfile
import wave
import json
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import difflib

# DIPENDENZE VOICE STACK (MASTER PLAN)
try:
    import whisper
    import numpy as np
    # Piper TTS import (gestito dinamicamente)
    import subprocess
    import platform
except ImportError as e:
    print(f"❌ Missing voice dependencies: {e}")
    print("Run: pip install openai-whisper numpy")
    exit(1)

@dataclass
class VoiceConfig:
    """Configurazione Voice Manager secondo Master Plan"""
    
    # STT Configuration (Whisper diretto)
    whisper_model: str = "base"
    whisper_language: str = "it"
    whisper_device: str = "cpu"
    
    # TTS Configuration (Piper nativo)
    tts_engine: str = "piper"
    piper_model: str = "it_IT-riccardo-x_low"
    piper_speed: float = 1.0
    
    # Audio Configuration (sistema nativo)
    sample_rate: int = 16000
    chunk_size: int = 1024
    audio_format: str = "wav"
    
    # Wake Words Configuration
    wake_words: List[str] = None
    wake_sensitivity: float = 0.8
    always_listening: bool = True
    
    # Performance Configuration
    max_stt_latency: float = 3.0
    max_tts_latency: float = 2.0
    voice_activity_detection: bool = True
    
    def __post_init__(self):
        if self.wake_words is None:
            self.wake_words = ["jarvis", "ehi jarvis", "hey jarvis"]

@dataclass
class VoiceEvent:
    """Eventi voice per comunicazione con core system"""
    event_type: str  # "wake_detected", "speech_recognized", "tts_completed"
    data: Dict[str, Any]
    timestamp: datetime
    confidence: float = 0.0

class AudioSystem:
    """Gestione audio nativo multi-piattaforma (NO PyAudio)"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.platform = platform.system().lower()
        self.logger = logging.getLogger("jarvis.voice.audio")
        
    async def record_audio(self, duration: float = 5.0) -> Optional[bytes]:
        """Registra audio usando sistema nativo"""
        try:
            temp_file = tempfile.mktemp(suffix=".wav")
            
            if self.platform == "windows":
                # Windows: usa SoundRecorder o ffmpeg
                cmd = [
                    "ffmpeg", "-f", "dshow", "-i", "audio=Microphone",
                    "-t", str(duration), "-ar", str(self.sample_rate),
                    "-ac", "1", "-y", temp_file
                ]
            elif self.platform == "darwin":  # macOS
                # macOS: usa sox o ffmpeg
                cmd = [
                    "sox", "-d", "-r", str(self.sample_rate), "-c", "1",
                    "-b", "16", "-t", "wav", temp_file, "trim", "0", str(duration)
                ]
            else:  # Linux
                # Linux: usa arecord
                cmd = [
                    "arecord", "-f", "S16_LE", "-r", str(self.sample_rate),
                    "-c", "1", "-d", str(int(duration)), temp_file
                ]
            
            # Esegui comando recording
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()
            
            # Leggi file audio generato
            if os.path.exists(temp_file):
                with open(temp_file, 'rb') as f:
                    audio_data = f.read()
                os.unlink(temp_file)
                return audio_data
            
        except Exception as e:
            self.logger.error(f"Audio recording failed: {e}")
            return None
    
    async def play_audio(self, audio_data: bytes) -> bool:
        """Riproduce audio usando sistema nativo"""
        try:
            temp_file = tempfile.mktemp(suffix=".wav")
            
            # Scrivi audio data su file temporaneo
            with open(temp_file, 'wb') as f:
                f.write(audio_data)
            
            if self.platform == "windows":
                # Windows: usa Windows Media Player o ffplay
                cmd = ["ffplay", "-nodisp", "-autoexit", temp_file]
            elif self.platform == "darwin":  # macOS
                # macOS: usa afplay
                cmd = ["afplay", temp_file]
            else:  # Linux
                # Linux: usa aplay
                cmd = ["aplay", temp_file]
            
            # Esegui playback
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()
            
            # Cleanup
            if os.path.exists(temp_file):
                os.unlink(temp_file)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Audio playback failed: {e}")
            return False

class PiperTTS:
    """Piper Neural TTS Engine (completamente locale)"""
    
    def __init__(self, model_name: str = "it_IT-riccardo-x_low"):
        self.model_name = model_name
        self.logger = logging.getLogger("jarvis.voice.piper")
        self.is_initialized = False
        
    async def initialize(self) -> bool:
        """Inizializza Piper TTS"""
        try:
            self.logger.info("Initializing Piper TTS...")
            
            # Verifica se piper è installato
            process = await asyncio.create_subprocess_exec(
                "piper", "--help",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                self.logger.error("Piper TTS not found. Install with: pip install piper-tts")
                return False
            
            self.is_initialized = True
            self.logger.info(f"Piper TTS initialized with model: {self.model_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Piper TTS initialization failed: {e}")
            return False
    
    async def synthesize(self, text: str) -> Optional[bytes]:
        """Sintetizza testo in audio con Piper"""
        if not self.is_initialized:
            return None
            
        try:
            temp_output = tempfile.mktemp(suffix=".wav")
            
            # Comando Piper per sintesi
            cmd = [
                "piper",
                "--model", self.model_name,
                "--output_file", temp_output
            ]
            
            # Esegui sintesi
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate(input=text.encode())
            
            if process.returncode == 0 and os.path.exists(temp_output):
                # Leggi audio generato
                with open(temp_output, 'rb') as f:
                    audio_data = f.read()
                os.unlink(temp_output)
                return audio_data
            else:
                self.logger.error(f"Piper synthesis failed: {stderr.decode()}")
                return None
                
        except Exception as e:
            self.logger.error(f"Piper synthesis error: {e}")
            return None

class WhisperSTT:
    """Whisper STT Engine (accesso diretto)"""
    
    def __init__(self, model_name: str = "base", language: str = "it"):
        self.model_name = model_name
        self.language = language
        self.model = None
        self.logger = logging.getLogger("jarvis.voice.whisper")
        self.is_initialized = False
        
    async def initialize(self) -> bool:
        """Inizializza Whisper model"""
        try:
            self.logger.info(f"Loading Whisper model: {self.model_name}")
            
            # Carica model in thread separato per non bloccare
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None, 
                lambda: whisper.load_model(self.model_name)
            )
            
            self.is_initialized = True
            self.logger.info("Whisper model loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Whisper initialization failed: {e}")
            return False
    
    async def transcribe_audio(self, audio_data: bytes) -> Optional[Dict[str, Any]]:
        """Trascrivi audio con Whisper"""
        if not self.is_initialized or not self.model:
            return None
            
        try:
            # Salva audio temporaneo
            temp_file = tempfile.mktemp(suffix=".wav")
            with open(temp_file, 'wb') as f:
                f.write(audio_data)
            
            # Transcription in thread separato
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.model.transcribe(
                    temp_file,
                    language=self.language,
                    fp16=False  # CPU compatibility
                )
            )
            
            # Cleanup
            if os.path.exists(temp_file):
                os.unlink(temp_file)
            
            return {
                "text": result["text"].strip(),
                "language": result["language"],
                "confidence": 1.0  # Whisper non fornisce confidence
            }
            
        except Exception as e:
            self.logger.error(f"Whisper transcription failed: {e}")
            return None

class VoiceManager:
    """
    Voice Manager principale secondo Master Plan:
    - Whisper STT diretto (no SpeechRecognition)
    - Piper TTS nativo
    - Sistema audio nativo (no PyAudio)
    - Wake word detection fuzzy
    - Eventi async per core integration
    """
    
    def __init__(self, config: VoiceConfig = None):
        self.config = config or VoiceConfig()
        self.logger = logging.getLogger("jarvis.voice")
        
        # Componenti voice
        self.whisper_stt = WhisperSTT(
            model_name=self.config.whisper_model,
            language=self.config.whisper_language
        )
        self.piper_tts = PiperTTS(model_name=self.config.piper_model)
        self.audio_system = AudioSystem(sample_rate=self.config.sample_rate)
        
        # Stato sistema
        self.is_initialized = False
        self.is_listening = False
        self.is_speaking = False
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {}
        
        # Statistics
        self.stats = {
            "wake_words_detected": 0,
            "speech_recognitions": 0,
            "tts_generations": 0,
            "errors": 0,
            "avg_stt_latency": 0.0,
            "avg_tts_latency": 0.0
        }
        
        # Threading
        self.listening_task = None
        self.shutdown_event = asyncio.Event()
        
    async def initialize(self) -> bool:
        """Inizializza Voice Manager completo"""
        try:
            self.logger.info("Initializing Voice Manager...")
            
            # 1. Inizializza Whisper STT
            self.logger.info("Initializing Whisper STT...")
            if not await self.whisper_stt.initialize():
                self.logger.error("Whisper STT initialization failed")
                return False
            
            # 2. Inizializza Piper TTS
            self.logger.info("Initializing Piper TTS...")
            if not await self.piper_tts.initialize():
                self.logger.error("Piper TTS initialization failed")
                return False
            
            # 3. Test sistema audio
            self.logger.info("Testing audio system...")
            test_audio = await self.audio_system.record_audio(duration=1.0)
            if test_audio is None:
                self.logger.warning("Audio system test failed, but continuing...")
            
            self.is_initialized = True
            self.logger.info("Voice Manager initialized successfully")
            
            # 4. Avvia listening se richiesto
            if self.config.always_listening:
                await self.start_listening()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Voice Manager initialization failed: {e}")
            return False
    
    def add_event_handler(self, event_type: str, handler: Callable):
        """Aggiungi handler per eventi voice"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    async def emit_event(self, event: VoiceEvent):
        """Emetti evento voice"""
        if event.event_type in self.event_handlers:
            for handler in self.event_handlers[event.event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                except Exception as e:
                    self.logger.error(f"Event handler error: {e}")
    
    def check_wake_word(self, text: str) -> bool:
        """Controlla wake words con fuzzy matching"""
        text_lower = text.lower().strip()
        
        for wake_word in self.config.wake_words:
            wake_word_lower = wake_word.lower()
            
            # Exact match
            if wake_word_lower in text_lower:
                return True
            
            # Fuzzy matching
            similarity = difflib.SequenceMatcher(
                None, wake_word_lower, text_lower
            ).ratio()
            
            if similarity >= self.config.wake_sensitivity:
                return True
        
        return False
    
    async def start_listening(self):
        """Avvia ascolto continuo per wake words"""
        if self.is_listening:
            return
        
        self.is_listening = True
        self.listening_task = asyncio.create_task(self._listening_loop())
        self.logger.info("Voice listening started")
    
    async def stop_listening(self):
        """Ferma ascolto continuo"""
        self.is_listening = False
        if self.listening_task:
            self.listening_task.cancel()
            try:
                await self.listening_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Voice listening stopped")
    
    async def _listening_loop(self):
        """Loop principale ascolto wake words"""
        while self.is_listening and not self.shutdown_event.is_set():
            try:
                # Registra audio per wake word detection
                audio_data = await self.audio_system.record_audio(duration=3.0)
                if audio_data is None:
                    await asyncio.sleep(1.0)
                    continue
                
                # Trascrivi con Whisper
                start_time = time.time()
                transcription = await self.whisper_stt.transcribe_audio(audio_data)
                stt_latency = time.time() - start_time
                
                if transcription and transcription["text"]:
                    self.stats["speech_recognitions"] += 1
                    self.stats["avg_stt_latency"] = (
                        self.stats["avg_stt_latency"] * 0.8 + stt_latency * 0.2
                    )
                    
                    # Controlla wake word
                    if self.check_wake_word(transcription["text"]):
                        self.stats["wake_words_detected"] += 1
                        
                        await self.emit_event(VoiceEvent(
                            event_type="wake_detected",
                            data={
                                "text": transcription["text"],
                                "wake_word_detected": True,
                                "confidence": transcription["confidence"]
                            },
                            timestamp=datetime.now(),
                            confidence=transcription["confidence"]
                        ))
                
            except Exception as e:
                self.logger.error(f"Listening loop error: {e}")
                self.stats["errors"] += 1
                await asyncio.sleep(1.0)
    
    async def process_voice_command(self, duration: float = 5.0) -> Optional[Dict[str, Any]]:
        """Processa comando vocale completo"""
        try:
            self.logger.info(f"Recording voice command for {duration}s...")
            
            # Registra comando esteso
            audio_data = await self.audio_system.record_audio(duration=duration)
            if audio_data is None:
                return None
            
            # Trascrivi con Whisper
            start_time = time.time()
            transcription = await self.whisper_stt.transcribe_audio(audio_data)
            stt_latency = time.time() - start_time
            
            if transcription:
                self.stats["speech_recognitions"] += 1
                self.stats["avg_stt_latency"] = (
                    self.stats["avg_stt_latency"] * 0.8 + stt_latency * 0.2
                )
                
                await self.emit_event(VoiceEvent(
                    event_type="speech_recognized",
                    data=transcription,
                    timestamp=datetime.now(),
                    confidence=transcription["confidence"]
                ))
                
                return transcription
            
        except Exception as e:
            self.logger.error(f"Voice command processing failed: {e}")
            self.stats["errors"] += 1
            return None
    
    async def speak(self, text: str) -> bool:
        """Sintetizza e riproduci testo con Piper TTS"""
        if self.is_speaking:
            return False
        
        try:
            self.is_speaking = True
            self.logger.info(f"Speaking: {text[:50]}...")
            
            # Sintetizza con Piper
            start_time = time.time()
            audio_data = await self.piper_tts.synthesize(text)
            tts_latency = time.time() - start_time
            
            if audio_data is None:
                return False
            
            # Riproduci audio
            success = await self.audio_system.play_audio(audio_data)
            
            if success:
                self.stats["tts_generations"] += 1
                self.stats["avg_tts_latency"] = (
                    self.stats["avg_tts_latency"] * 0.8 + tts_latency * 0.2
                )
                
                await self.emit_event(VoiceEvent(
                    event_type="tts_completed",
                    data={"text": text, "success": True},
                    timestamp=datetime.now(),
                    confidence=1.0
                ))
            
            return success
            
        except Exception as e:
            self.logger.error(f"Speech synthesis failed: {e}")
            self.stats["errors"] += 1
            return False
        finally:
            self.is_speaking = False
    
    async def get_status(self) -> Dict[str, Any]:
        """Ottieni stato Voice Manager"""
        return {
            "initialized": self.is_initialized,
            "listening": self.is_listening,
            "speaking": self.is_speaking,
            "whisper_model": self.config.whisper_model,
            "piper_model": self.config.piper_model,
            "wake_words": self.config.wake_words,
            "stats": self.stats.copy()
        }
    
    async def cleanup(self):
        """Cleanup Voice Manager"""
        self.logger.info("Cleaning up Voice Manager...")
        
        # Ferma listening
        await self.stop_listening()
        
        # Set shutdown event
        self.shutdown_event.set()
        
        self.logger.info("Voice Manager cleaned up")

# MAIN PER TESTING
async def main():
    """Test Voice Manager standalone"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test configurazione
    config = VoiceConfig(
        whisper_model="tiny",  # Veloce per test
        always_listening=False  # Test manuale
    )
    
    voice_manager = VoiceManager(config)
    
    # Test inizializzazione
    if await voice_manager.initialize():
        print("✅ Voice Manager initialized successfully")
        
        # Test TTS
        await voice_manager.speak("Ciao! Sono Jarvis, il tuo assistente AI.")
        
        # Test STT
        print("Say something for 3 seconds...")
        result = await voice_manager.process_voice_command(duration=3.0)
        if result:
            print(f"You said: {result['text']}")
        
        # Cleanup
        await voice_manager.cleanup()
    else:
        print("❌ Voice Manager initialization failed")

if __name__ == "__main__":
    asyncio.run(main())