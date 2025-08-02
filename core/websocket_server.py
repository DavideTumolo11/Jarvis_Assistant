#!/usr/bin/env python3
"""
JARVIS WEBSOCKET SERVER - VERSIONE INTEGRATA CON LLM MANAGER
===========================================================

WebSocket server completo che integra:
- Echo server per debug
- LLM Manager per risposte AI
- Chat completamente funzionante
- Debug massimo per troubleshooting

FUNZIONALITÀ:
- Riceve messaggi dal frontend
- Fa echo per debug
- Genera risposte AI tramite LLM Manager
- Invia text_command_response al frontend
- Log dettagliato di ogni operazione

BASATO SU: LLM Manager esistente + Echo server funzionante
"""

import asyncio
import websockets
import json
import logging
import time
import requests
from datetime import datetime
from pathlib import Path
import traceback
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict


# ================================
# CONFIGURAZIONE LLM
# ================================

@dataclass
class LLMConfig:
    """Configurazione LLM Manager"""
    ollama_host: str = "localhost:11434"
    ollama_timeout: int = 60
    primary_model: str = "llama3.2:1b"
    fallback_model: str = "qwen2.5:14b"
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    context_window: int = 4096
    max_conversation_turns: int = 10
    system_prompt_enabled: bool = True
    target_response_time: float = 5.0
    max_retries: int = 3
    stream_response: bool = False


@dataclass
class LLMResponse:
    """Risposta LLM strutturata"""
    text: str
    model_used: str
    tokens_generated: int
    generation_time: float
    context_used: int
    confidence: float
    metadata: Dict[str, Any]


@dataclass
class ConversationTurn:
    """Singolo turn di conversazione"""
    role: str
    content: str
    timestamp: float
    metadata: Dict[str, Any] = None


# ================================
# LLM MANAGER INTEGRATO
# ================================

class IntegratedLLMManager:
    """LLM Manager semplificato per integrazione WebSocket"""
    
    def __init__(self, config: LLMConfig = None):
        self.config = config or LLMConfig()
        self.is_initialized = False
        self.is_processing = False
        self.current_model = self.config.primary_model
        self.ollama_available = False
        self.conversation_history: List[ConversationTurn] = []
        self.system_prompt = self._build_system_prompt()
        
        # Stats
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0
        }
        
        # Logger
        self.logger = logging.getLogger("jarvis.llm")
    
    def _build_system_prompt(self) -> str:
        """System prompt per Jarvis"""
        return """Sei JARVIS, l'assistente AI di Iron Man. Caratteristiche:

PERSONALITA':
- Professionale ma amichevole
- Preciso e conciso nelle risposte
- Leggermente ironico come Tony Stark
- Sempre disponibile ad aiutare

COMPETENZE:
- Rispondi in italiano chiaro e naturale
- Fornisci informazioni accurate e utili
- Gestisci richieste di sistema e produttività
- Mantieni conversazioni coinvolgenti

STILE:
- Risposte brevi ma complete (max 2-3 frasi)
- Usa "Sir" quando appropriato
- Conferma sempre le azioni completate
- Se non sai qualcosa, dillo onestamente

Ricorda: sei l'assistente AI più avanzato del mondo, ma sempre al servizio dell'utente."""
    
    async def initialize(self) -> bool:
        """Inizializza LLM Manager"""
        try:
            self.logger.info("🧠 Initializing LLM Manager...")
            
            # Test Ollama connection
            if not await self._test_ollama_connection():
                self.logger.warning("⚠️ Ollama not available - using fallback responses")
                self.ollama_available = False
            else:
                self.ollama_available = True
            
            # Initialize context
            await self._initialize_context()
            
            self.is_initialized = True
            self.logger.info("✅ LLM Manager initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ LLM Manager initialization failed: {e}")
            return False
    
    async def _test_ollama_connection(self) -> bool:
        """Test Ollama connection"""
        try:
            response = requests.get(
                f"http://{self.config.ollama_host}/api/tags",
                timeout=5
            )
            return response.status_code == 200
        except:
            return False
    
    async def _initialize_context(self):
        """Initialize conversation context"""
        self.conversation_history.clear()
        
        if self.config.system_prompt_enabled and self.system_prompt:
            system_turn = ConversationTurn(
                role="system",
                content=self.system_prompt,
                timestamp=time.time(),
                metadata={"auto_generated": True}
            )
            self.conversation_history.append(system_turn)
    
    async def generate_response(self, user_input: str) -> str:
        """Genera risposta AI"""
        if not user_input.strip():
            return "Non ho ricevuto nessun messaggio. Come posso aiutarti?"
        
        try:
            start_time = time.time()
            self.is_processing = True
            self.stats["total_requests"] += 1
            
            self.logger.info(f"🎯 Generating response for: '{user_input[:50]}...'")
            
            # Add user input to history
            user_turn = ConversationTurn(
                role="user",
                content=user_input,
                timestamp=time.time()
            )
            self.conversation_history.append(user_turn)
            
            # Generate response
            if self.ollama_available:
                response = await self._generate_with_ollama(user_input)
            else:
                response = self._generate_fallback_response(user_input)
            
            # Add response to history
            if response:
                assistant_turn = ConversationTurn(
                    role="assistant",
                    content=response,
                    timestamp=time.time()
                )
                self.conversation_history.append(assistant_turn)
                
                # Update stats
                generation_time = time.time() - start_time
                self.stats["successful_requests"] += 1
                self.stats["average_response_time"] = (
                    (self.stats["average_response_time"] * (self.stats["successful_requests"] - 1) + generation_time)
                    / self.stats["successful_requests"]
                )
                
                self.logger.info(f"✅ Response generated in {generation_time:.2f}s")
                return response
            else:
                raise RuntimeError("Failed to generate response")
        
        except Exception as e:
            self.stats["failed_requests"] += 1
            self.logger.error(f"❌ Response generation failed: {e}")
            return f"Mi dispiace, ho avuto un problema nel processare la tua richiesta."
        
        finally:
            self.is_processing = False
    
    async def _generate_with_ollama(self, user_input: str) -> str:
        """Generate response with Ollama"""
        try:
            messages = []
            for turn in self.conversation_history:
                messages.append({
                    "role": turn.role,
                    "content": turn.content
                })
            
            payload = {
                "model": self.current_model,
                "messages": messages,
                "options": {
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                    "top_k": self.config.top_k,
                    "repeat_penalty": self.config.repeat_penalty,
                    "num_predict": self.config.max_tokens
                },
                "stream": False
            }
            
            response = requests.post(
                f"http://{self.config.ollama_host}/api/chat",
                json=payload,
                timeout=self.config.ollama_timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get("message", {}).get("content", "").strip()
                
                if response_text:
                    return response_text
                else:
                    raise ValueError("Empty response from Ollama")
            else:
                raise RuntimeError(f"Ollama API error: {response.status_code}")
        
        except Exception as e:
            self.logger.error(f"Ollama generation failed: {e}")
            return self._generate_fallback_response(user_input)
    
    def _generate_fallback_response(self, user_input: str) -> str:
        """Generate fallback response when Ollama is not available"""
        user_lower = user_input.lower()
        
        # Risposte statiche basate su pattern
        if any(word in user_lower for word in ["ciao", "salve", "buongiorno", "buonasera"]):
            return "Ciao! Sono JARVIS, il tuo assistente AI. Come posso aiutarti oggi?"
        
        elif any(word in user_lower for word in ["come stai", "come va", "tutto bene"]):
            return "Sto funzionando perfettamente, grazie per aver chiesto! Tutti i sistemi sono operativi."
        
        elif any(word in user_lower for word in ["che ore", "ora", "tempo"]):
            current_time = datetime.now().strftime("%H:%M")
            return f"Sono le {current_time}. Posso aiutarti con qualcos'altro?"
        
        elif any(word in user_lower for word in ["aiuto", "help", "cosa puoi fare"]):
            return "Posso aiutarti con informazioni, conversazioni e controllo sistema. Cosa ti serve?"
        
        elif any(word in user_lower for word in ["grazie", "thanks"]):
            return "Prego! È sempre un piacere essere d'aiuto."
        
        elif any(word in user_lower for word in ["chi sei", "cosa sei"]):
            return "Sono JARVIS, il tuo assistente AI personale. Sono qui per aiutarti con qualsiasi cosa tu abbia bisogno."
        
        else:
            return f"Ho ricevuto il tuo messaggio: '{user_input}'. Al momento sto usando risposte di fallback. Presto avrò Ollama attivo per risposte più intelligenti!"


# ================================
# WEBSOCKET SERVER INTEGRATO
# ================================

class JarvisWebSocketServer:
    """
    WebSocket Server integrato con LLM Manager.
    Gestisce echo + AI responses per chat completa.
    """
    
    def __init__(self):
        self.config = None
        self.server = None
        self.connected_clients = set()
        self.llm_manager = None
        
        # Setup logging
        self.setup_logging()
        self.logger.info("🚀 JARVIS WebSocket Server - VERSIONE INTEGRATA LLM")
        
        # Carica configurazione
        self.load_config()
        
        # Inizializza LLM Manager
        self.setup_llm_manager()
        
    def setup_logging(self):
        """Setup logging con output dettagliato"""
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s | %(levelname)8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)
        
    def load_config(self):
        """Carica configurazione da file JSON"""
        try:
            # Trova il path corretto per la config
            possible_paths = [
                Path("config/master_config.json"),
                Path("../config/master_config.json"),
                Path("../../config/master_config.json")
            ]
            
            config_path = None
            for path in possible_paths:
                if path.exists():
                    config_path = path
                    break
            
            if not config_path:
                self.logger.error("❌ Config file non trovato in nessuna posizione")
                self._use_default_config()
                return
                
            # Carica config con encoding sicuro per BOM
            with open(config_path, 'r', encoding='utf-8-sig') as f:
                self.config = json.load(f)
                
            self.logger.info(f"✅ Config caricata da: {config_path}")
            self.logger.debug(f"🔍 WebSocket config: {self.config.get('websocket', {})}")
            
        except Exception as e:
            self.logger.error(f"❌ Errore caricamento config: {e}")
            self._use_default_config()
    
    def _use_default_config(self):
        """Usa configurazione di default"""
        self.logger.info("📝 Usando configurazione di default")
        self.config = {
            "websocket": {
                "host": "localhost",
                "port": 8765
            },
            "llm": {
                "ollama_host": "localhost:11434",
                "primary_model": "llama3.2:1b",
                "max_tokens": 2048,
                "temperature": 0.7
            }
        }
    
    def setup_llm_manager(self):
        """Inizializza LLM Manager"""
        try:
            # Crea config LLM da configurazione generale
            llm_config_data = self.config.get('llm', {})
            llm_config = LLMConfig(
                ollama_host=llm_config_data.get('ollama_host', 'localhost:11434'),
                primary_model=llm_config_data.get('primary_model', 'llama3.2:1b'),
                max_tokens=llm_config_data.get('max_tokens', 2048),
                temperature=llm_config_data.get('temperature', 0.7)
            )
            
            self.llm_manager = IntegratedLLMManager(llm_config)
            self.logger.info("🧠 LLM Manager setup completato")
            
        except Exception as e:
            self.logger.error(f"❌ LLM Manager setup failed: {e}")
            self.llm_manager = None
    
    async def handle_client(self, websocket):
        """Gestisce connessione client singolo"""
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        
        try:
            # Registra client
            self.connected_clients.add(websocket)
            self.logger.info(f"🔗 CLIENT CONNESSO: {client_id}")
            self.logger.info(f"👥 Clients totali: {len(self.connected_clients)}")
            
            # Inizializza LLM Manager se non fatto
            if self.llm_manager and not self.llm_manager.is_initialized:
                await self.llm_manager.initialize()
            
            # Invia messaggio di benvenuto
            welcome_msg = {
                "type": "connection_established",
                "message": "WebSocket connesso - Chat AI attiva",
                "client_id": client_id,
                "timestamp": datetime.now().isoformat(),
                "llm_available": self.llm_manager.ollama_available if self.llm_manager else False
            }
            
            await websocket.send(json.dumps(welcome_msg))
            self.logger.info(f"📤 WELCOME inviato a {client_id}")
            
            # Loop principale - ascolta messaggi
            async for message in websocket:
                await self.process_message(websocket, message, client_id)
                
        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"🔌 CLIENT DISCONNESSO: {client_id}")
            
        except Exception as e:
            self.logger.error(f"❌ ERRORE gestione client {client_id}: {e}")
            self.logger.debug(f"🔍 Traceback: {traceback.format_exc()}")
            
        finally:
            # Cleanup
            self.connected_clients.discard(websocket)
            self.logger.info(f"🧹 Client {client_id} rimosso. Rimasti: {len(self.connected_clients)}")
    
    async def process_message(self, websocket, message, client_id):
        """Processa messaggio ricevuto - Echo + AI Response"""
        try:
            self.logger.info(f"📥 MESSAGGIO RICEVUTO da {client_id}")
            self.logger.debug(f"📄 Raw message: {message}")
            
            # Parse JSON
            try:
                data = json.loads(message)
                self.logger.info(f"✅ JSON VALIDO: {data}")
            except json.JSONDecodeError as e:
                self.logger.warning(f"⚠️ JSON NON VALIDO: {e}")
                data = {"raw_message": message, "parse_error": str(e)}
            
            # 1. SEMPRE INVIA ECHO RESPONSE (per debug)
            await self.send_echo_response(websocket, data, client_id, message)
            
            # 2. SE È TEXT_COMMAND, GENERA ANCHE AI RESPONSE
            if isinstance(data, dict) and data.get("type") == "text_command":
                await self.handle_text_command(websocket, data, client_id)
            
            # 3. GESTISCI ALTRI TIPI DI MESSAGGIO
            elif isinstance(data, dict):
                await self.handle_other_message(websocket, data, client_id)
            
        except Exception as e:
            self.logger.error(f"❌ ERRORE processing message da {client_id}: {e}")
            await self.send_error_response(websocket, str(e), client_id)
    
    async def send_echo_response(self, websocket, data, client_id, original_message):
        """Invia echo response per debug"""
        try:
            echo_response = {
                "type": "echo_response",
                "original_message": data,
                "processed_at": datetime.now().isoformat(),
                "client_id": client_id,
                "message_length": len(original_message),
                "debug_info": {
                    "server_status": "INTEGRATO LLM + ECHO",
                    "clients_connected": len(self.connected_clients),
                    "llm_available": self.llm_manager.ollama_available if self.llm_manager else False,
                    "message_processed": True
                }
            }
            
            response_json = json.dumps(echo_response, ensure_ascii=False, indent=2)
            await websocket.send(response_json)
            
            self.logger.info(f"📤 ECHO RESPONSE inviata a {client_id}")
            
        except Exception as e:
            self.logger.error(f"❌ Errore invio echo response: {e}")
    
    async def handle_text_command(self, websocket, data, client_id):
        """Gestisce text_command e genera AI response"""
        try:
            text = data.get("text", "").strip()
            if not text:
                return
            
            self.logger.info(f"💬 TEXT COMMAND: '{text}' da {client_id}")
            
            if not self.llm_manager:
                ai_response = "LLM Manager non disponibile. Echo mode attivo."
            else:
                # Genera risposta AI
                self.logger.info("🧠 Generazione risposta AI...")
                ai_response = await self.llm_manager.generate_response(text)
            
            # Invia AI response al frontend
            ai_response_msg = {
                "type": "text_command_response",
                "text": ai_response,
                "original_text": text,
                "client_id": client_id,
                "timestamp": datetime.now().isoformat(),
                "model_used": self.llm_manager.current_model if self.llm_manager else "fallback",
                "llm_stats": self.llm_manager.stats if self.llm_manager else {}
            }
            
            await websocket.send(json.dumps(ai_response_msg, ensure_ascii=False))
            self.logger.info(f"🤖 AI RESPONSE inviata a {client_id}: '{ai_response[:50]}...'")
            
        except Exception as e:
            self.logger.error(f"❌ Errore handle_text_command: {e}")
            await self.send_error_response(websocket, f"Errore generazione risposta: {e}", client_id)
    
    async def handle_other_message(self, websocket, data, client_id):
        """Gestisce altri tipi di messaggio"""
        msg_type = data.get("type", "unknown")
        self.logger.info(f"🎯 Altro messaggio tipo: {msg_type}")
        
        if msg_type == "ping":
            pong_response = {
                "type": "pong",
                "timestamp": datetime.now().isoformat(),
                "client_id": client_id
            }
            await websocket.send(json.dumps(pong_response))
            self.logger.info("🏓 Pong inviato")
        
        elif msg_type == "get_llm_status":
            if self.llm_manager:
                status = await self.llm_manager.get_status()
                status_response = {
                    "type": "llm_status_response",
                    "status": status,
                    "timestamp": datetime.now().isoformat()
                }
                await websocket.send(json.dumps(status_response))
                self.logger.info("📊 LLM status inviato")
    
    async def send_error_response(self, websocket, error_message, client_id):
        """Invia error response"""
        try:
            error_response = {
                "type": "error",
                "error": error_message,
                "client_id": client_id,
                "timestamp": datetime.now().isoformat()
            }
            await websocket.send(json.dumps(error_response))
            self.logger.info(f"📤 ERROR RESPONSE inviata a {client_id}")
        except:
            self.logger.error(f"❌ Impossibile inviare error response a {client_id}")
    
    async def start_server(self):
        """Avvia il WebSocket server"""
        try:
            ws_config = self.config.get("websocket", {})
            host = ws_config.get("host", "localhost")
            port = ws_config.get("port", 8765)
            
            self.logger.info(f"🚀 Avvio WebSocket server integrato su {host}:{port}")
            
            # Avvia server
            self.server = await websockets.serve(
                self.handle_client,
                host,
                port,
                ping_interval=20,
                ping_timeout=10
            )
            
            self.logger.info(f"✅ SERVER ATTIVO su ws://{host}:{port}")
            self.logger.info("🎯 MODALITÀ: ECHO + AI CHAT - Risposte complete")
            self.logger.info("💬 Chat testuale completamente funzionante")
            self.logger.info("🧠 LLM Manager integrato per AI responses")
            
            # Mantieni server attivo
            await self.server.wait_closed()
            
        except Exception as e:
            self.logger.error(f"❌ ERRORE avvio server: {e}")
            self.logger.debug(f"🔍 Traceback: {traceback.format_exc()}")
            raise
    
    def stop_server(self):
        """Ferma il server"""
        if self.server:
            self.logger.info("🛑 Fermando WebSocket server...")
            self.server.close()


# ================================
# ENTRY POINT
# ================================

async def main():
    """Funzione principale"""
    server = JarvisWebSocketServer()
    
    try:
        await server.start_server()
    except KeyboardInterrupt:
        server.logger.info("🛑 Interruzione da tastiera (Ctrl+C)")
    except Exception as e:
        server.logger.error(f"❌ ERRORE FATALE: {e}")
    finally:
        server.stop_server()
        if server.llm_manager:
            await server.llm_manager.cleanup()
        server.logger.info("👋 Server terminato")


if __name__ == "__main__":
    print("=" * 70)
    print("🎯 JARVIS WEBSOCKET SERVER - INTEGRATO CON LLM MANAGER")
    print("=" * 70)
    print("📋 FUNZIONALITÀ:")
    print("   🔄 Echo response per debug")
    print("   🤖 AI response tramite LLM Manager")
    print("   💬 Chat testuale completa")
    print("   🧠 Ollama + Mistral integration")
    print("=" * 70)
    print()
    
    asyncio.run(main())