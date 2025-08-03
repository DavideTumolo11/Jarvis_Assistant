#!/usr/bin/env python3
"""
JARVIS WEBSOCKET SERVER - VERSIONE DEBUG STREAMING COMPLETA
=========================================================

WebSocket server con debug dettagliato per identificare problemi streaming.
Logging massivo per capire esattamente cosa succede nel callback.
"""

import asyncio
import websockets
import json
import logging
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass

# IMPORT LLM MANAGER DEFINITIVO CON STREAMING
try:
    from llm_manager import LLMManager, LLMConfig
    LLM_MANAGER_AVAILABLE = True
    print("✅ Using FULL LLM Manager with streaming support")
except ImportError:
    LLM_MANAGER_AVAILABLE = False
    print("❌ LLM Manager not found - using fallback")


# ================================
# WEBSOCKET SERVER CONFIGURATION
# ================================

@dataclass
class WebSocketConfig:
    """Configurazione WebSocket Server"""
    host: str = "localhost"
    port: int = 8765
    max_connections: int = 20
    message_timeout: int = 45
    heartbeat_interval: int = 30
    max_message_size: int = 1024 * 1024  # 1MB
    
    # Performance
    ping_interval: int = 20
    ping_timeout: int = 10
    close_timeout: int = 10


# ================================
# WEBSOCKET SERVER PRINCIPALE
# ================================

class JarvisWebSocketServer:
    """
    WebSocket Server con DEBUG STREAMING completo
    """
    
    def __init__(self, config: WebSocketConfig = None):
        self.config = config or WebSocketConfig()
        self.server = None
        self.connected_clients = set()
        self.llm_manager = None
        
        # Setup logging
        self.setup_logging()
        self.logger.info("🚀 JARVIS WebSocket Server - DEBUG STREAMING VERSION")
        
        # Carica configurazione
        self.load_config()
        
        # Inizializza LLM Manager
        self.setup_llm_manager()
        
        # Statistics
        self.stats = {
            "connections_total": 0,
            "messages_processed": 0,
            "ai_responses_generated": 0,
            "streaming_chunks_sent": 0,
            "errors_count": 0,
            "uptime_start": time.time()
        }
        
    def setup_logging(self):
        """Setup logging con output dettagliato"""
        logging.basicConfig(
            level=logging.INFO,
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
                self.master_config = json.load(f)
                
            self.logger.info(f"✅ Config caricata da: {config_path}")
            
            # Update WebSocket config da master config
            if "websocket" in self.master_config:
                ws_config = self.master_config["websocket"]
                self.config.host = ws_config.get("host", self.config.host)
                self.config.port = ws_config.get("port", self.config.port)
                self.config.max_connections = ws_config.get("max_connections", self.config.max_connections)
            
        except Exception as e:
            self.logger.error(f"❌ Errore caricamento config: {e}")
            self._use_default_config()
    
    def _use_default_config(self):
        """Usa configurazione di default"""
        self.logger.info("📝 Usando configurazione di default")
        self.master_config = {
            "websocket": {
                "host": "localhost",
                "port": 8765
            },
            "llm": {
                "ollama_host": "localhost:11434",
                "primary_model": "mistral:7b",
                "max_tokens": 2048,
                "temperature": 0.7
            }
        }
    
    def setup_llm_manager(self):
        """Inizializza LLM Manager definitivo con streaming"""
        try:
            if not LLM_MANAGER_AVAILABLE:
                self.logger.error("❌ LLM Manager non disponibile")
                self.llm_manager = None
                return
            
            # Crea config LLM da configurazione generale
            llm_config_data = self.master_config.get('llm', {})
            llm_config = LLMConfig(
                ollama_host=llm_config_data.get('ollama_host', 'localhost:11434'),
                primary_model=llm_config_data.get('primary_model', 'mistral:7b'),
                fallback_model=llm_config_data.get('fallback_model', 'llama3.2:1b'),
                max_tokens=llm_config_data.get('max_tokens', 2048),
                temperature=llm_config_data.get('temperature', 0.7),
                stream_response=True  # ✅ SEMPRE STREAMING
            )
            
            # Event callback per ricevere chunks streaming
            def event_callback(event):
                self.logger.debug(f"📨 LLM Event: {event['type']}")
            
            self.llm_manager = LLMManager(llm_config, event_callback=event_callback)
            self.logger.info("🧠 LLM Manager setup completato con STREAMING support")
            
        except Exception as e:
            self.logger.error(f"❌ LLM Manager setup failed: {e}")
            self.llm_manager = None
    
    async def handle_client(self, websocket):
        """Gestisce connessione client singolo con streaming support"""
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        
        try:
            # Registra client
            self.connected_clients.add(websocket)
            self.stats["connections_total"] += 1
            self.logger.info(f"🔗 CLIENT CONNESSO: {client_id}")
            self.logger.info(f"👥 Clients totali: {len(self.connected_clients)}")
            
            # Inizializza LLM Manager se non fatto
            if self.llm_manager and not self.llm_manager.is_initialized:
                self.logger.info("🧠 Initializing LLM Manager...")
                init_success = await self.llm_manager.initialize()
                if init_success:
                    self.logger.info("✅ LLM Manager initialized successfully")
                else:
                    self.logger.error("❌ LLM Manager initialization failed")
            
            # Invia messaggio di benvenuto con status
            welcome_msg = {
                "type": "connection_established",
                "message": "WebSocket connesso - DEBUG STREAMING MODE",
                "client_id": client_id,
                "timestamp": datetime.now().isoformat(),
                "llm_available": self.llm_manager.ollama_available if self.llm_manager else False,
                "streaming_enabled": True,
                "debug_mode": True
            }
            
            await websocket.send(json.dumps(welcome_msg))
            self.logger.info(f"📤 WELCOME DEBUG inviato a {client_id}")
            
            # Loop principale - ascolta messaggi
            async for message in websocket:
                await self.process_message(websocket, message, client_id)
                
        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"🔌 CLIENT DISCONNESSO: {client_id}")
            
        except Exception as e:
            self.logger.error(f"❌ ERRORE gestione client {client_id}: {e}")
            self.logger.debug(f"🔍 Traceback: {traceback.format_exc()}")
            self.stats["errors_count"] += 1
            
        finally:
            # Cleanup
            self.connected_clients.discard(websocket)
            self.logger.info(f"🧹 Client {client_id} rimosso. Rimasti: {len(self.connected_clients)}")
    
    async def process_message(self, websocket, message, client_id):
        """Processa messaggio ricevuto con gestione streaming"""
        try:
            self.stats["messages_processed"] += 1
            self.logger.info(f"📥 MESSAGGIO RICEVUTO da {client_id}")
            self.logger.debug(f"📄 Raw message: {message}")
            
            # Parse JSON
            try:
                data = json.loads(message)
                self.logger.info(f"✅ JSON VALIDO: {data}")
            except json.JSONDecodeError as e:
                self.logger.warning(f"⚠️ JSON NON VALIDO: {e}")
                await self.send_error_response(websocket, f"Invalid JSON: {e}", client_id)
                return
            
            # Gestisci diversi tipi di messaggio
            if isinstance(data, dict):
                message_type = data.get("type", "unknown")
                
                if message_type == "text_command":
                    await self.handle_text_command_streaming(websocket, data, client_id)
                elif message_type == "ping":
                    await self.handle_ping(websocket, client_id)
                elif message_type == "get_status":
                    await self.handle_get_status(websocket, client_id)
                elif message_type == "get_llm_status":
                    await self.handle_get_llm_status(websocket, client_id)
                else:
                    self.logger.info(f"❓ Unknown message type: {message_type}")
                    await self.send_error_response(websocket, f"Unknown message type: {message_type}", client_id)
            
        except Exception as e:
            self.logger.error(f"❌ ERRORE processing message da {client_id}: {e}")
            self.stats["errors_count"] += 1
            await self.send_error_response(websocket, str(e), client_id)
    
    async def handle_text_command_streaming(self, websocket, data, client_id):
        """
        Gestisce text_command con STREAMING REAL-TIME - VERSION DEBUG COMPLETA
        
        Questo metodo ora ha logging dettagliato per debuggare il callback streaming
        """
        try:
            text = data.get("text", "").strip()
            if not text:
                await self.send_error_response(websocket, "Empty text command", client_id)
                return
            
            self.logger.info(f"💬 TEXT COMMAND STREAMING DEBUG: '{text}' da {client_id}")
            
            if not self.llm_manager:
                # Fallback response
                fallback_response = "LLM Manager non disponibile. Sistema in modalità fallback."
                await self.send_final_response(websocket, fallback_response, text, client_id)
                return
            
            # STREAMING RESPONSE GENERATION
            self.logger.info("🧠 Generazione risposta AI con STREAMING DEBUG...")
            
            # VARIABILI DEBUG
            chunks_sent = 0
            callback_calls = 0
            
            # Callback per ricevere chunks in real-time - VERSION DEBUG
            async def streaming_callback(chunk_data):
                """Callback chiamato per ogni chunk ricevuto dal LLM - DEBUG VERSION"""
                nonlocal chunks_sent, callback_calls
                callback_calls += 1
                
                try:
                    self.logger.info(f"🔥 CALLBACK CHIAMATO #{callback_calls}: {chunk_data}")
                    
                    if chunk_data.get("type") == "response_chunk":
                        chunk_text = chunk_data.get("chunk", "")
                        is_final = chunk_data.get("is_final", False)
                        chunk_number = chunk_data.get("chunk_number", 0)
                        
                        self.logger.info(f"📦 CHUNK #{chunk_number}: '{chunk_text[:30]}...' (final: {is_final})")
                        
                        if chunk_text:  # Solo chunks con contenuto
                            # Invia chunk al frontend immediatamente
                            chunk_message = {
                                "type": "ai_response_chunk",
                                "chunk": chunk_text,
                                "chunk_number": chunk_number,
                                "is_final": is_final,
                                "client_id": client_id,
                                "timestamp": datetime.now().isoformat()
                            }
                            
                            # INVIO CHUNK - DEBUG
                            self.logger.info(f"📤 SENDING CHUNK #{chunk_number} to frontend...")
                            await websocket.send(json.dumps(chunk_message, ensure_ascii=False))
                            chunks_sent += 1
                            self.stats["streaming_chunks_sent"] += 1
                            
                            self.logger.info(f"✅ CHUNK #{chunk_number} SENT! Total sent: {chunks_sent}")
                        
                        if is_final:
                            # Invia risposta finale completa
                            full_response = chunk_data.get("full_response", "")
                            self.logger.info(f"🏁 FINAL CHUNK - Full response: '{full_response[:50]}...'")
                            await self.send_final_response(websocket, full_response, text, client_id)
                            
                    else:
                        self.logger.warning(f"❓ Unknown chunk type: {chunk_data.get('type')}")
                            
                except Exception as e:
                    self.logger.error(f"❌ Error in streaming callback: {e}")
                    self.logger.error(f"🔍 Callback traceback: {traceback.format_exc()}")
            
            # Genera risposta con streaming callback - DEBUG
            try:
                self.logger.info("🚀 STARTING LLM GENERATION with callback...")
                
                # Usa il metodo generate_response con callback integrato
                ai_response = await self.llm_manager.generate_response(
                    user_input=text,
                    context={"websocket_callback": streaming_callback},
                    websocket_callback=streaming_callback
                )
                
                self.stats["ai_responses_generated"] += 1
                
                # DEBUG FINAL STATS
                self.logger.info(f"🎯 STREAMING DEBUG COMPLETED:")
                self.logger.info(f"   📞 Callback calls: {callback_calls}")
                self.logger.info(f"   📦 Chunks sent: {chunks_sent}")
                self.logger.info(f"   📝 Final response: '{ai_response[:50]}...'")
                
                if callback_calls == 0:
                    self.logger.error("🚨 CALLBACK NEVER CALLED! Streaming not working!")
                elif chunks_sent == 0:
                    self.logger.error("🚨 NO CHUNKS SENT! Problem in callback logic!")
                else:
                    self.logger.info(f"✅ STREAMING SUCCESS: {chunks_sent} chunks sent in real-time")
                
            except Exception as e:
                self.logger.error(f"❌ Streaming generation failed: {e}")
                self.logger.error(f"🔍 Generation traceback: {traceback.format_exc()}")
                
                # Fallback a metodo normale se streaming fallisce
                self.logger.info("🔄 Fallback to normal response...")
                ai_response = await self.llm_manager.generate_response(text)
                await self.send_final_response(websocket, ai_response, text, client_id)
            
        except Exception as e:
            self.logger.error(f"❌ Errore handle_text_command_streaming: {e}")
            self.logger.error(f"🔍 Handler traceback: {traceback.format_exc()}")
            await self.send_error_response(websocket, f"Errore generazione risposta: {e}", client_id)
    
    async def send_final_response(self, websocket, ai_response, original_text, client_id):
        """Invia risposta finale completa"""
        try:
            final_response_msg = {
                "type": "text_command_response", 
                "text": ai_response,
                "original_text": original_text,
                "client_id": client_id,
                "timestamp": datetime.now().isoformat(),
                "model_used": self.llm_manager.current_model if self.llm_manager else "fallback",
                "streaming": True,
                "response_complete": True
            }
            
            await websocket.send(json.dumps(final_response_msg, ensure_ascii=False))
            self.logger.info(f"🤖 FINAL RESPONSE inviata a {client_id}: '{ai_response[:50]}...'")
            
        except Exception as e:
            self.logger.error(f"❌ Error sending final response: {e}")
    
    async def handle_ping(self, websocket, client_id):
        """Gestisce ping messages"""
        pong_response = {
            "type": "pong",
            "timestamp": datetime.now().isoformat(),
            "client_id": client_id,
            "server_uptime": time.time() - self.stats["uptime_start"]
        }
        await websocket.send(json.dumps(pong_response))
        self.logger.debug(f"🏓 Pong inviato a {client_id}")
    
    async def handle_get_status(self, websocket, client_id):
        """Gestisce richieste status server"""
        status_response = {
            "type": "server_status_response",
            "status": self.get_server_stats(),
            "timestamp": datetime.now().isoformat()
        }
        await websocket.send(json.dumps(status_response))
        self.logger.info(f"📊 Server status inviato a {client_id}")
    
    async def handle_get_llm_status(self, websocket, client_id):
        """Gestisce richieste status LLM"""
        if self.llm_manager:
            llm_status = await self.llm_manager.get_status()
        else:
            llm_status = {"error": "LLM Manager not available"}
        
        status_response = {
            "type": "llm_status_response", 
            "status": llm_status,
            "timestamp": datetime.now().isoformat()
        }
        await websocket.send(json.dumps(status_response))
        self.logger.info(f"🧠 LLM status inviato a {client_id}")
    
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
            self.logger.info(f"📤 ERROR RESPONSE inviata a {client_id}: {error_message}")
        except:
            self.logger.error(f"❌ Impossibile inviare error response a {client_id}")
    
    def get_server_stats(self) -> Dict[str, Any]:
        """Ottieni statistiche server"""
        uptime = time.time() - self.stats["uptime_start"]
        
        return {
            "connected_clients": len(self.connected_clients),
            "total_connections": self.stats["connections_total"],
            "messages_processed": self.stats["messages_processed"],
            "ai_responses_generated": self.stats["ai_responses_generated"],
            "streaming_chunks_sent": self.stats["streaming_chunks_sent"],
            "errors_count": self.stats["errors_count"],
            "uptime_seconds": uptime,
            "uptime_formatted": f"{uptime//3600:.0f}h {(uptime%3600)//60:.0f}m {uptime%60:.0f}s",
            "llm_manager_available": self.llm_manager is not None,
            "llm_initialized": self.llm_manager.is_initialized if self.llm_manager else False,
            "streaming_enabled": True,
            "debug_mode": True
        }
    
    async def start_server(self):
        """Avvia il WebSocket server"""
        try:
            self.logger.info(f"🚀 Avvio WebSocket server DEBUG su {self.config.host}:{self.config.port}")
            
            # Configurazione server ottimizzata
            self.server = await websockets.serve(
                self.handle_client,
                self.config.host,
                self.config.port,
                ping_interval=self.config.ping_interval,
                ping_timeout=self.config.ping_timeout,
                close_timeout=self.config.close_timeout,
                max_size=self.config.max_message_size,
                compression=None  # Disabilita compressione per minor latency
            )
            
            self.logger.info(f"✅ SERVER DEBUG ATTIVO su ws://{self.config.host}:{self.config.port}")
            self.logger.info("🔍 MODALITÀ: DEBUG STREAMING MASSIVO")
            self.logger.info("💬 Logging dettagliato per ogni chunk")
            self.logger.info("🧠 Callback tracking completo")
            
            # Mantieni server attivo
            await self.server.wait_closed()
            
        except Exception as e:
            self.logger.error(f"❌ ERRORE avvio server: {e}")
            self.logger.debug(f"🔍 Traceback: {traceback.format_exc()}")
            raise
    
    async def stop_server(self):
        """Ferma il server gracefully"""
        if self.server:
            self.logger.info("🛑 Fermando WebSocket server...")
            self.server.close()
            await self.server.wait_closed()
            
            # Cleanup LLM Manager
            if self.llm_manager:
                await self.llm_manager.cleanup()
                
            self.logger.info("✅ Server fermato correttamente")


# ================================
# ENTRY POINT
# ================================

async def main():
    """Funzione principale"""
    # Configuration
    config = WebSocketConfig()
    server = JarvisWebSocketServer(config)
    
    try:
        await server.start_server()
    except KeyboardInterrupt:
        server.logger.info("🛑 Interruzione da tastiera (Ctrl+C)")
    except Exception as e:
        server.logger.error(f"❌ ERRORE FATALE: {e}")
    finally:
        await server.stop_server()
        server.logger.info("👋 Server terminato")


if __name__ == "__main__":
    print("=" * 70)
    print("🔍 JARVIS WEBSOCKET SERVER - DEBUG STREAMING VERSION")
    print("=" * 70)
    print("📋 DEBUG FEATURES:")
    print("   🔥 Logging massivo per ogni callback")
    print("   📦 Tracking dettagliato di ogni chunk")
    print("   🚨 Error detection completo")
    print("   📊 Statistics streaming real-time")
    print("=" * 70)
    print()
    
    asyncio.run(main())