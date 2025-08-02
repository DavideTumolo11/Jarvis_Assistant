"""
JARVIS AI ASSISTANT - CORE ORCHESTRATORE DEFINITIVO CORRETTO
===========================================================

Questo è il cervello centrale di Jarvis che coordina tutti i componenti:
- Inizializza tutti i Manager (Voice, LLM, Memory, Plugin, WebSocket)
- Gestisce il ciclo di vita completo del sistema
- Coordina comunicazioni inter-componenti
- Centralizza error handling e recovery
- Gestisce configurazione globale
- Monitoring e health checks

UNICODE SAFE: Nessuna emoji nei log per compatibilità Windows

ARCHITETTURA:
Frontend ↔ WebSocket Server ↔ JARVIS CORE ↔ Voice Manager
                                    ↕
                            LLM Manager ↔ Memory Manager
                                    ↕
                               Plugin Manager

VERSIONE: DEFINITIVA E CORRETTA
"""

import asyncio
import json
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, asdict
from datetime import datetime

# IMPORT COMPONENTI JARVIS
try:
    from websocket_server import JarvisWebSocketServer, WebSocketConfig
    from voice_manager import VoiceManager, VoiceConfig, VoiceEvent
    from llm_manager import LLMManager, LLMConfig
    from memory_manager import MemoryManager, MemoryConfig
    from plugin_manager import PluginManager, PluginConfig
except ImportError as e:
    print(f"Missing Jarvis components: {e}")
    print("Make sure all manager files are in the same directory")
    sys.exit(1)


# ================================
# CONFIGURAZIONE SISTEMA
# ================================

@dataclass
class JarvisConfig:
    """Configurazione master per tutto il sistema Jarvis"""
    # SYSTEM
    name: str = "JARVIS"
    version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"
    
    # COMPONENT CONFIGS
    voice_config: VoiceConfig = None
    llm_config: LLMConfig = None
    websocket_config: WebSocketConfig = None
    memory_config: MemoryConfig = None
    plugin_config: PluginConfig = None
    
    # PERFORMANCE
    startup_timeout: int = 30
    shutdown_timeout: int = 10
    health_check_interval: int = 60
    
    def __post_init__(self):
        # Inizializza config componenti se non fornite
        if self.voice_config is None:
            self.voice_config = VoiceConfig()
        if self.llm_config is None:
            self.llm_config = LLMConfig()
        if self.websocket_config is None:
            self.websocket_config = WebSocketConfig()
        if self.memory_config is None:
            self.memory_config = MemoryConfig()
        if self.plugin_config is None:
            self.plugin_config = PluginConfig()


@dataclass
class SystemStatus:
    """Stato completo del sistema Jarvis"""
    core_status: str = "INITIALIZING"
    voice_status: str = "OFFLINE"
    llm_status: str = "OFFLINE"
    memory_status: str = "OFFLINE"
    plugin_status: str = "OFFLINE"
    websocket_status: str = "OFFLINE"
    
    uptime_seconds: int = 0
    total_requests: int = 0
    errors_count: int = 0
    last_error: Optional[str] = None
    
    performance_metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.performance_metrics is None:
            self.performance_metrics = {}


# ================================
# JARVIS CORE PRINCIPALE
# ================================

class JarvisCore:
    """
    Core System definitivo per Jarvis AI Assistant
    
    Responsabilità:
    - Orchestrazione completa di tutti i componenti
    - Lifecycle management (startup → running → shutdown)
    - Inter-component communication e events
    - Error handling centralizzato
    - Configuration management
    - Health monitoring e auto-recovery
    - Performance tracking
    """
    
    def __init__(self, config: JarvisConfig = None):
        self.config = config or JarvisConfig()
        
        # STATO SISTEMA
        self.is_initialized = False
        self.is_running = False
        self.start_time = time.time()
        self.system_status = SystemStatus()
        
        # COMPONENTI MANAGER
        self.voice_manager: Optional[VoiceManager] = None
        self.llm_manager: Optional[LLMManager] = None
        self.memory_manager: Optional[MemoryManager] = None
        self.plugin_manager: Optional[PluginManager] = None
        self.websocket_server: Optional[JarvisWebSocketServer] = None
        
        # EVENT SYSTEM
        self._event_queue = asyncio.Queue()
        self._event_handlers: Dict[str, List] = {}
        self._event_task: Optional[asyncio.Task] = None
        
        # HEALTH MONITORING
        self._health_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        # LOGGING - Unicode Safe
        self.logger = self._setup_logging()
        
        # SIGNAL HANDLERS
        self._setup_signal_handlers()
        
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging centralizzato UNICODE SAFE per tutto Jarvis"""
        # Crea directory logs se non esiste
        log_dir = Path("../data/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure root logger con encoding UTF-8
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, self.config.log_level))
        
        file_handler = logging.FileHandler(
            log_dir / 'jarvis.log', 
            mode='a', 
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, self.config.log_level))
        
        # Formatter senza emoji
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.config.log_level))
        
        # Evita handler duplicati
        if not root_logger.handlers:
            root_logger.addHandler(console_handler)
            root_logger.addHandler(file_handler)
        
        logger = logging.getLogger("jarvis.core")
        logger.setLevel(getattr(logging, self.config.log_level))
        
        return logger
        
        
    def _setup_signal_handlers(self):
        """Setup signal handlers per shutdown graceful"""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating shutdown...")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)


    # ================================
    # SYSTEM LIFECYCLE
    # ================================
    
    async def initialize(self) -> bool:
        """Inizializza tutto il sistema Jarvis step-by-step"""
        try:
            self.logger.info("Initializing JARVIS AI Assistant...")
            self.logger.info(f"Version: {self.config.version}")
            self.logger.info(f"Debug mode: {self.config.debug}")
            
            self.system_status.core_status = "INITIALIZING"
            
            # 1. LOAD CONFIGURATION
            await self._load_configuration()
            
            # 2. INITIALIZE COMPONENTS
            if not await self._initialize_components():
                return False
            
            # 3. SETUP EVENT SYSTEM
            await self._setup_event_system()
            
            # 4. CROSS-COMPONENT INTEGRATION
            await self._integrate_components()
            
            # 5. START MONITORING
            await self._start_monitoring()
            
            # 6. FINAL VALIDATION
            if not await self._validate_system():
                return False
            
            self.is_initialized = True
            self.system_status.core_status = "READY"
            
            self.logger.info("JARVIS Core initialized successfully")
            await self._emit_system_event("core_initialized", {"version": self.config.version})
            
            return True
            
        except Exception as e:
            self.logger.error(f"JARVIS Core initialization failed: {e}")
            self.system_status.core_status = "ERROR"
            self.system_status.last_error = str(e)
            return False
    
    
    async def _load_configuration(self):
        """Carica configurazione da file JSON"""
        try:
            config_path = Path("config/master_config.json")
            
            if config_path.exists():
                self.logger.info("Loading configuration from master_config.json...")
                
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                # Update config con valori da file
                if "system" in config_data:
                    system_config = config_data["system"]
                    self.config.debug = system_config.get("debug", self.config.debug)
                    self.config.log_level = system_config.get("log_level", self.config.log_level)
                
                self.logger.info("Configuration loaded successfully")
            else:
                self.logger.info("Using default configuration (master_config.json not found)")
                
        except Exception as e:
            self.logger.warning(f"Configuration loading failed, using defaults: {e}")
    
    
    async def _initialize_components(self) -> bool:
        """Inizializza tutti i componenti in ordine corretto"""
        try:
            self.logger.info("Initializing system components...")
            
            # 1. VOICE MANAGER
            self.logger.info("Initializing Voice Manager...")
            self.voice_manager = VoiceManager(
                config=self.config.voice_config,
            )
            
            self.voice_manager.add_event_handler("wake_detected", self._handle_voice_event)
            self.voice_manager.add_event_handler("speech_recognized", self._handle_voice_event)
            
            if await self.voice_manager.initialize():
                self.system_status.voice_status = "READY"
                self.logger.info("Voice Manager initialized")
            else:
                self.system_status.voice_status = "ERROR"
                self.logger.error("Voice Manager initialization failed")
                return False
            
            # 2. LLM MANAGER
            self.logger.info("Initializing LLM Manager...")
            self.llm_manager = LLMManager(
                config=self.config.llm_config,
                memory_manager=None  # Sarà collegato dopo
            )
            
            if await self.llm_manager.initialize():
                self.system_status.llm_status = "READY"
                self.logger.info("LLM Manager initialized")
            else:
                self.system_status.llm_status = "ERROR"
                self.logger.error("LLM Manager initialization failed")
                return False
            
            # 3. MEMORY MANAGER
            self.logger.info("Initializing Memory Manager...")
            self.memory_manager = MemoryManager(
                config=self.config.memory_config
            )
            
            if await self.memory_manager.initialize():
                self.system_status.memory_status = "READY"
                self.logger.info("Memory Manager initialized")
                
                # Now inject memory manager into LLM manager
                self.llm_manager.memory_manager = self.memory_manager
            else:
                self.system_status.memory_status = "ERROR"
                self.logger.error("Memory Manager initialization failed")
                return False
            
            # 4. PLUGIN MANAGER
            self.logger.info("Initializing Plugin Manager...")
            self.plugin_manager = PluginManager(
                config=self.config.plugin_config
            )
            
            # Inject services into plugin manager
            self.plugin_manager.inject_services(
                voice_manager=self.voice_manager,
                llm_manager=self.llm_manager,
                memory_manager=self.memory_manager
            )
            
            if await self.plugin_manager.initialize():
                self.system_status.plugin_status = "READY"
                self.logger.info("Plugin Manager initialized")
            else:
                self.system_status.plugin_status = "ERROR"
                self.logger.error("Plugin Manager initialization failed")
                return False
            
            # 5. WEBSOCKET SERVER
            self.logger.info("Initializing WebSocket Server...")
            self.websocket_server = JarvisWebSocketServer(
                config=self.config.websocket_config
            )
            
            # Inject managers BEFORE validation
            await self.websocket_server.inject_managers(
                voice_manager=self.voice_manager,
                llm_manager=self.llm_manager,
                memory_manager=self.memory_manager,
                plugin_manager=self.plugin_manager
            )
            
            self.system_status.websocket_status = "READY"
            self.logger.info("WebSocket Server initialized")                 
            
            # Inietta TUTTI i manager nel WebSocket server
            self.websocket_server.inject_managers(
                voice_manager=self.voice_manager,
                llm_manager=self.llm_manager,
                memory_manager=self.memory_manager,
                plugin_manager=self.plugin_manager
            )
            
            self.system_status.websocket_status = "READY"
            self.logger.info("WebSocket Server initialized")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Component initialization failed: {e}")
            return False
    
    
    async def _setup_event_system(self):
        """Setup sistema eventi inter-componenti"""
        try:
            self.logger.info("Setting up event system...")
            
            # Avvia event processing task
            self._event_task = asyncio.create_task(self._process_events())
            
            # Register event handlers
            self._register_event_handlers()
            
            self.logger.info("Event system ready")
            
        except Exception as e:
            self.logger.error(f"Event system setup failed: {e}")
            raise
    
    
    def _register_event_handlers(self):
        """Registra handlers per eventi sistema"""
        # Voice events
        self.register_event_handler("wake_word_detected", self._on_wake_word_detected)
        self.register_event_handler("speech_recognized", self._on_speech_recognized)
        self.register_event_handler("voice_error", self._on_voice_error)
        
        # LLM events
        self.register_event_handler("generation_completed", self._on_llm_response)
        self.register_event_handler("llm_error", self._on_llm_error)
        
        # System events
        self.register_event_handler("component_error", self._on_component_error)
    
    
    async def _integrate_components(self):
        """Integra i componenti tra loro per comunicazione fluida"""
        try:
            self.logger.info("Integrating components...")
            
            # Collegamento Voice Manager → LLM Manager → Voice Manager (per TTS)
            # Questo è dove i componenti vengono collegati per lavorare insieme
            
            self.logger.info("Components integrated")
            
        except Exception as e:
            self.logger.error(f"Component integration failed: {e}")
            raise
    
    
    async def _start_monitoring(self):
        """Avvia monitoring e health checks"""
        try:
            self.logger.info("Starting system monitoring...")
            
            # Avvia health check task
            self._health_task = asyncio.create_task(self._health_check_loop())
            
            self.logger.info("Monitoring started")
            
        except Exception as e:
            self.logger.error(f"Monitoring startup failed: {e}")
            raise
    
    
    async def _validate_system(self) -> bool:
        """Validazione finale sistema"""
        try:
            self.logger.info("Validating system integrity...")
            
            # Test basic functionality
            validation_tests = [
                ("Voice Manager", self.voice_manager and self.voice_manager.is_initialized),
                ("LLM Manager", self.llm_manager and self.llm_manager.is_initialized),
                ("Memory Manager", self.memory_manager and self.memory_manager.is_initialized),
                ("Plugin Manager", self.plugin_manager and self.plugin_manager.is_initialized),
                ("WebSocket Server", self.websocket_server is not None),
                ("Event System", self._event_task and not self._event_task.done())
            ]
            
            for test_name, test_result in validation_tests:
                if test_result:
                    self.logger.info(f"{test_name} validation passed")
                else:
                    self.logger.error(f"{test_name} validation failed")
                    return False
            
            self.logger.info("System validation completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"System validation failed: {e}")
            return False


    # ================================
    # SYSTEM OPERATIONS
    # ================================
    
    async def start(self):
        """Avvia tutto il sistema Jarvis"""
        try:
            if not self.is_initialized:
                self.logger.error("System not initialized, run initialize() first")
                return False
            
            self.logger.info("Starting JARVIS system...")
            self.is_running = True
            self.system_status.core_status = "RUNNING"
            
            # Start components
            tasks = []
            
            # 1. Start Voice Manager listening
            if self.voice_manager:
                self.logger.info("Starting voice listening...")
                await self.voice_manager.start_listening()
            
            # 2. Start WebSocket Server
            if self.websocket_server:
                self.logger.info("Starting WebSocket server...")
                tasks.append(
                    asyncio.create_task(self.websocket_server.start_server())
                )
            
            # Emit startup complete event
            await self._emit_system_event("system_started", {
                "uptime": time.time() - self.start_time,
                "components": {
                    "voice": self.system_status.voice_status,
                    "llm": self.system_status.llm_status,
                    "websocket": self.system_status.websocket_status
                }
            })
            
            self.logger.info("JARVIS system started successfully")
            self.logger.info("Ready to assist! Try saying 'Hey Jarvis'")
            
            # Wait for shutdown signal
            await self._shutdown_event.wait()
            
            return True
            
        except Exception as e:
            self.logger.error(f"System startup failed: {e}")
            self.system_status.core_status = "ERROR"
            self.system_status.last_error = str(e)
            return False
    
    
    async def shutdown(self):
        """Shutdown graceful di tutto il sistema"""
        try:
            self.logger.info("Shutting down JARVIS system...")
            self.is_running = False
            self.system_status.core_status = "SHUTTING_DOWN"
            
            # Stop components in reverse order
            shutdown_tasks = []
            
            # 1. Stop WebSocket Server
            if self.websocket_server:
                self.logger.info("Stopping WebSocket server...")
                shutdown_tasks.append(
                    asyncio.create_task(self.websocket_server.stop_server())
                )
            
            # 2. Stop Plugin Manager
            if self.plugin_manager:
                self.logger.info("Stopping plugin system...")
                shutdown_tasks.append(
                    asyncio.create_task(self.plugin_manager.cleanup())
                )
            
            # 3. Stop Voice Manager
            if self.voice_manager:
                self.logger.info("Stopping voice system...")
                shutdown_tasks.append(
                    asyncio.create_task(self.voice_manager.cleanup())
                )
            
            # 4. Stop LLM Manager
            if self.llm_manager:
                self.logger.info("Stopping LLM system...")
                shutdown_tasks.append(
                    asyncio.create_task(self.llm_manager.cleanup())
                )
            
            # 5. Stop Memory Manager
            if self.memory_manager:
                self.logger.info("Stopping memory system...")
                shutdown_tasks.append(
                    asyncio.create_task(self.memory_manager.cleanup())
                )
            
            # Wait for all shutdowns
            if shutdown_tasks:
                await asyncio.gather(*shutdown_tasks, return_exceptions=True)
            
            # Stop internal tasks
            if self._event_task and not self._event_task.done():
                self._event_task.cancel()
            
            if self._health_task and not self._health_task.done():
                self._health_task.cancel()
            
            # Signal shutdown complete
            self._shutdown_event.set()
            
            self.system_status.core_status = "OFFLINE"
            self.logger.info("JARVIS system shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Shutdown error: {e}")
            self.system_status.last_error = str(e)


    # ================================
    # EVENT SYSTEM
    # ================================
    
    async def _process_events(self):
        """Process eventi dal queue in loop continuo"""
        self.logger.info("Event processing started")
        
        while True:
            try:
                # Get next event from queue
                event = await self._event_queue.get()
                
                # Process event
                await self._dispatch_event(event)
                
                # Mark task done
                self._event_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Event processing error: {e}")
                await asyncio.sleep(1)  # Avoid tight error loop
        
        self.logger.info("Event processing stopped")
    
    
    async def _dispatch_event(self, event: Dict[str, Any]):
        """Dispatch evento ai handler registrati"""
        event_type = event.get("type", "unknown")
        
        if event_type in self._event_handlers:
            handlers = self._event_handlers[event_type]
            
            for handler in handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                except Exception as e:
                    self.logger.error(f"Event handler error for {event_type}: {e}")
    
    
    def register_event_handler(self, event_type: str, handler):
        """Registra handler per tipo evento"""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)
    
    
    async def _emit_system_event(self, event_type: str, data: Dict[str, Any]):
        """Emetti evento sistema"""
        event = {
            "type": event_type,
            "data": data,
            "timestamp": time.time(),
            "source": "jarvis_core"
        }
        
        await self._event_queue.put(event)
    
    
    # ================================
    # EVENT HANDLERS
    # ================================
    
    async def _handle_voice_event(self, voice_event: VoiceEvent):
        """Handler per eventi dal Voice Manager"""
        event = {
            "type": f"voice_{voice_event.event_type}",
            "data": voice_event.data,
            "timestamp": voice_event.timestamp,
            "source": "voice_manager"
        }
        await self._event_queue.put(event)
    
    
    async def _handle_llm_event(self, llm_event: Dict[str, Any]):
        """Handler per eventi dal LLM Manager"""
        await self._event_queue.put(llm_event)
    
    
    async def _on_wake_word_detected(self, event: Dict[str, Any]):
        """Handler per wake word detection"""
        self.logger.info("Wake word detected!")
        self.system_status.total_requests += 1
        
        # Notify all components
        if self.websocket_server:
            await self.websocket_server._broadcast_message({
                "type": "wake_word_detected",
                "data": event["data"]
            })
    
    
    async def _on_speech_recognized(self, event: Dict[str, Any]):
        """Handler per speech recognition"""
        speech_data = event["data"]
        text = speech_data.get("text", "")
        
        self.logger.info(f"Speech recognized: '{text}'")
        
        # Se c'è wake word, processa con LLM
        if speech_data.get("wake_word_detected", False) and self.llm_manager:
            try:
                # Generate AI response
                response = await self.llm_manager.generate_response(text)
                
                # Speak response
                if response and self.voice_manager:
                    await self.voice_manager.speak(response)
                
            except Exception as e:
                self.logger.error(f"Speech processing error: {e}")
                self.system_status.errors_count += 1
                self.system_status.last_error = str(e)
    
    
    async def _on_llm_response(self, event: Dict[str, Any]):
        """Handler per risposta LLM completata"""
        self.logger.info("LLM response generated")
        
        # Broadcast to WebSocket clients
        if self.websocket_server:
            await self.websocket_server._broadcast_message({
                "type": "ai_response",
                "data": event["data"]
            })
    
    
    async def _on_component_error(self, event: Dict[str, Any]):
        """Handler per errori componenti"""
        error_data = event["data"]
        component = error_data.get("component", "unknown")
        error = error_data.get("error", "Unknown error")
        
        self.logger.error(f"Component error in {component}: {error}")
        
        self.system_status.errors_count += 1
        self.system_status.last_error = f"{component}: {error}"
        
        # Implement recovery logic if needed
        await self._attempt_component_recovery(component)
    
    
    async def _on_voice_error(self, event: Dict[str, Any]):
        """Handler specifico per errori voice"""
        await self._on_component_error({
            "type": "component_error",
            "data": {"component": "voice_manager", "error": event["data"].get("error")}
        })
    
    
    async def _on_llm_error(self, event: Dict[str, Any]):
        """Handler specifico per errori LLM"""
        await self._on_component_error({
            "type": "component_error", 
            "data": {"component": "llm_manager", "error": event["data"].get("error")}
        })


    # ================================
    # HEALTH MONITORING
    # ================================
    
    async def _health_check_loop(self):
        """Loop continuo per health checks"""
        self.logger.info("Health monitoring started")
        
        while not self._shutdown_event.is_set():
            try:
                await self._perform_health_check()
                await asyncio.sleep(self.config.health_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health check error: {e}")
                await asyncio.sleep(60)  # Retry after 1 minute
        
        self.logger.info("Health monitoring stopped")
    
    
    async def _perform_health_check(self):
        """Esegue health check di tutti i componenti"""
        try:
            # Update uptime
            self.system_status.uptime_seconds = int(time.time() - self.start_time)
            
            # Check each component
            if self.voice_manager:
                voice_status = await self.voice_manager.get_status()
                self.system_status.voice_status = "READY" if voice_status["initialized"] else "ERROR"
            
            if self.llm_manager:
                llm_status = await self.llm_manager.get_status()
                self.system_status.llm_status = "READY" if llm_status["initialized"] else "ERROR"
            
            # Log health status periodically
            if self.system_status.uptime_seconds % 300 == 0:  # Every 5 minutes
                self.logger.info(f"System healthy - Uptime: {self.system_status.uptime_seconds}s")
                
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
    
    
    async def _attempt_component_recovery(self, component: str):
        """Tenta recovery automatico di un componente"""
        try:
            self.logger.info(f"Attempting recovery for {component}...")
            
            # Implement component-specific recovery logic
            if component == "voice_manager" and self.voice_manager:
                # Try to restart voice listening
                await self.voice_manager.stop_listening()
                await asyncio.sleep(2)
                await self.voice_manager.start_listening()
            
            elif component == "llm_manager" and self.llm_manager:
                # Try to test LLM connection
                await self.llm_manager._test_ollama_connection()
            
            self.logger.info(f"Recovery attempt completed for {component}")
            
        except Exception as e:
            self.logger.error(f"Recovery failed for {component}: {e}")


    # ================================
    # STATUS & UTILITIES
    # ================================
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Ottieni stato completo del sistema"""
        status = asdict(self.system_status)
        
        # Add component details
        if self.voice_manager:
            status["voice_details"] = await self.voice_manager.get_status()
        
        if self.llm_manager:
            status["llm_details"] = await self.llm_manager.get_status()
        
        if self.memory_manager:
            status["memory_details"] = await self.memory_manager.get_status()
        
        if self.plugin_manager:
            status["plugin_details"] = await self.plugin_manager.get_status()
        
        return status
    
    
    def is_healthy(self) -> bool:
        """Controlla se il sistema è healthy"""
        critical_statuses = [
            self.system_status.core_status,
            self.system_status.voice_status,
            self.system_status.llm_status
        ]
        
        return all(status in ["READY", "RUNNING"] for status in critical_statuses)


# ================================
# MAIN ENTRY POINT
# ================================

async def main():
    """Entry point principale per Jarvis"""
    print("Starting JARVIS AI Assistant...")
    print("=" * 50)
    
    # Create Jarvis Core
    jarvis_config = JarvisConfig(debug=True)  # Debug mode for development
    jarvis = JarvisCore(jarvis_config)
    
    try:
        # Initialize system
        print("Initializing system...")
        if not await jarvis.initialize():
            print("Initialization failed")
            return 1
        
        print("Initialization completed")
        print("Starting system...")
        
        # Start system
        await jarvis.start()
        
        return 0
        
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
        return 0
    except Exception as e:
        print(f"Critical error: {e}")
        return 1
    finally:
        # Ensure cleanup
        await jarvis.shutdown()
        print("JARVIS shutdown completed")


if __name__ == "__main__":
    # Run JARVIS
    exit_code = asyncio.run(main())
    sys.exit(exit_code)