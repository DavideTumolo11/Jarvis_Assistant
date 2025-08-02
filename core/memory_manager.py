"""
JARVIS AI ASSISTANT - MEMORY MANAGER DEFINITIVO
===============================================

Questo file gestisce tutta la memoria e persistenza di Jarvis:
- Database SQLite per conversazioni e dati strutturati
- ChromaDB opzionale per semantic search
- Cache in-memory per performance
- Backup automatico locale
- Context management per LLM
- Privacy-first: tutto locale, nessun cloud

STACK TECNOLOGICO DEFINITIVO:
- SQLite: Database primario sempre disponibile
- ChromaDB: Vector database opzionale per semantic search
- SQLAlchemy: ORM per gestione database elegante
- In-memory cache: TTL cache per performance

IMPORTANTE: Questo file è DEFINITIVO e COMPLETO
"""

import asyncio
import json
import logging
import sqlite3
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import hashlib

# DIPENDENZE DATABASE
try:
    import sqlalchemy as sa
    from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Text, Float, DateTime, Boolean
    from sqlalchemy.orm import sessionmaker, declarative_base
    from sqlalchemy.pool import StaticPool
except ImportError:
    print("❌ Missing SQLAlchemy: pip install sqlalchemy")
    exit(1)

# DIPENDENZE OPZIONALI
try:
    import chromadb
    from chromadb import Client
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("ℹ️ ChromaDB not available - using SQLite only")


# ================================
# CONFIGURAZIONE E MODELLI
# ================================

@dataclass
class MemoryConfig:
    """Configurazione Memory Manager"""
    # DATABASE CONFIGURATION
    sqlite_path: str = "data/jarvis_memory.db"
    chromadb_path: str = "data/chromadb"
    
    # CACHE CONFIGURATION
    cache_enabled: bool = True
    cache_ttl: int = 3600  # 1 hour
    max_cache_size: int = 1000
    
    # CONVERSATION MANAGEMENT
    max_conversation_history: int = 100
    conversation_cleanup_days: int = 30
    
    # SEMANTIC SEARCH
    semantic_search_enabled: bool = CHROMADB_AVAILABLE
    embedding_model: str = "all-MiniLM-L6-v2"
    semantic_threshold: float = 0.7
    
    # BACKUP
    backup_enabled: bool = True
    backup_interval_hours: int = 24
    max_backups: int = 7


# SQLALCHEMY MODELS
Base = declarative_base()

class Conversation(Base):
    """Modello per conversazioni"""
    __tablename__ = 'conversations'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String(50), default="default_user")
    user_input = Column(Text, nullable=False)
    ai_response = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    session_id = Column(String(50))
    meta_data = Column(Text)  # ← RINOMINATO!
    processing_time = Column(Float)
    tokens_used = Column(Integer)

class UserPreference(Base):
    """Modello per preferenze utente"""
    __tablename__ = 'user_preferences'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String(50), nullable=False)
    key = Column(String(100), nullable=False)
    value = Column(Text)
    updated_at = Column(DateTime, default=datetime.utcnow)

class SystemMemory(Base):
    """Modello per memoria sistema"""
    __tablename__ = 'system_memory'
    
    id = Column(Integer, primary_key=True)
    key = Column(String(100), nullable=False, unique=True)
    value = Column(Text)
    type = Column(String(20))  # string, json, binary
    expires_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)


@dataclass
class MemoryEntry:
    """Singola entry di memoria"""
    id: Optional[int]
    content: str
    metadata: Dict[str, Any]
    timestamp: datetime
    embedding: Optional[List[float]] = None


# ================================
# MEMORY MANAGER PRINCIPALE
# ================================

class MemoryManager:
    """
    Memory Manager definitivo per Jarvis AI Assistant
    
    Gestisce:
    - SQLite database per persistenza strutturata
    - ChromaDB per semantic search (opzionale)
    - In-memory cache per performance
    - Conversation history e context management
    - User preferences e system settings
    - Backup automatico e cleanup
    """
    
    def __init__(self, config: MemoryConfig = None):
        self.config = config or MemoryConfig()
        
        # STATO MEMORY MANAGER
        self.is_initialized = False
        self.db_engine = None
        self.db_session = None
        self.chroma_client = None
        self.chroma_collection = None
        
        # IN-MEMORY CACHE
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self._cache_lock = threading.RLock()
        
        # BACKUP SYSTEM
        self._backup_task: Optional[asyncio.Task] = None
        self._stop_backup = asyncio.Event()
        
        # LOGGING
        self.logger = self._setup_logging()
        
        # STATISTICS
        self.stats = {
            "total_conversations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "semantic_searches": 0,
            "backups_created": 0
        }
        
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging per Memory Manager"""
        logger = logging.getLogger("jarvis.memory")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger


    # ================================
    # INITIALIZATION
    # ================================
    
    async def initialize(self) -> bool:
        """Inizializza Memory Manager e database"""
        try:
            self.logger.info("💾 Initializing Memory Manager...")
            
            # 1. Setup directories
            await self._setup_directories()
            
            # 2. Initialize SQLite
            if not await self._init_sqlite():
                return False
            
            # 3. Initialize ChromaDB (opzionale)
            if self.config.semantic_search_enabled:
                 import os
                 os.environ["ANONYMIZED_TELEMETRY"] = "False"  # Add this line
    
                 self.chroma_client = chromadb.PersistentClient(
                      path=self.config.chromadb_path
                )
            
            # 4. Load cache
            await self._load_cache()
            
            # 5. Start backup system
            if self.config.backup_enabled:
                await self._start_backup_system()
            
            # 6. Run maintenance
            await self._run_maintenance()
            
            self.is_initialized = True
            self.logger.info("✅ Memory Manager initialized successfully")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Memory Manager initialization failed: {e}")
            return False
    
    
    async def _setup_directories(self):
        """Crea directory necessarie"""
        try:
            Path("data").mkdir(exist_ok=True)
            Path("data/logs").mkdir(exist_ok=True)
            Path("data/backups").mkdir(exist_ok=True)
            
            if self.config.semantic_search_enabled:
                Path(self.config.chromadb_path).mkdir(parents=True, exist_ok=True)
            
            self.logger.info("📁 Directories setup completed")
            
        except Exception as e:
            self.logger.error(f"❌ Directory setup failed: {e}")
            raise
    
    
    async def _init_sqlite(self) -> bool:
        """Inizializza database SQLite"""
        try:
            self.logger.info(f"🗃️ Initializing SQLite database: {self.config.sqlite_path}")
            
            # Create engine
            self.db_engine = create_engine(
                f"sqlite:///{self.config.sqlite_path}",
                poolclass=StaticPool,
                connect_args={
                    "check_same_thread": False,
                    "timeout": 20
                },
                echo=False  # Set to True for SQL debugging
            )
            
            # Create tables
            Base.metadata.create_all(self.db_engine)
            
            # Create session factory
            Session = sessionmaker(bind=self.db_engine)
            self.db_session = Session()
            
            # Test connection
            result = self.db_session.execute(sa.text("SELECT 1")).fetchone()
            if result[0] != 1:
                raise RuntimeError("Database connection test failed")
            
            # Load statistics
            await self._load_statistics()
            
            self.logger.info("✅ SQLite database initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ SQLite initialization failed: {e}")
            return False
    
    
    async def _init_chromadb(self):
        """Inizializza ChromaDB per semantic search"""
        try:
            if not CHROMADB_AVAILABLE:
                self.logger.warning("⚠️ ChromaDB not available, skipping")
                self.config.semantic_search_enabled = False
                return
            
            self.logger.info("🧠 Initializing ChromaDB for semantic search...")
            
            # Create client
            self.chroma_client = chromadb.PersistentClient(
                path=self.config.chromadb_path
            )
            
            # Get or create collection
            self.chroma_collection = self.chroma_client.get_or_create_collection(
                name="jarvis_conversations",
                metadata={"description": "Jarvis conversation embeddings"}
            )
            
            self.logger.info("✅ ChromaDB initialized")
            
        except Exception as e:
            self.logger.warning(f"⚠️ ChromaDB initialization failed, disabling: {e}")
            self.config.semantic_search_enabled = False
    
    
    async def _load_cache(self):
        """Carica cache da database per performance"""
        try:
            # Load recent conversations in cache
            recent_conversations = self.db_session.query(Conversation)\
                .filter(Conversation.timestamp > datetime.utcnow() - timedelta(hours=24))\
                .limit(50).all()
            
            for conv in recent_conversations:
                try:
                    cache_key = f"conv_{conv.id}"
                    metadata = json.loads(conv.metadata) if conv.metadata and isinstance(conv.metadata, str) else {}
                    self._set_cache(cache_key, {
                        "user_input": conv.user_input,
                        "ai_response": conv.ai_response,
                        "timestamp": conv.timestamp.isoformat(),
                        "metadata": metadata
                   })
                except Exception as e:
                    self.logger.warning(f"Skipping cache entry {conv.id}: {e}")
            
        except Exception as e:
            self.logger.warning(f"Cache loading failed: {e}")
            
            
        
    
    
    async def _load_statistics(self):
        """Carica statistiche dal database"""
        try:
            # Count total conversations
            total_convs = self.db_session.query(Conversation).count()
            self.stats["total_conversations"] = total_convs
            
            self.logger.info(f"📊 Loaded statistics: {total_convs} total conversations")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Statistics loading failed: {e}")


    # ================================
    # CONVERSATION MANAGEMENT
    # ================================
    
    async def add_conversation_turn(self, user_input: str, ai_response: str, 
                                  user_id: str = "default_user", 
                                  session_id: Optional[str] = None,
                                  metadata: Optional[Dict[str, Any]] = None,
                                  processing_time: Optional[float] = None,
                                  tokens_used: Optional[int] = None) -> int:
        """Aggiunge nuovo turn di conversazione"""
        try:
            # Generate session_id if not provided
            if not session_id:
                session_id = self._generate_session_id(user_id)
            
            # Create conversation entry
            conversation = Conversation(
                user_id=user_id,
                user_input=user_input,
                ai_response=ai_response,
                session_id=session_id,
                metadata=json.dumps(metadata) if metadata else None,
                processing_time=processing_time,
                tokens_used=tokens_used
            )
            
            # Save to database
            self.db_session.add(conversation)
            self.db_session.commit()
            
            conversation_id = conversation.id
            
            # Add to cache
            cache_key = f"conv_{conversation_id}"
            self._set_cache(cache_key, {
                "user_input": user_input,
                "ai_response": ai_response,
                "timestamp": conversation.timestamp.isoformat(),
                "metadata": metadata or {}
            })
            
            # Add to semantic search if enabled
            if self.config.semantic_search_enabled and self.chroma_collection:
                await self._add_to_semantic_search(conversation_id, user_input, ai_response)
            
            # Update statistics
            self.stats["total_conversations"] += 1
            
            self.logger.debug(f"💬 Conversation turn saved: ID {conversation_id}")
            
            return conversation_id
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save conversation: {e}")
            self.db_session.rollback()
            raise
    
    
    async def get_conversation_history(self, user_id: str = "default_user", 
                                     limit: int = 10,
                                     session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Ottiene cronologia conversazioni"""
        try:
            # Build query
            query = self.db_session.query(Conversation)\
                .filter(Conversation.user_id == user_id)
            
            if session_id:
                query = query.filter(Conversation.session_id == session_id)
            
            conversations = query.order_by(Conversation.timestamp.desc())\
                .limit(limit).all()
            
            # Convert to dict format
            history = []
            for conv in conversations:
                history.append({
                    "id": conv.id,
                    "user_input": conv.user_input,
                    "ai_response": conv.ai_response,
                    "timestamp": conv.timestamp.isoformat(),
                    "session_id": conv.session_id,
                    "metadata": json.loads(conv.metadata) if conv.metadata else {},
                    "processing_time": conv.processing_time,
                    "tokens_used": conv.tokens_used
                })
            
            # Reverse per avere ordine cronologico
            history.reverse()
            
            self.logger.debug(f"📚 Retrieved {len(history)} conversation turns")
            return history
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get conversation history: {e}")
            return []
    
    
    async def get_context_for_llm(self, user_id: str = "default_user", 
                                limit: int = 5) -> List[Dict[str, str]]:
        """Ottiene context conversation per LLM"""
        try:
            history = await self.get_conversation_history(user_id, limit)
            
            # Format per LLM (alternating user/assistant)
            context = []
            for turn in history:
                context.append({
                    "role": "user",
                    "content": turn["user_input"]
                })
                context.append({
                    "role": "assistant", 
                    "content": turn["ai_response"]
                })
            
            self.logger.debug(f"🧠 Generated LLM context: {len(context)} messages")
            return context
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get LLM context: {e}")
            return []
    
    
    def _generate_session_id(self, user_id: str) -> str:
        """Genera session ID unico"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        hash_input = f"{user_id}_{timestamp}_{time.time()}"
        session_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        return f"session_{timestamp}_{session_hash}"


    # ================================
    # SEMANTIC SEARCH
    # ================================
    
    async def _add_to_semantic_search(self, conversation_id: int, 
                                    user_input: str, ai_response: str):
        """Aggiunge conversazione a semantic search"""
        try:
            if not self.chroma_collection:
                return
            
            # Combine user input and AI response for embedding
            combined_text = f"User: {user_input}\nAssistant: {ai_response}"
            
            # Add to ChromaDB
            self.chroma_collection.add(
                documents=[combined_text],
                ids=[str(conversation_id)],
                metadatas=[{
                    "conversation_id": conversation_id,
                    "user_input": user_input,
                    "ai_response": ai_response,
                    "timestamp": datetime.utcnow().isoformat()
                }]
            )
            
            self.logger.debug(f"🔍 Added conversation {conversation_id} to semantic search")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to add to semantic search: {e}")
    
    
    async def search_conversations(self, query: str, 
                                 user_id: str = "default_user",
                                 limit: int = 5) -> List[Dict[str, Any]]:
        """Cerca conversazioni usando semantic search"""
        try:
            results = []
            
            # ChromaDB semantic search se disponibile
            if self.config.semantic_search_enabled and self.chroma_collection:
                chroma_results = self.chroma_collection.query(
                    query_texts=[query],
                    n_results=limit
                )
                
                if chroma_results['documents'] and chroma_results['documents'][0]:
                    for i, doc in enumerate(chroma_results['documents'][0]):
                        metadata = chroma_results['metadatas'][0][i]
                        distance = chroma_results['distances'][0][i] if chroma_results.get('distances') else 0
                        
                        results.append({
                            "conversation_id": metadata["conversation_id"],
                            "user_input": metadata["user_input"],
                            "ai_response": metadata["ai_response"],
                            "timestamp": metadata["timestamp"],
                            "similarity_score": 1.0 - distance,  # Convert distance to similarity
                            "search_type": "semantic"
                        })
                
                self.stats["semantic_searches"] += 1
            
            # Fallback: SQL text search
            if not results:
                sql_results = self.db_session.query(Conversation)\
                    .filter(Conversation.user_id == user_id)\
                    .filter(
                        sa.or_(
                            Conversation.user_input.contains(query),
                            Conversation.ai_response.contains(query)
                        )
                    )\
                    .order_by(Conversation.timestamp.desc())\
                    .limit(limit).all()
                
                for conv in sql_results:
                    results.append({
                        "conversation_id": conv.id,
                        "user_input": conv.user_input,
                        "ai_response": conv.ai_response,
                        "timestamp": conv.timestamp.isoformat(),
                        "similarity_score": 0.8,  # Fixed score per SQL search
                        "search_type": "text"
                    })
            
            self.logger.info(f"🔍 Search for '{query}' returned {len(results)} results")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Search failed: {e}")
            return []


    # ================================
    # USER PREFERENCES
    # ================================
    
    async def set_user_preference(self, user_id: str, key: str, value: Any) -> bool:
        """Imposta preferenza utente"""
        try:
            # Check if preference exists
            existing = self.db_session.query(UserPreference)\
                .filter(UserPreference.user_id == user_id)\
                .filter(UserPreference.key == key).first()
            
            if existing:
                existing.value = json.dumps(value) if not isinstance(value, str) else value
                existing.updated_at = datetime.utcnow()
            else:
                preference = UserPreference(
                    user_id=user_id,
                    key=key,
                    value=json.dumps(value) if not isinstance(value, str) else value
                )
                self.db_session.add(preference)
            
            self.db_session.commit()
            
            # Update cache
            cache_key = f"pref_{user_id}_{key}"
            self._set_cache(cache_key, value)
            
            self.logger.debug(f"⚙️ User preference set: {user_id}.{key} = {value}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to set user preference: {e}")
            self.db_session.rollback()
            return False
    
    
    async def get_user_preference(self, user_id: str, key: str, default: Any = None) -> Any:
        """Ottiene preferenza utente"""
        try:
            # Check cache first
            cache_key = f"pref_{user_id}_{key}"
            cached_value = self._get_cache(cache_key)
            if cached_value is not None:
                return cached_value
            
            # Query database
            preference = self.db_session.query(UserPreference)\
                .filter(UserPreference.user_id == user_id)\
                .filter(UserPreference.key == key).first()
            
            if preference:
                try:
                    value = json.loads(preference.value)
                except (json.JSONDecodeError, TypeError):
                    value = preference.value
                
                # Cache result
                self._set_cache(cache_key, value)
                return value
            
            return default
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get user preference: {e}")
            return default
    
    
    async def get_all_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Ottiene tutte le preferenze utente"""
        try:
            preferences = self.db_session.query(UserPreference)\
                .filter(UserPreference.user_id == user_id).all()
            
            result = {}
            for pref in preferences:
                try:
                    value = json.loads(pref.value)
                except (json.JSONDecodeError, TypeError):
                    value = pref.value
                result[pref.key] = value
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get user preferences: {e}")
            return {}


    # ================================
    # SYSTEM MEMORY
    # ================================
    
    async def set_system_memory(self, key: str, value: Any, 
                              expires_hours: Optional[int] = None) -> bool:
        """Imposta memoria sistema"""
        try:
            # Calculate expiration
            expires_at = None
            if expires_hours:
                expires_at = datetime.utcnow() + timedelta(hours=expires_hours)
            
            # Determine value type and serialize
            if isinstance(value, (dict, list)):
                value_str = json.dumps(value)
                value_type = "json"
            else:
                value_str = str(value)
                value_type = "string"
            
            # Check if exists
            existing = self.db_session.query(SystemMemory)\
                .filter(SystemMemory.key == key).first()
            
            if existing:
                existing.value = value_str
                existing.type = value_type
                existing.expires_at = expires_at
                existing.updated_at = datetime.utcnow()
            else:
                memory = SystemMemory(
                    key=key,
                    value=value_str,
                    type=value_type,
                    expires_at=expires_at
                )
                self.db_session.add(memory)
            
            self.db_session.commit()
            
            # Update cache
            self._set_cache(f"sys_{key}", value)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to set system memory: {e}")
            self.db_session.rollback()
            return False
    
    
    async def get_system_memory(self, key: str, default: Any = None) -> Any:
        """Ottiene memoria sistema"""
        try:
            # Check cache
            cache_key = f"sys_{key}"
            cached_value = self._get_cache(cache_key)
            if cached_value is not None:
                return cached_value
            
            # Query database
            memory = self.db_session.query(SystemMemory)\
                .filter(SystemMemory.key == key).first()
            
            if memory:
                # Check expiration
                if memory.expires_at and memory.expires_at < datetime.utcnow():
                    # Delete expired entry
                    self.db_session.delete(memory)
                    self.db_session.commit()
                    return default
                
                # Deserialize value
                if memory.type == "json":
                    value = json.loads(memory.value)
                else:
                    value = memory.value
                
                # Cache result
                self._set_cache(cache_key, value)
                return value
            
            return default
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get system memory: {e}")
            return default


    # ================================
    # CACHE MANAGEMENT
    # ================================
    
    def _set_cache(self, key: str, value: Any, ttl: Optional[int] = None):
        """Imposta valore in cache"""
        with self._cache_lock:
            self._cache[key] = value
            self._cache_timestamps[key] = time.time()
            
            # Cleanup old entries se necessario
            if len(self._cache) > self.config.max_cache_size:
                self._cleanup_cache()
    
    
    def _get_cache(self, key: str) -> Any:
        """Ottiene valore da cache"""
        with self._cache_lock:
            if key not in self._cache:
                self.stats["cache_misses"] += 1
                return None
            
            # Check TTL
            age = time.time() - self._cache_timestamps[key]
            if age > self.config.cache_ttl:
                del self._cache[key]
                del self._cache_timestamps[key]
                self.stats["cache_misses"] += 1
                return None
            
            self.stats["cache_hits"] += 1
            return self._cache[key]
    
    
    def _cleanup_cache(self):
        """Pulisce cache rimuovendo entries vecchie"""
        current_time = time.time()
        keys_to_remove = []
        
        # Find expired keys
        for key, timestamp in self._cache_timestamps.items():
            if current_time - timestamp > self.config.cache_ttl:
                keys_to_remove.append(key)
        
        # Remove expired keys
        for key in keys_to_remove:
            del self._cache[key]
            del self._cache_timestamps[key]
        
        # If still too many, remove oldest
        if len(self._cache) > self.config.max_cache_size:
            sorted_keys = sorted(
                self._cache_timestamps.items(),
                key=lambda x: x[1]
            )
            
            # Remove oldest 25%
            remove_count = len(sorted_keys) // 4
            for key, _ in sorted_keys[:remove_count]:
                del self._cache[key]
                del self._cache_timestamps[key]


    # ================================
    # BACKUP SYSTEM
    # ================================
    
    async def _start_backup_system(self):
        """Avvia sistema backup automatico"""
        try:
            self._backup_task = asyncio.create_task(self._backup_loop())
            self.logger.info("💾 Backup system started")
            
        except Exception as e:
            self.logger.error(f"❌ Backup system startup failed: {e}")
    
    
    async def _backup_loop(self):
        """Loop backup automatico"""
        while not self._stop_backup.is_set():
            try:
                await asyncio.sleep(self.config.backup_interval_hours * 3600)
                
                if not self._stop_backup.is_set():
                    await self.create_backup()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"❌ Backup loop error: {e}")
                await asyncio.sleep(3600)  # Retry in 1 hour
    
    
    async def create_backup(self) -> bool:
        """Crea backup del database"""
        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            backup_path = f"data/backups/jarvis_memory_backup_{timestamp}.db"
            
            # SQLite backup
            import shutil
            shutil.copy2(self.config.sqlite_path, backup_path)
            
            # Cleanup old backups
            await self._cleanup_old_backups()
            
            self.stats["backups_created"] += 1
            self.logger.info(f"💾 Backup created: {backup_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Backup creation failed: {e}")
            return False
    
    
    async def _cleanup_old_backups(self):
        """Rimuove backup vecchi mantenendo solo i più recenti"""
        try:
            backup_dir = Path("data/backups")
            backup_files = list(backup_dir.glob("jarvis_memory_backup_*.db"))
            
            if len(backup_files) > self.config.max_backups:
                # Sort per data (dal nome file)
                backup_files.sort(key=lambda x: x.stem)
                
                # Rimuovi i più vecchi
                files_to_remove = backup_files[:-self.config.max_backups]
                for file_path in files_to_remove:
                    file_path.unlink()
                    self.logger.debug(f"🗑️ Removed old backup: {file_path.name}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Backup cleanup failed: {e}")


    # ================================
    # MAINTENANCE
    # ================================
    
    async def _run_maintenance(self):
        """Esegue manutenzione database"""
        try:
            self.logger.info("🔧 Running database maintenance...")
            
            # Clean expired system memory
            expired = self.db_session.query(SystemMemory)\
                .filter(SystemMemory.expires_at < datetime.utcnow()).all()
            
            for item in expired:
                self.db_session.delete(item)
            
            if expired:
                self.db_session.commit()
                self.logger.info(f"🧹 Cleaned {len(expired)} expired system memories")
            
            # Clean old conversations se necessario
            if self.config.conversation_cleanup_days > 0:
                cutoff_date = datetime.utcnow() - timedelta(days=self.config.conversation_cleanup_days)
                old_conversations = self.db_session.query(Conversation)\
                    .filter(Conversation.timestamp < cutoff_date).all()
                
                if old_conversations:
                    for conv in old_conversations:
                        self.db_session.delete(conv)
                    
                    self.db_session.commit()
                    self.logger.info(f"🧹 Cleaned {len(old_conversations)} old conversations")
            
            # SQLite VACUUM per ottimizzare
            self.db_session.execute(sa.text("VACUUM"))
            self.db_session.commit()
            
            self.logger.info("✅ Database maintenance completed")
            
        except Exception as e:
            self.logger.error(f"❌ Database maintenance failed: {e}")
            self.db_session.rollback()


    # ================================
    # CONVERSATION SUMMARIES
    # ================================
    
    async def save_conversation_summary(self, summary: str, 
                                      session_id: Optional[str] = None,
                                      user_id: str = "default_user") -> bool:
        """Salva riassunto conversazione"""
        try:
            summary_key = f"summary_{session_id}" if session_id else f"summary_{user_id}_{int(time.time())}"
            
            return await self.set_system_memory(
                key=summary_key,
                value={
                    "summary": summary,
                    "session_id": session_id,
                    "user_id": user_id,
                    "created_at": datetime.utcnow().isoformat()
                },
                expires_hours=24 * 30  # 30 giorni
            )
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save conversation summary: {e}")
            return False
    
    
    async def get_recent_summaries(self, user_id: str = "default_user", 
                                 limit: int = 5) -> List[Dict[str, Any]]:
        """Ottiene riassunti conversazioni recenti"""
        try:
            # Query summaries from system memory
            summaries_query = self.db_session.query(SystemMemory)\
                .filter(SystemMemory.key.like(f"summary_%"))\
                .filter(SystemMemory.value.contains(user_id))\
                .order_by(SystemMemory.created_at.desc())\
                .limit(limit).all()
            
            summaries = []
            for item in summaries_query:
                try:
                    value = json.loads(item.value)
                    if value.get("user_id") == user_id:
                        summaries.append(value)
                except (json.JSONDecodeError, KeyError):
                    continue
            
            return summaries
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get recent summaries: {e}")
            return []


    # ================================
    # STATUS & UTILITIES
    # ================================
    
    async def get_status(self) -> Dict[str, Any]:
        """Ottieni stato completo Memory Manager"""
        try:
            # Database info
            total_conversations = self.db_session.query(Conversation).count()
            total_preferences = self.db_session.query(UserPreference).count()
            total_system_memory = self.db_session.query(SystemMemory).count()
            
            # Database size
            db_path = Path(self.config.sqlite_path)
            db_size_mb = db_path.stat().st_size / (1024 * 1024) if db_path.exists() else 0
            
            return {
                "initialized": self.is_initialized,
                "database": {
                    "path": self.config.sqlite_path,
                    "size_mb": round(db_size_mb, 2),
                    "conversations": total_conversations,
                    "user_preferences": total_preferences,
                    "system_memory_entries": total_system_memory
                },
                "cache": {
                    "enabled": self.config.cache_enabled,
                    "size": len(self._cache),
                    "max_size": self.config.max_cache_size,
                    "hit_rate": (
                        self.stats["cache_hits"] / 
                        max(self.stats["cache_hits"] + self.stats["cache_misses"], 1) * 100
                    )
                },
                "semantic_search": {
                    "enabled": self.config.semantic_search_enabled,
                    "available": CHROMADB_AVAILABLE,
                    "searches_performed": self.stats["semantic_searches"]
                },
                "backup": {
                    "enabled": self.config.backup_enabled,
                    "interval_hours": self.config.backup_interval_hours,
                    "backups_created": self.stats["backups_created"]
                },
                "statistics": self.stats.copy()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get status: {e}")
            return {"error": str(e)}
    
    
    async def clear_all_data(self, confirm_key: str) -> bool:
        """Cancella tutti i dati (PERICOLOSO - richiede conferma)"""
        if confirm_key != "CLEAR_ALL_JARVIS_DATA":
            self.logger.warning("⚠️ Clear all data attempted with wrong confirm key")
            return False
        
        try:
            self.logger.warning("🚨 CLEARING ALL DATA - This cannot be undone!")
            
            # Clear database tables
            self.db_session.query(Conversation).delete()
            self.db_session.query(UserPreference).delete()
            self.db_session.query(SystemMemory).delete()
            self.db_session.commit()
            
            # Clear cache
            with self._cache_lock:
                self._cache.clear()
                self._cache_timestamps.clear()
            
            # Clear ChromaDB se disponibile
            if self.chroma_collection:
                self.chroma_client.delete_collection("jarvis_conversations")
                self.chroma_collection = self.chroma_client.create_collection(
                    name="jarvis_conversations"
                )
            
            # Reset statistics
            self.stats = {
                "total_conversations": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "semantic_searches": 0,
                "backups_created": 0
            }
            
            self.logger.warning("🗑️ All data cleared successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to clear all data: {e}")
            self.db_session.rollback()
            return False
    
    
    async def export_data(self, export_path: str) -> bool:
        """Esporta tutti i dati in formato JSON"""
        try:
            self.logger.info(f"📤 Exporting data to: {export_path}")
            
            # Export conversations
            conversations = self.db_session.query(Conversation).all()
            conversations_data = []
            for conv in conversations:
                conversations_data.append({
                    "id": conv.id,
                    "user_id": conv.user_id,
                    "user_input": conv.user_input,
                    "ai_response": conv.ai_response,
                    "timestamp": conv.timestamp.isoformat(),
                    "session_id": conv.session_id,
                    "metadata": json.loads(conv.metadata) if conv.metadata else {},
                    "processing_time": conv.processing_time,
                    "tokens_used": conv.tokens_used
                })
            
            # Export preferences
            preferences = self.db_session.query(UserPreference).all()
            preferences_data = []
            for pref in preferences:
                preferences_data.append({
                    "user_id": pref.user_id,
                    "key": pref.key,
                    "value": pref.value,
                    "updated_at": pref.updated_at.isoformat()
                })
            
            # Export system memory
            system_memory = self.db_session.query(SystemMemory).all()
            system_memory_data = []
            for mem in system_memory:
                system_memory_data.append({
                    "key": mem.key,
                    "value": mem.value,
                    "type": mem.type,
                    "expires_at": mem.expires_at.isoformat() if mem.expires_at else None,
                    "created_at": mem.created_at.isoformat(),
                    "updated_at": mem.updated_at.isoformat()
                })
            
            # Create export data
            export_data = {
                "export_info": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "jarvis_version": "1.0.0",
                    "total_conversations": len(conversations_data),
                    "total_preferences": len(preferences_data),
                    "total_system_memory": len(system_memory_data)
                },
                "conversations": conversations_data,
                "user_preferences": preferences_data,
                "system_memory": system_memory_data
            }
            
            # Write to file
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info("✅ Data export completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Data export failed: {e}")
            return False
    
    
    async def cleanup(self):
        """Cleanup risorse Memory Manager"""
        try:
            self.logger.info("🧹 Cleaning up Memory Manager...")
            
            # Stop backup system
            if self._backup_task and not self._backup_task.done():
                self._stop_backup.set()
                self._backup_task.cancel()
                try:
                    await self._backup_task
                except asyncio.CancelledError:
                    pass
            
            # Close database connections
            if self.db_session:
                self.db_session.close()
            
            if self.db_engine:
                self.db_engine.dispose()
            
            # Clear cache
            with self._cache_lock:
                self._cache.clear()
                self._cache_timestamps.clear()
            
            self.is_initialized = False
            self.logger.info("✅ Memory Manager cleaned up")
            
        except Exception as e:
            self.logger.error(f"❌ Cleanup error: {e}")


# ================================
# TESTING E ENTRY POINT
# ================================

async def test_memory_manager():
    """Test standalone Memory Manager"""
    print("🧪 Testing Memory Manager...")
    
    config = MemoryConfig()
    memory_manager = MemoryManager(config)
    
    try:
        # Inizializza
        if not await memory_manager.initialize():
            print("❌ Initialization failed")
            return
        
        # Test conversation
        conv_id = await memory_manager.add_conversation_turn(
            user_input="Ciao Jarvis, come va?",
            ai_response="Ciao! Tutto bene, grazie. Come posso aiutarti?"
        )
        print(f"💬 Conversation saved with ID: {conv_id}")
        
        # Test history
        history = await memory_manager.get_conversation_history(limit=5)
        print(f"📚 Retrieved {len(history)} conversation turns")
        
        # Test preferences
        await memory_manager.set_user_preference("test_user", "theme", "dark")
        theme = await memory_manager.get_user_preference("test_user", "theme")
        print(f"⚙️ User preference: theme = {theme}")
        
        # Test system memory
        await memory_manager.set_system_memory("last_startup", datetime.utcnow().isoformat())
        startup = await memory_manager.get_system_memory("last_startup")
        print(f"🧠 System memory: last_startup = {startup}")
        
        # Test search
        results = await memory_manager.search_conversations("ciao")
        print(f"🔍 Search results: {len(results)} found")
        
        # Status
        status = await memory_manager.get_status()
        print(f"📊 Status: {json.dumps(status, indent=2)}")
        
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted")
    finally:
        await memory_manager.cleanup()


if __name__ == "__main__":
    # Test standalone
    asyncio.run(test_memory_manager())