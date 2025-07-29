# 🤖 JARVIS AI ASSISTANT

Un assistente AI completamente locale ispirato al Jarvis di Iron Man, con controllo vocale e interfaccia futuristica.

## 🎯 CARATTERISTICHE

- **100% Locale**: Nessun dato trasmesso online
- **Voice-First**: Controllo primario vocale (Whisper + Piper)
- **AI Locale**: Mistral 7B via Ollama
- **Interfaccia Futuristica**: UI ispirata a Iron Man
- **Plugin System**: Architettura modulare ed estensibile

## 📋 REQUISITI SISTEMA

- **OS**: Windows 10/11, macOS 10.15+, Ubuntu 20.04+
- **Python**: 3.9-3.11
- **RAM**: 8GB minimo, 16GB consigliati
- **Storage**: 5GB liberi
- **Internet**: Solo per installazione iniziale

## 🚀 INSTALLAZIONE RAPIDA

### 1. Clone Repository
\\\ash
git clone [repository-url]
cd Jarvis_Assistant
\\\

### 2. Virtual Environment
\\\ash
# Windows
python -m venv jarvis_env
jarvis_env\Scripts\activate

# macOS/Linux
python3 -m venv jarvis_env
source jarvis_env/bin/activate
\\\

### 3. Installa Dependencies
\\\ash
pip install --upgrade pip
pip install -r requirements.txt
\\\

### 4. Installa Ollama + Mistral
\\\ash
# Scarica Ollama da https://ollama.ai
ollama pull mistral:7b
ollama serve
\\\

### 5. Avvia Sistema
\\\ash
cd core
python websocket_server.py
\\\

### 6. Apri Frontend
Apri browser su: \http://localhost:8765\

## 🎤 COMANDI VOCALI

- \"Ehi Jarvis, ciao"\ - Saluto
- \"Jarvis, che ore sono?"\ - Informazioni
- \"Jarvis, mostra le statistiche"\ - System status

## 📁 STRUTTURA PROGETTO

\\\
Jarvis_Assistant/
├── core/                   # Backend Python
├── frontend/              # UI Interface
├── data/                  # Database locale
├── config/                # Configurazioni
└── docs/                  # Documentazione
\\\

## 🔧 TECNOLOGIE

- **Voice**: Whisper (STT) + Piper (TTS)
- **AI**: Ollama + Mistral 7B
- **Database**: SQLite + ChromaDB
- **Frontend**: HTML/CSS/JS + WebSocket
- **Communication**: AsyncIO + WebSocket

## 📞 SUPPORTO

- **Logs**: \data/logs/\ per debugging
- **Config**: \config/\ per personalizzazioni
- **Docs**: \docs/\ per documentazione completa

## 🛡️ PRIVACY

- ✅ Nessun dato trasmesso online
- ✅ Processing solo in memoria
- ✅ Database completamente locale
- ✅ Nessuna API key richiesta

---

*Sviluppato seguendo i principi LOCAL-FIRST, VOICE-FIRST, PRIVACY-FIRST* 🎯
