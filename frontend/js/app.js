/**
 * JARVIS AI ASSISTANT - FRONTEND APPLICATION LOGIC CON STREAMING
 * 
 * Versione aggiornata con supporto completo per streaming chunks:
 * - Gestione ai_response_chunk in real-time
 * - Effetto typing progressivo come ADA
 * - Accumulo chunks durante streaming
 * - Visual feedback per streaming attivo
 * 
 * NUOVO: Streaming real-time invece di attesa 19s
 */

class JarvisApp {
    constructor() {
        // CONFIGURAZIONE
        this.config = {
            websocketUrl: 'ws://localhost:8765',
            reconnectInterval: 3000,
            particleCount: 300,
            updateInterval: 1000
        };

        // STATO APPLICAZIONE
        this.state = {
            connected: false,
            currentState: 'normal',
            voiceActive: false,
            systemMetrics: {
                cpu: 0,
                memory: 0,
                voiceStatus: 'READY',
                aiModel: 'MISTRAL-7B'
            },
            notifications: []
        };

        // STREAMING STATE - NUOVO
        this.streamingState = {
            isStreaming: false,
            currentMessage: "",
            currentMessageElement: null,
            chunksReceived: 0,
            streamStartTime: null
        };

        // ELEMENTI DOM
        this.elements = {};
        this.websocket = null;
        this.particlesCanvas = null;
        this.particlesCtx = null;
        this.particles = [];

        // INIZIALIZZAZIONE
        this.init();
    }

    /**
     * INIZIALIZZAZIONE PRINCIPALE
     */
    async init() {
        console.log('🚀 Initializing Jarvis Frontend with STREAMING support...');

        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.setup());
        } else {
            this.setup();
        }
    }

    /**
     * SETUP COMPLETO INTERFACCIA
     */
    async setup() {
        try {
            // 1. Inizializza elementi DOM
            this.setupElements();

            // 2. Setup particelle canvas
            this.setupParticles();

            // 3. Setup event listeners
            this.setupEventListeners();

            // 4. Connetti WebSocket
            await this.connectWebSocket();

            // 5. Avvia loops di aggiornamento
            this.startUpdateLoops();

            // 6. Nascondi loading overlay
            this.hideLoadingOverlay();

            console.log('✅ Jarvis Frontend with STREAMING initialized successfully');
            this.showNotification('J.A.R.V.I.S Interface Online - STREAMING Mode', 'success');

        } catch (error) {
            console.error('❌ Error initializing Jarvis Frontend:', error);
            this.showNotification('Initialization Error: ' + error.message, 'error');
        }
    }

    /**
     * SETUP ELEMENTI DOM
     */
    setupElements() {
        this.elements = {
            // Status elements
            cpuUsage: document.getElementById('cpu-usage'),
            memoryUsage: document.getElementById('memory-usage'),
            voiceStatus: document.getElementById('voice-status'),
            aiModel: document.getElementById('ai-model'),
            backendStatus: document.getElementById('backend-status'),
            wsStatus: document.getElementById('ws-status'),

            // Control buttons
            voiceToggle: document.getElementById('voice-toggle'),
            chatBtn: document.getElementById('chat-btn'),
            settingsBtn: document.getElementById('settings-btn'),
            minimizeBtn: document.getElementById('minimize-btn'),
            fullscreenBtn: document.getElementById('fullscreen-btn'),

            // Chat elements
            chatPanel: document.getElementById('chat-panel'),
            chatClose: document.getElementById('chat-close'),
            chatInput: document.getElementById('chat-input'),
            chatSend: document.getElementById('chat-send'),
            chatMessages: document.getElementById('chat-messages'),

            // Progress squares
            cpuProgress: document.getElementById('cpu-progress'),
            memoryProgress: document.getElementById('memory-progress'),
            networkProgress: document.getElementById('network-progress'),
            aiProgress: document.getElementById('ai-progress'),

            // Loading
            loadingOverlay: document.getElementById('loading-overlay'),
            loadingProgress: document.getElementById('loading-progress'),

            // Notifications
            notifications: document.getElementById('notifications'),

            // Voice activity
            voiceActivity: document.querySelector('.voice-activity'),
            waveBars: document.querySelectorAll('.wave-bar')
        };

        console.log('📋 DOM elements initialized with streaming support');
    }

    /**
     * SETUP PARTICELLE CANVAS
     */
    setupParticles() {
        this.particlesCanvas = document.getElementById('particles-canvas');
        if (!this.particlesCanvas) {
            console.warn('⚠️ Particles canvas not found');
            return;
        }

        this.particlesCtx = this.particlesCanvas.getContext('2d');
        this.resizeCanvas();
        window.addEventListener('resize', () => this.resizeCanvas());
        this.createParticles();
        this.animateParticles();

        console.log('✨ Particles system initialized');
    }

    /**
     * RESIZE CANVAS
     */
    resizeCanvas() {
        if (this.particlesCanvas) {
            this.particlesCanvas.width = window.innerWidth;
            this.particlesCanvas.height = window.innerHeight;
        }
    }

    /**
     * CREA PARTICELLE
     */
    createParticles() {
        this.particles = [];

        for (let i = 0; i < this.config.particleCount; i++) {
            this.particles.push({
                x: Math.random() * this.particlesCanvas.width,
                y: Math.random() * this.particlesCanvas.height,
                vx: (Math.random() - 0.5) * 0.8,
                vy: (Math.random() - 0.5) * 0.8,
                size: Math.random() * 3 + 0.5,
                opacity: Math.random() * 0.7 + 0.3,
                pulsePhase: Math.random() * Math.PI * 2,
                type: Math.random() > 0.7 ? 'neural' : 'normal'
            });
        }
    }

    /**
     * ANIMAZIONE PARTICELLE
     */
    animateParticles() {
        if (!this.particlesCtx || !this.particlesCanvas) return;

        this.particlesCtx.clearRect(0, 0, this.particlesCanvas.width, this.particlesCanvas.height);

        this.particles.forEach((particle, index) => {
            particle.x += particle.vx;
            particle.y += particle.vy;

            if (particle.x < 0 || particle.x > this.particlesCanvas.width) {
                particle.vx *= -1;
            }
            if (particle.y < 0 || particle.y > this.particlesCanvas.height) {
                particle.vy *= -1;
            }

            particle.x = Math.max(0, Math.min(this.particlesCanvas.width, particle.x));
            particle.y = Math.max(0, Math.min(this.particlesCanvas.height, particle.y));

            particle.pulsePhase += 0.02;
            const pulse = Math.sin(particle.pulsePhase) * 0.3 + 0.7;

            this.particlesCtx.beginPath();
            this.particlesCtx.arc(particle.x, particle.y, particle.size * pulse, 0, Math.PI * 2);
            this.particlesCtx.fillStyle = `rgba(0, 212, 255, ${particle.opacity * pulse})`;
            this.particlesCtx.fill();

            if (index % 5 === 0) {
                this.particles.slice(index + 1, index + 21).forEach(other => {
                    const dx = particle.x - other.x;
                    const dy = particle.y - other.y;
                    const distance = Math.sqrt(dx * dx + dy * dy);

                    if (distance < 100) {
                        this.particlesCtx.beginPath();
                        this.particlesCtx.moveTo(particle.x, particle.y);
                        this.particlesCtx.lineTo(other.x, other.y);
                        this.particlesCtx.strokeStyle = `rgba(0, 212, 255, ${0.1 * (1 - distance / 100)})`;
                        this.particlesCtx.lineWidth = 0.5;
                        this.particlesCtx.stroke();
                    }
                });
            }
        });

        requestAnimationFrame(() => this.animateParticles());
    }

    /**
     * SETUP EVENT LISTENERS
     */
    setupEventListeners() {
        // Control buttons
        this.elements.voiceToggle?.addEventListener('click', () => this.toggleVoice());
        this.elements.settingsBtn?.addEventListener('click', () => this.openSettings());
        this.elements.minimizeBtn?.addEventListener('click', () => this.minimizeWindow());
        this.elements.fullscreenBtn?.addEventListener('click', () => this.toggleFullscreen());

        // Chat functionality
        this.setupChatEventListeners();

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => this.handleKeyboard(e));

        // Mouse cursor tracking
        document.addEventListener('mousemove', (e) => this.updateCursor(e));

        console.log('🎛️ Event listeners setup complete');
    }

    /**
     * SETUP CHAT EVENT LISTENERS
     */
    setupChatEventListeners() {
        console.log('💬 Setting up chat event listeners with STREAMING support...');

        if (!this.elements.chatBtn || !this.elements.chatPanel) {
            console.error('❌ Chat elements not found!');
            return;
        }

        // Apri/chiudi chat panel
        this.elements.chatBtn.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            console.log('🔘 Chat button clicked');

            const isOpen = this.elements.chatPanel.style.right === '20px';
            this.elements.chatPanel.style.right = isOpen ? '-400px' : '20px';
            this.elements.chatPanel.style.display = 'block';

            console.log(`💬 Chat panel ${isOpen ? 'closed' : 'opened'}`);
        });

        // Chiudi chat
        if (this.elements.chatClose) {
            this.elements.chatClose.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                console.log('❌ Chat close clicked');
                this.elements.chatPanel.style.right = '-400px';
            });
        }

        // Funzione invio messaggio
        const sendMessage = () => {
            if (!this.elements.chatInput) {
                console.error('❌ Chat input not found!');
                return;
            }

            const message = this.elements.chatInput.value.trim();
            console.log('📤 Attempting to send message:', message);

            if (message) {
                // Aggiungi messaggio utente alla chat
                this.addChatMessage('USER', message);

                // Invia al backend
                this.sendWebSocketMessage('text_command', { text: message });

                // Reset input
                this.elements.chatInput.value = '';
                console.log('✅ Message sent and input cleared');
            } else {
                console.log('⚠️ Empty message, not sending');
            }
        };

        // Event listener bottone invio
        if (this.elements.chatSend) {
            this.elements.chatSend.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                console.log('🔘 Send button clicked');
                sendMessage();
            });
        }

        // Event listener Enter key
        if (this.elements.chatInput) {
            this.elements.chatInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    console.log('⌨️ Enter key pressed');
                    sendMessage();
                }
            });
        }

        console.log('✅ Chat event listeners setup complete with streaming');
    }

    /**
     * CONNESSIONE WEBSOCKET
     */
    async connectWebSocket() {
        try {
            console.log('🌐 Connecting to WebSocket...', this.config.websocketUrl);

            this.websocket = new WebSocket(this.config.websocketUrl);

            this.websocket.onopen = () => {
                console.log('✅ WebSocket connected');
                this.state.connected = true;
                this.updateConnectionStatus(true);
                this.showNotification('Connected to Jarvis Core - STREAMING Active', 'success');
            };

            this.websocket.onmessage = (event) => {
                this.handleWebSocketMessage(event.data);
            };

            this.websocket.onclose = (event) => {
                console.log('🔌 WebSocket disconnected:', event.code, event.reason);
                this.state.connected = false;
                this.updateConnectionStatus(false);

                if (event.code !== 1000 && event.code !== 1001) {
                    this.showNotification('Connection lost. Reconnecting...', 'warning');
                    setTimeout(() => this.connectWebSocket(), this.config.reconnectInterval);
                } else {
                    console.log('💫 Normal WebSocket close, not reconnecting automatically');
                }
            };

            this.websocket.onerror = (error) => {
                console.error('❌ WebSocket error:', error);
                this.showNotification('Connection error', 'error');
            };

        } catch (error) {
            console.error('❌ Failed to connect WebSocket:', error);
            this.showNotification('Failed to connect to backend', 'error');
            setTimeout(() => this.connectWebSocket(), this.config.reconnectInterval);
        }
    }

    /**
     * GESTIONE MESSAGGI WEBSOCKET - CON STREAMING SUPPORT
     */
    handleWebSocketMessage(data) {
        console.log('📨 RAW MESSAGE RECEIVED:', data);

        try {
            const message = typeof data === 'string' ? JSON.parse(data) : data;
            console.log('📋 PARSED MESSAGE:', message);

            switch (message.type) {
                // ✅ NUOVO - GESTIONE STREAMING CHUNKS
                case 'ai_response_chunk':
                    console.log('🔥 STREAMING CHUNK RECEIVED:', message);
                    this.handleStreamingChunk(message);
                    break;

                // Gestione echo response
                case 'echo_response':
                    console.log('🔄 ECHO RICEVUTO:', message);
                    if (message.original_message?.type === 'text_command') {
                        const originalText = message.original_message?.text || 'messaggio vuoto';
                        console.log('💬 Echo di text_command processato correttamente');
                    }
                    break;

                // Gestione connessione stabilita
                case 'connection_established':
                    console.log('🔗 CONNESSIONE STABILITA:', message.message);
                    this.addChatMessage('SYSTEM', `Connesso: ${message.message}`);
                    break;

                case 'state_change':
                    this.setState(message.state);
                    break;

                case 'voice_activity':
                    this.updateVoiceActivity(message.active);
                    break;

                case 'system_metrics':
                    this.updateSystemMetrics(message.data);
                    break;

                // ✅ RISPOSTA AI FINALE - STREAMING COMPLETATO
                case 'text_command_response':
                    console.log('🏁 FINAL RESPONSE RECEIVED:', message);
                    this.handleFinalResponse(message);
                    break;

                case 'ai_response':
                    if (message.data && message.data.text) {
                        this.addChatMessage('JARVIS', message.data.text);
                        this.handleJarvisResponse(message.data);
                    } else if (message.text) {
                        this.addChatMessage('JARVIS', message.text);
                    }
                    break;

                case 'notification':
                    this.showNotification(message.text || message.message, message.level || 'info');
                    break;

                case 'wake_word_detected':
                    this.setState('processing');
                    this.showNotification('Wake word detected!', 'info');
                    break;

                case 'initial_state':
                    console.log('📋 Received initial state:', message.data);
                    if (message.data) {
                        this.setState(message.data.current_state || 'normal');
                        this.updateVoiceActivity(message.data.voice_active || false);
                    }
                    break;

                case 'error':
                    console.error('Backend error:', message);
                    this.showNotification(`Error: ${message.message}`, 'error');
                    this.addChatMessage('SYSTEM', `❌ ${message.message}`, 'error');
                    break;

                case 'pong':
                    console.log('🏓 Pong received');
                    break;

                default:
                    console.log('❓ Unknown message type:', message.type, message);
                    break;
            }
        } catch (error) {
            console.error('❌ Error parsing WebSocket message:', error, data);
        }
    }

    /**
     * ✅ NUOVO - GESTIONE STREAMING CHUNKS
     */
    handleStreamingChunk(message) {
        try {
            const chunk = message.chunk || '';
            const chunkNumber = message.chunk_number || 0;
            const isFinal = message.is_final || false;

            console.log(`📦 Processing chunk #${chunkNumber}: "${chunk}" (final: ${isFinal})`);

            // Se è il primo chunk, inizia nuovo messaggio streaming
            if (chunkNumber === 1 || !this.streamingState.isStreaming) {
                this.startStreamingMessage();
            }

            // Accumula chunk nel messaggio corrente
            if (chunk) {
                this.streamingState.currentMessage += chunk;
                this.streamingState.chunksReceived++;

                // Aggiorna il messaggio nel DOM in real-time
                this.updateStreamingMessage(this.streamingState.currentMessage);
            }

            // Se è il chunk finale, completa il streaming
            if (isFinal) {
                this.completeStreamingMessage();
            }

        } catch (error) {
            console.error('❌ Error handling streaming chunk:', error);
        }
    }

    /**
     * ✅ NUOVO - INIZIA MESSAGGIO STREAMING
     */
    startStreamingMessage() {
        console.log('🚀 Starting streaming message...');

        this.streamingState.isStreaming = true;
        this.streamingState.currentMessage = "";
        this.streamingState.chunksReceived = 0;
        this.streamingState.streamStartTime = Date.now();

        // Crea elemento messaggio per streaming
        const messageDiv = document.createElement('div');
        messageDiv.className = 'chat-message jarvis-message streaming';

        const backgroundColor = 'rgba(0, 255, 127, 0.1)';
        const borderColor = '#00ff7f';

        messageDiv.style.cssText = `
            margin-bottom: 15px; 
            padding: 12px 15px; 
            border-radius: 12px; 
            background: ${backgroundColor}; 
            border-left: 3px solid ${borderColor};
            animation: fadeInUp 0.3s ease-out;
        `;

        const timestamp = new Date().toLocaleTimeString('it-IT', {
            hour: '2-digit',
            minute: '2-digit'
        });

        messageDiv.innerHTML = `
            <div style="font-weight: bold; font-size: 11px; margin-bottom: 5px; color: ${borderColor}; text-transform: uppercase;">
                JARVIS - ${timestamp} - STREAMING...
            </div>
            <div class="streaming-content" style="line-height: 1.5; color: #ffffff; word-wrap: break-word;">
                <span class="typing-cursor">|</span>
            </div>
        `;

        this.elements.chatMessages.appendChild(messageDiv);
        this.streamingState.currentMessageElement = messageDiv;

        // Scroll to bottom
        this.elements.chatMessages.scrollTop = this.elements.chatMessages.scrollHeight;
    }

    /**
     * ✅ FIXED - AGGIORNA MESSAGGIO STREAMING
     */
    updateStreamingMessage(text) {
        if (!this.streamingState.currentMessageElement) {
            console.warn('⚠️ No streaming message element found');
            return;
        }

        const contentDiv = this.streamingState.currentMessageElement.querySelector('.streaming-content');
        if (contentDiv) {
            // METODO DIRETTO - Force update DOM
            contentDiv.innerHTML = '';
            const textNode = document.createTextNode(text);
            const cursorSpan = document.createElement('span');
            cursorSpan.className = 'typing-cursor';
            cursorSpan.textContent = '|';
            cursorSpan.style.cssText = 'animation: blink 1s infinite; color: #00d4ff;';

            contentDiv.appendChild(textNode);
            contentDiv.appendChild(cursorSpan);

            console.log(`📝 Updated streaming text: "${text.substring(0, 20)}..." (${text.length} chars)`);

            // Force scroll to bottom
            setTimeout(() => {
                this.elements.chatMessages.scrollTop = this.elements.chatMessages.scrollHeight;
            }, 0);
        } else {
            console.error('❌ Streaming content div not found');
        }
    }

    /**
     * ✅ FIXED - COMPLETA MESSAGGIO STREAMING
     */
    completeStreamingMessage() {
        console.log('🏁 Completing streaming message...');

        if (this.streamingState.currentMessageElement) {
            // Rimuovi cursor e mostra testo finale
            const contentDiv = this.streamingState.currentMessageElement.querySelector('.streaming-content');
            if (contentDiv) {
                // FORCE CLEAN UPDATE
                contentDiv.innerHTML = '';
                const finalTextNode = document.createTextNode(this.streamingState.currentMessage);
                contentDiv.appendChild(finalTextNode);

                console.log(`✅ Final text set: "${this.streamingState.currentMessage.substring(0, 50)}..."`);
            }

            // Aggiorna header con statistiche
            const headerDiv = this.streamingState.currentMessageElement.querySelector('div');
            if (headerDiv) {
                const streamDuration = ((Date.now() - this.streamingState.streamStartTime) / 1000).toFixed(1);
                const currentText = headerDiv.innerHTML;
                const newText = currentText.replace('STREAMING...', `COMPLETED (${streamDuration}s, ${this.streamingState.chunksReceived} chunks)`);
                headerDiv.innerHTML = newText;
            }

            // Rimuovi classe streaming
            this.streamingState.currentMessageElement.classList.remove('streaming');
        }

        // Reset streaming state
        this.streamingState.isStreaming = false;
        this.streamingState.currentMessage = "";
        this.streamingState.currentMessageElement = null;
        this.streamingState.chunksReceived = 0;

        console.log('✅ Streaming message completed and cleaned up');
    }

    /**
     * ✅ GESTIONE RISPOSTA FINALE
     */
    handleFinalResponse(message) {
        console.log('🏁 Handling final response:', message);

        // Se non è in streaming, aggiungi messaggio normale
        if (!this.streamingState.isStreaming) {
            this.addChatMessage('JARVIS', message.text);
        }
        // Se è in streaming, il messaggio è già stato gestito dai chunks
    }

    /**
     * AGGIUNGI MESSAGGIO ALLA CHAT
     */
    addChatMessage(sender, text, type = 'normal') {
        if (!this.elements.chatMessages) return;

        // Se è in streaming, non aggiungere messaggi duplicati
        if (this.streamingState.isStreaming && sender === 'JARVIS') {
            return;
        }

        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${sender.toLowerCase()}-message`;

        const isUser = sender === 'USER';
        const backgroundColor = isUser ? 'rgba(0, 212, 255, 0.1)' : 'rgba(0, 255, 127, 0.1)';
        const borderColor = isUser ? '#00d4ff' : '#00ff7f';
        const textColor = '#ffffff';

        messageDiv.style.cssText = `
            margin-bottom: 15px; 
            padding: 12px 15px; 
            border-radius: 12px; 
            background: ${backgroundColor}; 
            border-left: 3px solid ${borderColor};
            animation: fadeInUp 0.3s ease-out;
        `;

        const timestamp = new Date().toLocaleTimeString('it-IT', {
            hour: '2-digit',
            minute: '2-digit'
        });

        messageDiv.innerHTML = `
            <div style="font-weight: bold; font-size: 11px; margin-bottom: 5px; color: ${borderColor}; text-transform: uppercase;">
                ${sender} - ${timestamp}
            </div>
            <div style="line-height: 1.5; color: ${textColor}; word-wrap: break-word;">
                ${text}
            </div>
        `;

        this.elements.chatMessages.appendChild(messageDiv);

        setTimeout(() => {
            this.elements.chatMessages.scrollTop = this.elements.chatMessages.scrollHeight;
        }, 100);

        console.log(`💬 Added ${sender} message: ${text.substring(0, 50)}...`);
    }

    /**
     * INVIA MESSAGGIO WEBSOCKET
     */
    sendWebSocketMessage(type, data = {}) {
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            const message = { type, ...data };
            console.log('📤 SENDING:', JSON.stringify(message, null, 2));
            this.websocket.send(JSON.stringify(message));
            return true;
        } else {
            console.warn('⚠️ WebSocket not connected');
            return false;
        }
    }

    /**
     * GESTIONE STATI COLORE
     */
    setState(newState) {
        if (this.state.currentState !== newState) {
            console.log(`🎨 State change: ${this.state.currentState} → ${newState}`);

            document.body.classList.remove(`state-${this.state.currentState}`);
            this.state.currentState = newState;
            document.body.classList.add(`state-${newState}`);

            this.updateStateIndicators(newState);

            const stateNames = {
                normal: 'Normal Mode',
                whisper: 'Whisper Mode',
                muted: 'Muted',
                sleeping: 'Sleeping',
                processing: 'Processing...',
                speaking: 'Speaking'
            };

            this.showNotification(`Mode: ${stateNames[newState] || newState}`, 'info');
        }
    }

    /**
     * UPDATE STATE INDICATORS
     */
    updateStateIndicators(state) {
        const stateIndicator = document.querySelector('.state-indicator');
        if (stateIndicator) {
            stateIndicator.textContent = state.toUpperCase();
            stateIndicator.className = `state-indicator state-${state}`;
        }
    }

    /**
     * UPDATE VOICE ACTIVITY
     */
    updateVoiceActivity(active) {
        this.state.voiceActive = active;

        if (this.elements.voiceActivity) {
            if (active) {
                this.elements.voiceActivity.classList.add('active');
            } else {
                this.elements.voiceActivity.classList.remove('active');
            }
        }

        if (this.elements.waveBars) {
            this.elements.waveBars.forEach((bar, index) => {
                if (active) {
                    bar.style.animationPlayState = 'running';
                    bar.style.animationDelay = `${index * 0.1}s`;
                } else {
                    bar.style.animationPlayState = 'paused';
                }
            });
        }
    }

    /**
     * UPDATE SYSTEM METRICS
     */
    updateSystemMetrics(metrics) {
        if (!metrics) return;

        this.state.systemMetrics = { ...this.state.systemMetrics, ...metrics };

        if (this.elements.cpuUsage && metrics.cpu_percent !== undefined) {
            this.elements.cpuUsage.textContent = `${Math.round(metrics.cpu_percent)}%`;
        }
        if (this.elements.memoryUsage && metrics.memory_percent !== undefined) {
            this.elements.memoryUsage.textContent = `${Math.round(metrics.memory_percent)}%`;
        }
        if (this.elements.voiceStatus && metrics.voice_status) {
            this.elements.voiceStatus.textContent = metrics.voice_status;
        }
        if (this.elements.aiModel && metrics.ai_model) {
            this.elements.aiModel.textContent = metrics.ai_model;
        }

        this.updateProgressSquare(this.elements.cpuProgress, metrics.cpu_percent || 0);
        this.updateProgressSquare(this.elements.memoryProgress, metrics.memory_percent || 0);
        this.updateProgressSquare(this.elements.networkProgress, metrics.disk_usage_percent || 0);
        this.updateProgressSquare(this.elements.aiProgress, metrics.uptime_seconds > 0 ? 100 : 0);
    }

    /**
     * UPDATE PROGRESS SQUARE
     */
    updateProgressSquare(element, percentage) {
        if (!element) return;

        percentage = Math.max(0, Math.min(100, percentage));

        if (percentage > 70) {
            element.classList.add('active');
        } else {
            element.classList.remove('active');
        }

        element.style.transition = 'all 0.5s ease';
        element.style.setProperty('--fill-height', `${percentage}%`);
    }

    /**
     * UPDATE CONNECTION STATUS
     */
    updateConnectionStatus(connected) {
        if (this.elements.wsStatus) {
            if (connected) {
                this.elements.wsStatus.classList.add('connected');
                this.elements.wsStatus.textContent = '●';
            } else {
                this.elements.wsStatus.classList.remove('connected');
                this.elements.wsStatus.textContent = '○';
            }
        }

        if (this.elements.backendStatus) {
            this.elements.backendStatus.textContent = connected ? 'Connected - STREAMING' : 'Disconnected';
            this.elements.backendStatus.style.color = connected ? '#00ff7f' : '#ff4757';
        }

        document.body.classList.toggle('websocket-connected', connected);
    }

    /**
     * CONTROL ACTIONS
     */
    toggleVoice() {
        console.log('🎤 Voice toggle requested');
        this.sendWebSocketMessage('voice_toggle');
        this.showNotification('Voice toggle requested', 'info');
    }

    openSettings() {
        this.showNotification('Settings panel coming soon...', 'info');
        console.log('⚙️ Settings panel requested');
    }

    minimizeWindow() {
        if (window.require) {
            const { ipcRenderer } = window.require('electron');
            ipcRenderer.invoke('minimize-window');
        } else {
            console.log('🪟 Minimize requested (Electron not available)');
        }
    }

    toggleFullscreen() {
        if (document.fullscreenElement) {
            document.exitFullscreen();
        } else {
            document.documentElement.requestFullscreen();
        }
    }

    /**
     * KEYBOARD SHORTCUTS
     */
    handleKeyboard(event) {
        if (event.target === this.elements.chatInput) {
            return;
        }

        switch (event.key) {
            case 'F1':
                event.preventDefault();
                this.toggleVoice();
                break;

            case 'F2':
                event.preventDefault();
                this.openSettings();
                break;

            case 'F3':
                event.preventDefault();
                if (this.elements.chatBtn) {
                    this.elements.chatBtn.click();
                }
                break;

            case 'Escape':
                if (document.fullscreenElement) {
                    document.exitFullscreen();
                }
                if (this.elements.chatPanel && this.elements.chatPanel.style.right === '20px') {
                    this.elements.chatPanel.style.right = '-400px';
                }
                break;

            case 'Enter':
                if (this.elements.chatPanel && this.elements.chatPanel.style.right !== '20px') {
                    this.elements.chatBtn?.click();
                    setTimeout(() => {
                        this.elements.chatInput?.focus();
                    }, 300);
                }
                break;
        }
    }

    /**
     * CURSOR TRACKING
     */
    updateCursor(event) {
        if (!this.lastCursorUpdate || Date.now() - this.lastCursorUpdate > 16) {
            document.documentElement.style.setProperty('--cursor-x', event.clientX + 'px');
            document.documentElement.style.setProperty('--cursor-y', event.clientY + 'px');
            this.lastCursorUpdate = Date.now();
        }
    }

    /**
     * SISTEMA NOTIFICHE
     */
    showNotification(text, level = 'info', duration = 5000) {
        if (!this.elements.notifications) return;

        const notification = document.createElement('div');
        notification.className = `notification notification-${level}`;

        const icons = {
            info: 'ℹ️',
            success: '✅',
            warning: '⚠️',
            error: '❌'
        };

        notification.innerHTML = `
            <div class="notification-content">
                <span class="notification-icon">${icons[level] || 'ℹ️'}</span>
                <span class="notification-text">${text}</span>
                <button class="notification-close">×</button>
            </div>
        `;

        notification.style.cssText = `
            transform: translateX(100%);
            transition: transform 0.3s ease-out;
        `;

        this.elements.notifications.appendChild(notification);

        setTimeout(() => {
            notification.style.transform = 'translateX(0)';
        }, 10);

        const removeNotification = () => {
            notification.style.transform = 'translateX(100%)';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        };

        const autoRemoveTimer = setTimeout(removeNotification, duration);

        notification.querySelector('.notification-close').addEventListener('click', () => {
            clearTimeout(autoRemoveTimer);
            removeNotification();
        });

        console.log(`📢 ${level.toUpperCase()}: ${text}`);
    }

    /**
     * LOADING OVERLAY
     */
    hideLoadingOverlay() {
        const stages = [
            'Initializing Core Systems...',
            'Connecting to Backend...',
            'Loading AI Models...',
            'Activating Voice Systems...',
            'Enabling STREAMING Mode...',
            'Finalizing Interface...'
        ];

        let progress = 0;
        let currentStage = 0;

        const progressInterval = setInterval(() => {
            progress += Math.random() * 15 + 5;

            if (progress >= 100) {
                progress = 100;
            }

            if (this.elements.loadingProgress) {
                this.elements.loadingProgress.style.width = `${progress}%`;
            }

            const newStage = Math.floor((progress / 100) * stages.length);
            if (newStage !== currentStage && newStage < stages.length) {
                currentStage = newStage;
                const stageElement = document.querySelector('.loading-stage');
                if (stageElement) {
                    stageElement.textContent = stages[currentStage];
                }
            }

            if (progress >= 100) {
                clearInterval(progressInterval);

                setTimeout(() => {
                    if (this.elements.loadingOverlay) {
                        this.elements.loadingOverlay.style.opacity = '0';
                        this.elements.loadingOverlay.style.transition = 'opacity 0.5s ease-out';

                        setTimeout(() => {
                            this.elements.loadingOverlay.classList.add('hidden');
                        }, 500);
                    }
                }, 500);
            }
        }, 200);
    }

    /**
     * START UPDATE LOOPS
     */
    startUpdateLoops() {
        // Metrics update loop disabilitato per modalità streaming
        const metricsLoop = setInterval(() => {
            if (this.state.connected) {
                console.log('🔇 Metrics request disabilitato in modalità streaming');
            } else {
                this.updateSystemMetrics({
                    cpu_percent: Math.floor(Math.random() * 30) + 10,
                    memory_percent: Math.floor(Math.random() * 40) + 20,
                    disk_usage_percent: Math.floor(Math.random() * 20) + 5,
                    voice_status: 'OFFLINE',
                    ai_model: 'DISCONNECTED',
                    uptime_seconds: 0
                });
            }
        }, this.config.updateInterval);

        // Heartbeat loop ridotto
        const heartbeatLoop = setInterval(() => {
            if (this.state.connected) {
                console.log('🔇 Ping disabilitato in modalità streaming');
            }
        }, 30000);

        // Voice activity simulation
        const voiceSimLoop = setInterval(() => {
            if (!this.state.connected && Math.random() > 0.95) {
                this.updateVoiceActivity(true);
                setTimeout(() => this.updateVoiceActivity(false), 2000);
            }
        }, 5000);

        console.log('🔄 Update loops started (streaming mode)');
        this.updateIntervals = [metricsLoop, heartbeatLoop, voiceSimLoop];
    }

    /**
     * HANDLE JARVIS RESPONSE
     */
    handleJarvisResponse(message) {
        console.log('🤖 Handling Jarvis response:', message);

        this.setState('speaking');
        this.updateVoiceActivity(true);

        const estimatedDuration = Math.max(2000, (message.text?.length || 0) * 50);

        setTimeout(() => {
            this.setState('normal');
            this.updateVoiceActivity(false);
        }, message.duration || estimatedDuration);

        this.showNotification('Response generated', 'success', 2000);
    }

    /**
     * UTILITY METHODS
     */
    getSystemState() {
        return {
            connected: this.state.connected,
            currentState: this.state.currentState,
            voiceActive: this.state.voiceActive,
            metrics: this.state.systemMetrics,
            websocketReady: this.websocket?.readyState === WebSocket.OPEN,
            streamingSupport: true,
            streamingState: this.streamingState
        };
    }

    forceReconnect() {
        console.log('🔄 Forcing WebSocket reconnection...');

        if (this.websocket) {
            this.websocket.close();
        }

        setTimeout(() => {
            this.connectWebSocket();
        }, 1000);

        this.showNotification('Reconnecting to backend...', 'info');
    }

    clearChat() {
        if (this.elements.chatMessages) {
            this.elements.chatMessages.innerHTML = '';
            console.log('💬 Chat cleared');
            this.showNotification('Chat cleared', 'info');
        }
    }

    enableDebugMode() {
        console.log('🐛 Debug mode enabled');
        document.body.classList.add('debug-mode');
        this.debugMode = true;

        const debugOverlay = document.createElement('div');
        debugOverlay.id = 'debug-overlay';
        debugOverlay.style.cssText = `
            position: fixed;
            top: 10px;
            left: 10px;
            background: rgba(0, 0, 0, 0.8);
            color: #00ff00;
            padding: 10px;
            font-family: monospace;
            font-size: 12px;
            border-radius: 5px;
            z-index: 10000;
            max-width: 300px;
        `;
        document.body.appendChild(debugOverlay);

        setInterval(() => {
            if (debugOverlay) {
                const state = this.getSystemState();
                debugOverlay.innerHTML = `
                    <div><strong>JARVIS DEBUG - STREAMING</strong></div>
                    <div>Connected: ${state.connected ? '✅' : '❌'}</div>
                    <div>State: ${state.currentState}</div>
                    <div>Voice: ${state.voiceActive ? '🎤' : '🔇'}</div>
                    <div>WS: ${state.websocketReady ? 'OPEN' : 'CLOSED'}</div>
                    <div>Streaming: ${state.streamingState.isStreaming ? '🔥' : '💤'}</div>
                    <div>Chunks: ${state.streamingState.chunksReceived}</div>
                    <div>Particles: ${this.particles.length}</div>
                `;
            }
        }, 1000);
    }

    /**
     * Cleanup on destroy
     */
    destroy() {
        console.log('🗑️ Destroying Jarvis App...');

        if (this.websocket) {
            this.websocket.close();
        }

        if (this.updateIntervals) {
            this.updateIntervals.forEach(interval => clearInterval(interval));
        }

        window.removeEventListener('resize', this.resizeCanvas);
        document.removeEventListener('keydown', this.handleKeyboard);
        document.removeEventListener('mousemove', this.updateCursor);

        this.particles = [];

        console.log('✅ Jarvis App destroyed');
    }
}

/**
 * GLOBAL ERROR HANDLER
 */
window.addEventListener('error', (event) => {
    console.error('🚨 Global error:', event.error);
    if (window.jarvis) {
        window.jarvis.showNotification(`Error: ${event.error.message}`, 'error');
    }
});

window.addEventListener('unhandledrejection', (event) => {
    console.error('🚨 Unhandled promise rejection:', event.reason);
    if (window.jarvis) {
        window.jarvis.showNotification(`Promise error: ${event.reason}`, 'error');
    }
});

/**
 * INIZIALIZZAZIONE APP
 */
/* CSS per typing cursor animation */
const style = document.createElement('style');
style.textContent = `
    @keyframes blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0; }
    }
    .typing-cursor {
        animation: blink 1s infinite;
        color: #00d4ff;
        font-weight: bold;
    }
    .streaming-content {
        min-height: 20px;
    }
`;
document.head.appendChild(style);

document.addEventListener('DOMContentLoaded', () => {
    console.log('📄 DOM loaded, initializing Jarvis with STREAMING support...');

    // Create global instance
    window.jarvis = new JarvisApp();

    // Debug helpers con streaming support
    window.jarvisDebug = {
        getState: () => window.jarvis.getSystemState(),
        reconnect: () => window.jarvis.forceReconnect(),
        clearChat: () => window.jarvis.clearChat(),
        enableDebug: () => window.jarvis.enableDebugMode(),
        sendTest: (msg) => window.jarvis.sendWebSocketMessage('text_command', { text: msg }),
        sendEcho: (msg) => window.jarvis.sendWebSocketMessage('test_echo', { text: msg }),
        particles: () => window.jarvis.particles.length,
        toggleVoice: () => window.jarvis.toggleVoice(),
        streamingState: () => window.jarvis.streamingState,
        startStreaming: () => window.jarvis.startStreamingMessage(),
        completeStreaming: () => window.jarvis.completeStreamingMessage()
    };

    console.log('🎮 Debug helpers available: window.jarvisDebug');
    console.log('🚀 Streaming support: ENABLED');
    console.log('✅ Jarvis Frontend Application with STREAMING loaded successfully');
});

/**
 * CLEANUP ON PAGE UNLOAD
 */
window.addEventListener('beforeunload', () => {
    if (window.jarvis) {
        window.jarvis.destroy();
    }
});