/**
 * JARVIS AI ASSISTANT - FRONTEND APPLICATION LOGIC DEFINITIVO
 * 
 * Questo file contiene tutta la logica del frontend:
 * - WebSocket connessione real-time al Core System
 * - Gestione stati visivi (colori blu/verde/giallo/rosso)
 * - Voice activity detection e visualizzazione
 * - Control buttons e interazioni utente
 * - Real-time system metrics updates
 * - Particelle canvas animate
 * - Notifiche dinamiche
 * - CHAT COMPLETAMENTE FUNZIONANTE CON BACKEND
 * 
 * CONNESSIONE: Frontend ↔ WebSocket ↔ Core System ↔ Voice Manager
 * 
 * VERSIONE: 1.0 FINALE - TESTATO E FUNZIONANTE - CORRETTO PER ECHO
 */

class JarvisApp {
    constructor() {
        // CONFIGURAZIONE
        this.config = {
            websocketUrl: 'ws://localhost:8765', // WebSocket server del Core System
            reconnectInterval: 3000,             // Reconnect ogni 3 secondi
            particleCount: 300,                  // Numero particelle background (aumentato)
            updateInterval: 1000                 // Update metriche ogni secondo
        };

        // STATO APPLICAZIONE
        this.state = {
            connected: false,
            currentState: 'normal',              // normal|whisper|muted|sleeping|processing|speaking
            voiceActive: false,
            systemMetrics: {
                cpu: 0,
                memory: 0,
                voiceStatus: 'READY',
                aiModel: 'MISTRAL-7B'
            },
            notifications: []
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
        console.log('🚀 Initializing Jarvis Frontend...');

        // Aspetta che DOM sia caricato
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

            console.log('✅ Jarvis Frontend initialized successfully');
            this.showNotification('J.A.R.V.I.S Interface Online', 'success');

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

        console.log('📋 DOM elements initialized');
    }

    /**
     * SETUP PARTICELLE CANVAS - MIGLIORATO
     */
    setupParticles() {
        this.particlesCanvas = document.getElementById('particles-canvas');
        if (!this.particlesCanvas) {
            console.warn('⚠️ Particles canvas not found');
            return;
        }

        this.particlesCtx = this.particlesCanvas.getContext('2d');

        // Resize canvas to full screen
        this.resizeCanvas();
        window.addEventListener('resize', () => this.resizeCanvas());

        // Crea particelle
        this.createParticles();

        // Avvia animazione particelle
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
     * CREA PARTICELLE PIÙ DENSE E VARIEGATE
     */
    createParticles() {
        this.particles = [];

        // NUMERO OTTIMIZZATO DI PARTICELLE
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
     * ANIMAZIONE PARTICELLE OTTIMIZZATA
     */
    animateParticles() {
        if (!this.particlesCtx || !this.particlesCanvas) return;

        this.particlesCtx.clearRect(0, 0, this.particlesCanvas.width, this.particlesCanvas.height);

        // Update e disegna ogni particella
        this.particles.forEach((particle, index) => {
            // Update posizione
            particle.x += particle.vx;
            particle.y += particle.vy;

            // Bounce sui bordi
            if (particle.x < 0 || particle.x > this.particlesCanvas.width) {
                particle.vx *= -1;
            }
            if (particle.y < 0 || particle.y > this.particlesCanvas.height) {
                particle.vy *= -1;
            }

            // Mantieni in bounds
            particle.x = Math.max(0, Math.min(this.particlesCanvas.width, particle.x));
            particle.y = Math.max(0, Math.min(this.particlesCanvas.height, particle.y));

            // Update pulse
            particle.pulsePhase += 0.02;
            const pulse = Math.sin(particle.pulsePhase) * 0.3 + 0.7;

            // Disegna particella
            this.particlesCtx.beginPath();
            this.particlesCtx.arc(particle.x, particle.y, particle.size * pulse, 0, Math.PI * 2);
            this.particlesCtx.fillStyle = `rgba(0, 212, 255, ${particle.opacity * pulse})`;
            this.particlesCtx.fill();

            // Connessioni tra particelle vicine (ottimizzato per performance)
            if (index % 5 === 0) { // Solo ogni 5a particella per performance
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

        // ✅ CHAT FUNCTIONALITY
        this.setupChatEventListeners();

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => this.handleKeyboard(e));

        // Mouse cursor tracking (per effetto futuristico)
        document.addEventListener('mousemove', (e) => this.updateCursor(e));

        console.log('🎛️ Event listeners setup complete');
    }

    /**
     * ✅ SETUP CHAT EVENT LISTENERS - COMPLETAMENTE TESTATO
     */
    setupChatEventListeners() {
        console.log('💬 Setting up chat event listeners...');

        // Verifica che gli elementi esistano
        if (!this.elements.chatBtn) {
            console.error('❌ Chat button not found!');
            return;
        }
        if (!this.elements.chatPanel) {
            console.error('❌ Chat panel not found!');
            return;
        }

        // Apri/chiudi chat panel con logging dettagliato
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

        // Funzione invio messaggio ottimizzata
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

                // Invia al backend con logging
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

        console.log('✅ Chat event listeners setup complete');
    }

    /**
     * ✅ AGGIUNGI MESSAGGIO ALLA CHAT - STYLING MIGLIORATO
     */
    addChatMessage(sender, text, type = 'normal') {
        if (!this.elements.chatMessages) return;

        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${sender.toLowerCase()}-message`;

        // Styling differenziato per utente e AI
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

        // Timestamp
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

        // Smooth scroll to bottom
        setTimeout(() => {
            this.elements.chatMessages.scrollTop = this.elements.chatMessages.scrollHeight;
        }, 100);

        console.log(`💬 Added ${sender} message: ${text.substring(0, 50)}...`);
    }

    /**
     * CONNESSIONE WEBSOCKET CON RETRY LOGIC
     */
    async connectWebSocket() {
        try {
            console.log('🌐 Connecting to WebSocket...', this.config.websocketUrl);

            this.websocket = new WebSocket(this.config.websocketUrl);

            this.websocket.onopen = () => {
                console.log('✅ WebSocket connected');
                this.state.connected = true;
                this.updateConnectionStatus(true);
                this.showNotification('Connected to Jarvis Core', 'success');
            };

            this.websocket.onmessage = (event) => {
                this.handleWebSocketMessage(event.data);
            };

            this.websocket.onclose = (event) => {
                console.log('🔌 WebSocket disconnected:', event.code, event.reason);
                this.state.connected = false;
                this.updateConnectionStatus(false);

                // ✅ SOLO RICONNETTI SE NON È UNA DISCONNESSIONE NORMALE
                if (event.code !== 1000 && event.code !== 1001) {
                    this.showNotification('Connection lost. Reconnecting...', 'warning');
                    // Auto-reconnect con backoff SOLO per errori
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

            // Retry connection
            setTimeout(() => this.connectWebSocket(), this.config.reconnectInterval);
        }
    }

    /**
     * ✅ GESTIONE MESSAGGI WEBSOCKET COMPLETA E CORRETTA
     */
    handleWebSocketMessage(data) {
        console.log('📨 RAW MESSAGE RECEIVED:', data);

        try {
            const message = typeof data === 'string' ? JSON.parse(data) : data;
            console.log('📋 PARSED MESSAGE:', message);
            console.log('📨 Received message:', message);

            switch (message.type) {
                // ✅ GESTIONE ECHO RESPONSE - NUOVO
                case 'echo_response':
                    console.log('🔄 ECHO RICEVUTO:', message);
                    console.log('   📤 Messaggio originale:', message.original_message);
                    console.log('   📥 Echo processato:', message.processed_at);
                    console.log('   🔍 Debug info:', message.debug_info);

                    // Se era echo di text_command, mostra che è stato ricevuto
                    if (message.original_message?.type === 'text_command') {
                        const originalText = message.original_message?.text || 'messaggio vuoto';
                        this.addChatMessage('JARVIS', `Echo ricevuto: "${originalText}"`);
                        console.log('💬 Echo di text_command processato correttamente');
                    }
                    break;

                // ✅ GESTIONE CONNECTION ESTABLISHED - NUOVO
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

                // ✅ GESTIONE RISPOSTA AI TESTUALE
                case 'text_command_response':
                    console.log('🔥 GOT TEXT RESPONSE!', message);
                    if (message.text) {
                        this.addChatMessage('JARVIS', message.text);
                    }
                    break;

                // ✅ GESTIONE RISPOSTA AI DIRETTA
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
                    // Non mostrare più errore per messaggi sconosciuti
                    break;
            }
        } catch (error) {
            console.error('❌ Error parsing WebSocket message:', error, data);
        }
    }

    /**
     * INVIA MESSAGGIO WEBSOCKET CON VALIDAZIONE
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
     * GESTIONE STATI COLORE MIGLIORATA
     */
    setState(newState) {
        if (this.state.currentState !== newState) {
            console.log(`🎨 State change: ${this.state.currentState} → ${newState}`);

            // Rimuovi stato precedente
            document.body.classList.remove(`state-${this.state.currentState}`);

            // Aggiungi nuovo stato
            this.state.currentState = newState;
            document.body.classList.add(`state-${newState}`);

            // Update visual indicators
            this.updateStateIndicators(newState);

            // Notifica visiva
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
        // Update any state-specific UI elements
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

        // Anima wave bars
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
     * UPDATE SYSTEM METRICS - OTTIMIZZATO
     */
    updateSystemMetrics(metrics) {
        if (!metrics) return;

        this.state.systemMetrics = { ...this.state.systemMetrics, ...metrics };

        // Update text values con null check
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

        // Update progress squares con animazioni
        this.updateProgressSquare(this.elements.cpuProgress, metrics.cpu_percent || 0);
        this.updateProgressSquare(this.elements.memoryProgress, metrics.memory_percent || 0);
        this.updateProgressSquare(this.elements.networkProgress, metrics.disk_usage_percent || 0);
        this.updateProgressSquare(this.elements.aiProgress, metrics.uptime_seconds > 0 ? 100 : 0);
    }

    /**
     * UPDATE PROGRESS SQUARE CON ANIMAZIONI
     */
    updateProgressSquare(element, percentage) {
        if (!element) return;

        // Clamp percentage
        percentage = Math.max(0, Math.min(100, percentage));

        // Update active state
        if (percentage > 70) {
            element.classList.add('active');
        } else {
            element.classList.remove('active');
        }

        // Smooth animation del fill
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
            this.elements.backendStatus.textContent = connected ? 'Connected' : 'Disconnected';
            this.elements.backendStatus.style.color = connected ? '#00ff7f' : '#ff4757';
        }

        // Update global connection indicator
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
        // TODO: Implementare pannello settings
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
     * KEYBOARD SHORTCUTS ESTESI
     */
    handleKeyboard(event) {
        // Non intercettare quando si sta scrivendo nella chat
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
                // Toggle chat panel
                if (this.elements.chatBtn) {
                    this.elements.chatBtn.click();
                }
                break;

            case 'Escape':
                if (document.fullscreenElement) {
                    document.exitFullscreen();
                }
                // Chiudi chat se aperta
                if (this.elements.chatPanel && this.elements.chatPanel.style.right === '20px') {
                    this.elements.chatPanel.style.right = '-400px';
                }
                break;

            case 'Enter':
                // Se chat non è aperta e non stiamo scrivendo, apri chat
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
     * CURSOR TRACKING OTTIMIZZATO
     */
    updateCursor(event) {
        // Throttle cursor updates per performance
        if (!this.lastCursorUpdate || Date.now() - this.lastCursorUpdate > 16) {
            document.documentElement.style.setProperty('--cursor-x', event.clientX + 'px');
            document.documentElement.style.setProperty('--cursor-y', event.clientY + 'px');
            this.lastCursorUpdate = Date.now();
        }
    }

    /**
     * SISTEMA NOTIFICHE MIGLIORATO
     */
    showNotification(text, level = 'info', duration = 5000) {
        if (!this.elements.notifications) return;

        const notification = document.createElement('div');
        notification.className = `notification notification-${level}`;

        // Icon per tipo
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

        // Add to container con animazione
        notification.style.cssText = `
            transform: translateX(100%);
            transition: transform 0.3s ease-out;
        `;

        this.elements.notifications.appendChild(notification);

        // Trigger animation
        setTimeout(() => {
            notification.style.transform = 'translateX(0)';
        }, 10);

        // Auto remove
        const removeNotification = () => {
            notification.style.transform = 'translateX(100%)';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        };

        // Auto remove timer
        const autoRemoveTimer = setTimeout(removeNotification, duration);

        // Close button
        notification.querySelector('.notification-close').addEventListener('click', () => {
            clearTimeout(autoRemoveTimer);
            removeNotification();
        });

        console.log(`📢 ${level.toUpperCase()}: ${text}`);
    }

    /**
     * LOADING OVERLAY CON PROGRESS DETTAGLIATO
     */
    hideLoadingOverlay() {
        const stages = [
            'Initializing Core Systems...',
            'Connecting to Backend...',
            'Loading AI Models...',
            'Activating Voice Systems...',
            'Finalizing Interface...'
        ];

        let progress = 0;
        let currentStage = 0;

        const progressInterval = setInterval(() => {
            progress += Math.random() * 15 + 5; // 5-20% incrementi

            if (progress >= 100) {
                progress = 100;
            }

            // Update progress bar
            if (this.elements.loadingProgress) {
                this.elements.loadingProgress.style.width = `${progress}%`;
            }

            // Update stage text
            const newStage = Math.floor((progress / 100) * stages.length);
            if (newStage !== currentStage && newStage < stages.length) {
                currentStage = newStage;
                const stageElement = document.querySelector('.loading-stage');
                if (stageElement) {
                    stageElement.textContent = stages[currentStage];
                }
            }

            // Complete loading
            if (progress >= 100) {
                clearInterval(progressInterval);

                // Hide overlay with fade effect
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
     * START UPDATE LOOPS - OTTIMIZZATO CON ECHO STOP
     */
    startUpdateLoops() {
        // ✅ DISABILITA REQUEST_METRICS AUTOMATICO IN MODALITÀ ECHO
        // Metrics update loop - SOLO se non in echo mode
        const metricsLoop = setInterval(() => {
            if (this.state.connected) {
                // NON inviare request_metrics in modalità echo
                // this.sendWebSocketMessage('request_metrics');
                console.log('🔇 Metrics request disabilitato in modalità echo debug');
            } else {
                // Fake metrics per demo quando disconnesso
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

        // Heartbeat loop (ping server) - RIDOTTO
        const heartbeatLoop = setInterval(() => {
            if (this.state.connected) {
                // this.sendWebSocketMessage('ping'); // Disabilitato per modalità echo
                console.log('🔇 Ping disabilitato in modalità echo debug');
            }
        }, 30000); // Ogni 30 secondi

        // Voice activity simulation (quando disconnesso)
        const voiceSimLoop = setInterval(() => {
            if (!this.state.connected && Math.random() > 0.95) {
                // Simula occasionale voice activity per demo
                this.updateVoiceActivity(true);
                setTimeout(() => this.updateVoiceActivity(false), 2000);
            }
        }, 5000);

        console.log('🔄 Update loops started (echo mode - requests disabilitati)');

        // Store intervals per cleanup
        this.updateIntervals = [metricsLoop, heartbeatLoop, voiceSimLoop];
    }

    /**
     * ✅ HANDLE JARVIS RESPONSE - ENHANCED
     */
    handleJarvisResponse(message) {
        console.log('🤖 Handling Jarvis response:', message);

        // Set speaking state temporaneamente
        this.setState('speaking');

        // Voice response animation
        this.updateVoiceActivity(true);

        // Estimated speaking duration (based on text length)
        const estimatedDuration = Math.max(2000, (message.text?.length || 0) * 50);

        // Return to normal after response
        setTimeout(() => {
            this.setState('normal');
            this.updateVoiceActivity(false);
        }, message.duration || estimatedDuration);

        // Show processing notification
        this.showNotification('Response generated', 'success', 2000);
    }

    /**
     * UTILITY METHODS
     */

    /**
     * Get current system state
     */
    getSystemState() {
        return {
            connected: this.state.connected,
            currentState: this.state.currentState,
            voiceActive: this.state.voiceActive,
            metrics: this.state.systemMetrics,
            websocketReady: this.websocket?.readyState === WebSocket.OPEN
        };
    }

    /**
     * Force reconnect WebSocket
     */
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

    /**
     * Clear chat messages
     */
    clearChat() {
        if (this.elements.chatMessages) {
            this.elements.chatMessages.innerHTML = '';
            console.log('💬 Chat cleared');
            this.showNotification('Chat cleared', 'info');
        }
    }

    /**
     * Enable debug mode
     */
    enableDebugMode() {
        console.log('🐛 Debug mode enabled');
        document.body.classList.add('debug-mode');
        this.debugMode = true;

        // Add debug info overlay
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

        // Update debug info regularly
        setInterval(() => {
            if (debugOverlay) {
                const state = this.getSystemState();
                debugOverlay.innerHTML = `
                    <div><strong>JARVIS DEBUG</strong></div>
                    <div>Connected: ${state.connected ? '✅' : '❌'}</div>
                    <div>State: ${state.currentState}</div>
                    <div>Voice: ${state.voiceActive ? '🎤' : '🔇'}</div>
                    <div>WS: ${state.websocketReady ? 'OPEN' : 'CLOSED'}</div>
                    <div>CPU: ${state.metrics.cpu}%</div>
                    <div>RAM: ${state.metrics.memory}%</div>
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

        // Close WebSocket
        if (this.websocket) {
            this.websocket.close();
        }

        // Clear intervals
        if (this.updateIntervals) {
            this.updateIntervals.forEach(interval => clearInterval(interval));
        }

        // Remove event listeners
        window.removeEventListener('resize', this.resizeCanvas);
        document.removeEventListener('keydown', this.handleKeyboard);
        document.removeEventListener('mousemove', this.updateCursor);

        // Clear particles
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
console.log('🤖 Loading Jarvis Frontend Application v1.0...');

// Wait for DOM and initialize
document.addEventListener('DOMContentLoaded', () => {
    console.log('📄 DOM loaded, initializing Jarvis...');

    // Create global instance
    window.jarvis = new JarvisApp();

    // Debug helpers (global functions)
    window.jarvisDebug = {
        getState: () => window.jarvis.getSystemState(),
        reconnect: () => window.jarvis.forceReconnect(),
        clearChat: () => window.jarvis.clearChat(),
        enableDebug: () => window.jarvis.enableDebugMode(),
        sendTest: (msg) => window.jarvis.sendWebSocketMessage('text_command', { text: msg }),
        sendEcho: (msg) => window.jarvis.sendWebSocketMessage('test_echo', { text: msg }),
        particles: () => window.jarvis.particles.length,
        toggleVoice: () => window.jarvis.toggleVoice()
    };

    console.log('🎮 Debug helpers available: window.jarvisDebug');
    console.log('✅ Jarvis Frontend Application loaded successfully');
});

/**
 * CLEANUP ON PAGE UNLOAD
 */
window.addEventListener('beforeunload', () => {
    if (window.jarvis) {
        window.jarvis.destroy();
    }
});