/**
 * JARVIS AI ASSISTANT - FRONTEND APPLICATION LOGIC
 * 
 * Questo file contiene tutta la logica del frontend:
 * - WebSocket connessione real-time al Core System
 * - Gestione stati visivi (colori blu/verde/giallo/rosso)
 * - Voice activity detection e visualizzazione
 * - Control buttons e interazioni utente
 * - Real-time system metrics updates
 * - Particelle canvas animate
 * - Notifiche dinamiche
 * 
 * CONNESSIONE: Frontend ↔ WebSocket ↔ Core System ↔ Voice Manager
 */

class JarvisApp {
    constructor() {
        // CONFIGURAZIONE
        this.config = {
            websocketUrl: 'ws://localhost:8765/ws', // WebSocket server del Core System
            reconnectInterval: 3000,             // Reconnect ogni 3 secondi
            particleCount: 150,                  // Numero particelle background
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
            settingsBtn: document.getElementById('settings-btn'),
            minimizeBtn: document.getElementById('minimize-btn'),
            fullscreenBtn: document.getElementById('fullscreen-btn'),

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
     * SETUP PARTICELLE CANVAS
     */
    setupParticles() {
        this.particlesCanvas = document.getElementById('particles-canvas');
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
        this.particlesCanvas.width = window.innerWidth;
        this.particlesCanvas.height = window.innerHeight;
    }

    /**
     * CREA PARTICELLE PIÙ DENSE
     */
    createParticles() {
        this.particles = [];

        // AUMENTATO IL NUMERO DI PARTICELLE: 300 invece di 150
        for (let i = 0; i < 300; i++) {
            this.particles.push({
                x: Math.random() * this.particlesCanvas.width,
                y: Math.random() * this.particlesCanvas.height,
                vx: (Math.random() - 0.5) * 0.8, // Velocità leggermente maggiore
                vy: (Math.random() - 0.5) * 0.8,
                size: Math.random() * 3 + 0.5, // Particelle leggermente più grandi
                opacity: Math.random() * 0.7 + 0.3, // Più visibili
                pulsePhase: Math.random() * Math.PI * 2,
                // NUOVI: Tipo di particella per varietà
                type: Math.random() > 0.7 ? 'neural' : 'normal'
            });
        }
    }

    /**
     * ANIMAZIONE PARTICELLE
     */
    animateParticles() {
        this.particlesCtx.clearRect(0, 0, this.particlesCanvas.width, this.particlesCanvas.height);

        // Update e disegna ogni particella
        this.particles.forEach(particle => {
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

            // Connessioni tra particelle vicine
            this.particles.forEach(other => {
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

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => this.handleKeyboard(e));

        // Mouse cursor tracking (per effetto futuristico)
        document.addEventListener('mousemove', (e) => this.updateCursor(e));

        console.log('🎛️ Event listeners setup complete');
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
                this.showNotification('Connected to Jarvis Core', 'success');
            };

            this.websocket.onmessage = (event) => {
                this.handleWebSocketMessage(event.data);
            };

            this.websocket.onclose = () => {
                console.log('🔌 WebSocket disconnected');
                this.state.connected = false;
                this.updateConnectionStatus(false);
                this.showNotification('Connection lost. Reconnecting...', 'warning');

                // Auto-reconnect
                setTimeout(() => this.connectWebSocket(), this.config.reconnectInterval);
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
     * GESTIONE MESSAGGI WEBSOCKET
     */
    handleWebSocketMessage(data) {
        try {
            const message = JSON.parse(data);

            switch (message.type) {
                case 'state_change':
                    this.setState(message.state);
                    break;

                case 'voice_activity':
                    this.updateVoiceActivity(message.active);
                    break;

                case 'system_metrics':
                    this.updateSystemMetrics(message.data);  // ✅ QUI DOVREBBE FUNZIONARE
                    break;

                // ... resto del codice
            }
        } catch (error) {
            console.error('❌ Errore parsing messaggio:', error);
        }
    }

    /**
     * INVIA MESSAGGIO WEBSOCKET
     */
    sendWebSocketMessage(type, data = {}) {
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            this.websocket.send(JSON.stringify({ type, ...data }));
        } else {
            console.warn('⚠️ WebSocket not connected, cannot send message');
        }
    }

    /**
     * GESTIONE STATI COLORE
     */
    setState(newState) {
        if (this.state.currentState !== newState) {
            console.log(`🎨 State change: ${this.state.currentState} → ${newState}`);

            // Rimuovi stato precedente
            document.body.classList.remove(`state-${this.state.currentState}`);

            // Aggiungi nuovo stato
            this.state.currentState = newState;
            document.body.classList.add(`state-${newState}`);

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
     * UPDATE VOICE ACTIVITY
     */
    updateVoiceActivity(active) {
        this.state.voiceActive = active;

        if (active) {
            this.elements.voiceActivity?.classList.add('active');
        } else {
            this.elements.voiceActivity?.classList.remove('active');
        }

        // Anima wave bars
        this.elements.waveBars?.forEach((bar, index) => {
            if (active) {
                bar.style.animationPlayState = 'running';
            } else {
                bar.style.animationPlayState = 'paused';
            }
        });
    }

    /**
   * UPDATE SYSTEM METRICS
   */
    updateSystemMetrics(metrics) {
        this.state.systemMetrics = { ...this.state.systemMetrics, ...metrics };

        // Update text values - ✅ CORRETTI
        if (this.elements.cpuUsage) {
            this.elements.cpuUsage.textContent = `${metrics.cpu_percent || 0}%`;
        }
        if (this.elements.memoryUsage) {
            this.elements.memoryUsage.textContent = `${metrics.memory_percent || 0}%`;
        }
        if (this.elements.voiceStatus && metrics.voice_status) {
            this.elements.voiceStatus.textContent = metrics.voice_status;
        }
        if (this.elements.aiModel && metrics.ai_model) {
            this.elements.aiModel.textContent = metrics.ai_model;
        }

        // Update progress squares - ✅ CORRETTI
        this.updateProgressSquare(this.elements.cpuProgress, metrics.cpu_percent || 0);
        this.updateProgressSquare(this.elements.memoryProgress, metrics.memory_percent || 0);
        this.updateProgressSquare(this.elements.networkProgress, metrics.disk_usage_percent || 0);
        this.updateProgressSquare(this.elements.aiProgress, metrics.uptime_seconds > 0 ? 100 : 0);
    }

    /**
     * UPDATE PROGRESS SQUARE
     */
    updateProgressSquare(element, percentage) {
        if (element) {
            if (percentage > 70) {
                element.classList.add('active');
            } else {
                element.classList.remove('active');
            }

            // Update fill height
            element.style.setProperty('--fill-height', `${percentage}%`);
        }
    }

    /**
     * UPDATE CONNECTION STATUS
     */
    updateConnectionStatus(connected) {
        if (this.elements.wsStatus) {
            if (connected) {
                this.elements.wsStatus.classList.add('connected');
            } else {
                this.elements.wsStatus.classList.remove('connected');
            }
        }

        if (this.elements.backendStatus) {
            this.elements.backendStatus.textContent = connected ? 'Connected' : 'Disconnected';
        }
    }

    /**
     * CONTROL ACTIONS
     */
    toggleVoice() {
        this.sendWebSocketMessage('voice_toggle');
        this.showNotification('Voice toggle requested', 'info');
    }

    openSettings() {
        this.showNotification('Settings panel coming soon...', 'info');
        // TODO: Implementare pannello settings
    }

    minimizeWindow() {
        if (window.require) {
            const { ipcRenderer } = window.require('electron');
            ipcRenderer.invoke('minimize-window');
        }
    }

    toggleFullscreen() {
        if (window.require) {
            const { ipcRenderer } = window.require('electron');
            ipcRenderer.invoke('toggle-fullscreen');
        }
    }

    /**
     * KEYBOARD SHORTCUTS
     */
    handleKeyboard(event) {
        switch (event.key) {
            case 'F1':
                event.preventDefault();
                this.toggleVoice();
                break;

            case 'F2':
                event.preventDefault();
                this.openSettings();
                break;

            case 'Escape':
                if (document.fullscreenElement) {
                    document.exitFullscreen();
                }
                break;
        }
    }

    /**
     * CURSOR TRACKING
     */
    updateCursor(event) {
        // Move custom cursor
        const cursor = document.querySelector('body::before');
        if (cursor) {
            document.documentElement.style.setProperty('--cursor-x', event.clientX + 'px');
            document.documentElement.style.setProperty('--cursor-y', event.clientY + 'px');
        }
    }

    /**
     * NOTIFICHE SISTEMA
     */
    showNotification(text, level = 'info') {
        if (!this.elements.notifications) return;

        const notification = document.createElement('div');
        notification.className = `notification notification-${level}`;
        notification.innerHTML = `
            <div class="notification-content">
                <span class="notification-text">${text}</span>
                <button class="notification-close">×</button>
            </div>
        `;

        // Add to container
        this.elements.notifications.appendChild(notification);

        // Auto remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 5000);

        // Close button
        notification.querySelector('.notification-close').addEventListener('click', () => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        });

        console.log(`📢 ${level.toUpperCase()}: ${text}`);
    }

    /**
     * LOADING OVERLAY
     */
    hideLoadingOverlay() {
        // Progress animation
        let progress = 0;
        const progressInterval = setInterval(() => {
            progress += 10;
            if (this.elements.loadingProgress) {
                this.elements.loadingProgress.style.width = `${progress}%`;
            }

            if (progress >= 100) {
                clearInterval(progressInterval);

                // Hide overlay
                setTimeout(() => {
                    if (this.elements.loadingOverlay) {
                        this.elements.loadingOverlay.classList.add('hidden');
                    }
                }, 500);
            }
        }, 100);
    }

    /**
     * START UPDATE LOOPS
     */
    startUpdateLoops() {
        // Simulated metrics update (sarà sostituito da real data dal backend)
        setInterval(() => {
            if (this.state.connected) {
                // Request real metrics dal backend
                this.sendWebSocketMessage('request_metrics');
            } else {
                // Fake metrics per demo
                this.updateSystemMetrics({
                    cpu: Math.floor(Math.random() * 30) + 10,
                    memory: Math.floor(Math.random() * 40) + 20,
                    network: Math.floor(Math.random() * 20) + 5,
                    ai: this.state.currentState === 'processing' ? 90 : Math.floor(Math.random() * 30)
                });
            }
        }, this.config.updateInterval);

        console.log('🔄 Update loops started');
    }

    /**
     * HANDLE JARVIS RESPONSE
     */
    handleJarvisResponse(message) {
        this.showNotification(`Jarvis: ${message.text}`, 'success');

        // Set speaking state temporaneamente
        this.setState('speaking');

        // Return to normal after response
        setTimeout(() => {
            this.setState('normal');
        }, message.duration || 3000);
    }
}

/**
 * INIZIALIZZAZIONE APP
 */
console.log('🤖 Loading Jarvis Frontend Application...');

// Aspetta DOM ready e inizializza app
const jarvisApp = new JarvisApp();

// Export globale per debugging
window.jarvis = jarvisApp;