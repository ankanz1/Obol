import { useState, useCallback, useRef } from 'react';
import { Message, SUPPORTED_LANGUAGES } from './types';
import { useWebSocket } from './hooks/useWebSocket';
import { VoiceRecorder } from './components/VoiceRecorder';
import { MessageList } from './components/MessageList';
import { LanguageSelector } from './components/LanguageSelector';;

function generateSessionId(): string {
  return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

export default function App() {
  const [sessionId] = useState(generateSessionId);
  const [language, setLanguage] = useState('en');
  const [messages, setMessages] = useState<Message[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [wsConnected, setWsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const pendingAudioRef = useRef<string>('');
  const pendingTranscriptRef = useRef<string>('');

  // WebSocket
  const handleWsMessage = useCallback((data: any) => {
    switch (data.type) {
      case 'result':
        const newMessage: Message = {
          id: data.traceId || Date.now().toString(),
          role: 'user',
          content: data.transcript,
          language: data.language,
          timestamp: new Date(),
          grounded: data.grounded,
          refusalReason: data.refusalReason,
          latencyMs: data.latencyMs
        };
        
        const assistantMessage: Message = {
          id: `${data.traceId}_resp`,
          role: 'assistant',
          content: data.answer,
          language: data.language,
          timestamp: new Date(),
          audioUrl: data.audioBase64 ? `data:audio/wav;base64,${data.audioBase64}` : undefined,
          grounded: data.grounded,
          refusalReason: data.refusalReason,
          latencyMs: data.latencyMs
        };
        
        setMessages(prev => [...prev, newMessage, assistantMessage]);
        setIsProcessing(false);
        pendingAudioRef.current = '';
        pendingTranscriptRef.current = '';
        break;
        
      case 'error':
        setError(data.message);
        setIsProcessing(false);
        break;
        
      case 'config_ack':
        console.log('Config acknowledged');
        break;
    }
  }, []);

  const { send } = useWebSocket({
    sessionId,
    onMessage: handleWsMessage,
    onOpen: () => setWsConnected(true),
    onClose: () => setWsConnected(false),
    onError: () => setError('Connection error')
  });

  const handleAudioData = useCallback((audioBase64: string) => {
    send({
      type: 'audio',
      audio_base64: audioBase64,
      language
    });
    setIsProcessing(true);
  }, [send, language]);

  const handleLanguageChange = (newLang: string) => {
    setLanguage(newLang);
    send({ type: 'config', language: newLang });
  };

  const clearHistory = () => {
    setMessages([]);
    send({ type: 'clear_history' });
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>🎙️ Voice RAG</h1>
        <p className="subtitle">MSMARCO-XI • 18 Indic Languages</p>
        <div className="status-bar">
          <span className={`connection-status ${wsConnected ? 'connected' : 'disconnected'}`}>
            {wsConnected ? '🟢 Connected' : '🔴 Disconnected'}
          </span>
          <LanguageSelector value={language} onChange={handleLanguageChange} disabled={isProcessing} />
        </div>
      </header>

      <main className="app-main">
        <div className="chat-panel">
          <MessageList messages={messages} currentLanguage={language} />
        </div>

        <div className="recorder-panel">
          <VoiceRecorder
            isProcessing={isProcessing}
            language={language}
            disabled={!wsConnected}
            onAudioData={handleAudioData}
          />
          
          {error && (
            <div className="error-toast" role="alert">
              <span>⚠️ {error}</span>
              <button onClick={() => setError(null)}>✕</button>
            </div>
          )}
        </div>
      </main>

      <footer className="app-footer">
        <button onClick={clearHistory} disabled={messages.length === 0}>
          🗑️ Clear History
        </button>
        <div className="languages-info">
          Supported: {SUPPORTED_LANGUAGES.map(l => l.code.toUpperCase()).join(', ')}
        </div>
      </footer>
    </div>
  );
};