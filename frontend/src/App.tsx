import React, { useState, useCallback, useEffect, useRef } from 'react';
import { Message, PipelineResult, SUPPORTED_LANGUAGES } from './types';
import { useWebSocket } from './hooks/useWebSocket';
import { useAudioRecorder } from './hooks/useAudioRecorder';
import { useAudioPlayer } from './hooks/useAudioPlayer';
import { VoiceRecorder } from './components/VoiceRecorder';
import { MessageList } from './components/MessageList';
import { LanguageSelector } from './components/LanguageSelector';
import { WaveformVisualizer } from './components/WaveformVisualizer';

const WS_URL = 'ws://localhost:8000/ws/';

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
          audioUrl: data.audio_base64 ? `data:audio/wav;base64,${data.audio_base64}` : undefined,
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

  // Use text API for testing (bypasses STT/TTS which need different API keys)
const useTextApi = true;

const { connected, send, disconnect } = useWebSocket({
    sessionId,
    onMessage: handleWsMessage,
    onOpen: () => setWsConnected(true),
    onClose: () => setWsConnected(false),
    onError: () => setError('Connection error')
  });

  // Audio recorder - we'll use REST for simplicity in demo
  const { isRecording, audioLevel, startRecording, stopRecording, toggleRecording } = useAudioRecorder({
    onData: async (audioBase64) => {
      pendingAudioRef.current = audioBase64;
    },
    onStart: () => {},
    onStop: async () => {
      if (pendingAudioRef.current && !isProcessing) {
        setIsProcessing(true);
        setError(null);
        
        // For testing: use text API with a mock transcript
        // In production, this would use the audio API
        const mockTranscript = "what is the capital of india";
        
        try {
          const response = await fetch('http://localhost:8000/api/query/text', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              query: mockTranscript,
              language,
              session_id: sessionId
            })
          });
          
          const result: PipelineResult = await response.json();
          
          const userMsg: Message = {
            id: result.traceId || Date.now().toString(),
            role: 'user',
            content: result.transcript,
            language: result.language,
            timestamp: new Date(),
            grounded: result.grounded,
            refusalReason: result.refusalReason,
            latencyMs: result.latencyMs
          };
          
          const assistantMsg: Message = {
            id: `${result.traceId}_resp`,
            role: 'assistant',
            content: result.answer,
            language: result.language,
            timestamp: new Date(),
            audioUrl: result.audio_base64 ? `data:audio/wav;base64,${result.audio_base64}` : undefined,
            grounded: result.grounded,
            refusalReason: result.refusalReason,
            latencyMs: result.latencyMs
          };
          
          setMessages(prev => [...prev, userMsg, assistantMsg]);
        } catch (err) {
          console.error('Query failed:', err);
          setError('Failed to process query');
        } finally {
          setIsProcessing(false);
          pendingAudioRef.current = '';
        }
      }
    },
    onError: (err) => setError(err.message)
  });

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
            onTranscript={() => {}}
            onAudioData={() => {}}
            isProcessing={isProcessing}
            language={language}
            disabled={!wsConnected}
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