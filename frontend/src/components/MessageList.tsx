import React from 'react';
import { Message } from '../types';
import { AudioPlayer } from './AudioPlayer';

interface MessageListProps {
  messages: Message[];
  currentLanguage: string;
}

export const MessageList: React.FC<MessageListProps> = ({ messages, currentLanguage }) => {
  if (messages.length === 0) {
    return (
      <div className="empty-state">
        <div className="empty-icon">🎙️</div>
        <p>Start by asking a question in {currentLanguage.toUpperCase()}</p>
        <p className="hint">Press Space or click the microphone button</p>
      </div>
    );
  }

  return (
    <div className="message-list" role="log" aria-live="polite">
      {messages.map((msg) => (
        <div key={msg.id} className={`message ${msg.role}`}>
          <div className="message-header">
            <span className="message-role">
              {msg.role === 'user' ? '👤 You' : '🤖 Assistant'}
            </span>
            <span className="message-language">{msg.language.toUpperCase()}</span>
            {msg.latencyMs && (
              <span className="message-latency">{msg.latencyMs.toFixed(0)}ms</span>
            )}
          </div>
          
          <div className="message-content">
            <p>{msg.content}</p>
            
            {msg.grounded !== undefined && (
              <div className="message-meta">
                <span className={`grounding-badge ${msg.grounded ? 'grounded' : 'ungrounded'}`}>
                  {msg.grounded ? '✅ Grounded' : '⚠️ Not Grounded'}
                </span>
                {msg.refusalReason && (
                  <span className="refusal-badge">
                    Refused: {msg.refusalReason}
                  </span>
                )}
              </div>
            )}
          </div>
          
          {msg.audioUrl && (
            <AudioPlayer audioUrl={msg.audioUrl} />
          )}
        </div>
      ))}
    </div>
  );
};