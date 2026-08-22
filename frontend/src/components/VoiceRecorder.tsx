import React from 'react';
import { useAudioRecorder } from '../hooks/useAudioRecorder';
import { WaveformVisualizer } from './WaveformVisualizer';

interface VoiceRecorderProps {
  isProcessing: boolean;
  language: string;
  disabled?: boolean;
}

export const VoiceRecorder: React.FC<VoiceRecorderProps> = ({
  isProcessing,
  language,
  disabled = false
}) => {
  const { isRecording, audioLevel, toggleRecording } = useAudioRecorder({
    onData: () => {},
    onStart: () => console.log('Recording started'),
    onStop: () => console.log('Recording stopped'),
    onError: (err) => console.error('Recording error:', err)
  });

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.code === 'Space' && !e.repeat && !isProcessing && !disabled) {
      e.preventDefault();
      toggleRecording();
    }
  };

  return (
    <div className="voice-recorder" onKeyDown={handleKeyDown} tabIndex={0}>
      <div className="recorder-header">
        <span className="language-badge">{language.toUpperCase()}</span>
        <span className={`status-badge ${isRecording ? 'recording' : isProcessing ? 'processing' : 'idle'}`}>
          {isRecording ? '🎤 Recording...' : isProcessing ? '⚙️ Processing...' : '🎙️ Ready'}
        </span>
      </div>
      
      <WaveformVisualizer
        audioLevel={audioLevel}
        isRecording={isRecording}
        isProcessing={isProcessing}
        width={400}
        height={120}
      />
      
      <div className="recorder-controls">
        <button
          className={`record-btn ${isRecording ? 'recording' : ''} ${isProcessing ? 'disabled' : ''}`}
          onClick={toggleRecording}
          disabled={isProcessing || disabled}
          aria-label={isRecording ? 'Stop recording' : 'Start recording'}
          aria-pressed={isRecording}
        >
          <span className="btn-icon">{isRecording ? '⏹️' : '🎤'}</span>
          <span className="btn-text">{isRecording ? 'Stop' : 'Hold to Speak'}</span>
          <span className="hint">(Spacebar)</span>
        </button>
        
        {isRecording && (
          <div className="recording-timer">
            <span className="pulse">●</span> Recording... Release to send
          </div>
        )}
        
        {isProcessing && (
          <div className="processing-indicator">
            <div className="spinner"></div>
            <span>Processing your query...</span>
          </div>
        )}
      </div>
      
      <div className="recorder-help">
        <kbd>Space</kbd> to record • Speak in <strong>{language.toUpperCase()}</strong> • 
        Max 30 seconds per query
      </div>
    </div>
  );
};