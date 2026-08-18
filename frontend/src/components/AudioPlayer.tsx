import React, { useState, useRef, useEffect } from 'react';

interface AudioPlayerProps {
  audioUrl: string;
}

export const AudioPlayer: React.FC<AudioPlayerProps> = ({ audioUrl }) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const progressIntervalRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    audioRef.current = new Audio(audioUrl);
    
    audioRef.current.onended = () => {
      setIsPlaying(false);
      setProgress(0);
      if (progressIntervalRef.current) {
        clearInterval(progressIntervalRef.current);
      }
    };
    
    audioRef.current.ontimeupdate = () => {
      if (audioRef.current && audioRef.current.duration) {
        setProgress(audioRef.current.currentTime / audioRef.current.duration);
      }
    };
    
    return () => {
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current.src = '';
      }
      if (progressIntervalRef.current) {
        clearInterval(progressIntervalRef.current);
      }
    };
  }, [audioUrl]);

  const togglePlay = () => {
    if (!audioRef.current) return;
    
    if (isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
    } else {
      audioRef.current.play().catch(console.error);
      setIsPlaying(true);
    }
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="audio-player">
      <button
        className={`play-btn ${isPlaying ? 'playing' : ''}`}
        onClick={togglePlay}
        aria-label={isPlaying ? 'Pause' : 'Play'}
      >
        {isPlaying ? '⏸️' : '▶️'}
      </button>
      
      <div className="progress-bar" onClick={(e) => {
        if (audioRef.current) {
          const rect = e.currentTarget.getBoundingClientRect();
          const percent = (e.clientX - rect.left) / rect.width;
          audioRef.current.currentTime = percent * audioRef.current.duration;
        }
      }}>
        <div 
          className="progress-fill" 
          style={{ width: `${progress * 100}%` }}
        />
      </div>
      
      <span className="time-display">
        {audioRef.current ? formatTime(audioRef.current.currentTime) : '0:00'} / 
        {audioRef.current ? formatTime(audioRef.current.duration) : '0:00'}
      </span>
      
      <a 
        href={audioUrl} 
        download={`response_${Date.now()}.mp3`}
        className="download-link"
        aria-label="Download audio"
      >
        📥
      </a>
    </div>
  );
};