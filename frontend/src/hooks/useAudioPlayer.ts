import { useState, useCallback, useRef, useEffect } from 'react';

export function useAudioPlayer() {
  const [isPlaying, setIsPlaying] = useState(false);
  const audioContextRef = useRef<AudioContext | null>(null);
  const sourceRef = useRef<AudioBufferSourceNode | null>(null);
  const queueRef = useRef<AudioBuffer[]>([]);
  const playingRef = useRef(false);

  const initAudioContext = useCallback(() => {
    if (!audioContextRef.current) {
      audioContextRef.current = new AudioContext({ sampleRate: 16000 });
    }
    if (audioContextRef.current.state === 'suspended') {
      audioContextRef.current.resume();
    }
    return audioContextRef.current;
  }, []);

  const playAudioBase64 = useCallback(async (base64: string) => {
    if (!base64) return;
    
    try {
      const audioContext = initAudioContext();
      
      // Decode base64 to array buffer
      const binary = atob(base64);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) {
        bytes[i] = binary.charCodeAt(i);
      }
      
      // Decode audio data
      const audioBuffer = await audioContext.decodeAudioData(bytes.buffer);
      
      // Play immediately
      const source = audioContext.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(audioContext.destination);
      sourceRef.current = source;
      
      setIsPlaying(true);
      playingRef.current = true;
      
      source.onended = () => {
        if (playingRef.current) {
          setIsPlaying(false);
        }
      };
      
      source.start(0);
      
    } catch (err) {
      console.error('Failed to play audio:', err);
      setIsPlaying(false);
    }
  }, [initAudioContext]);

  const stop = useCallback(() => {
    if (sourceRef.current) {
      sourceRef.current.stop();
      sourceRef.current = null;
    }
    playingRef.current = false;
    setIsPlaying(false);
  }, []);

  useEffect(() => {
    return () => {
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
    };
  }, []);

  return { isPlaying, playAudioBase64, stop };
}