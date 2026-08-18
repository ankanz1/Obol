import React, { useEffect, useRef } from 'react';

interface WaveformVisualizerProps {
  audioLevel: number;
  isRecording: boolean;
  isProcessing: boolean;
  width?: number;
  height?: number;
}

export const WaveformVisualizer: React.FC<WaveformVisualizerProps> = ({
  audioLevel,
  isRecording,
  isProcessing,
  width = 300,
  height = 100
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number | null>(null);
  const barsRef = useRef<number[]>(Array(30).fill(0));

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = width;
    canvas.height = height;

    const draw = () => {
      if (!ctx) return;
      
      ctx.clearRect(0, 0, width, height);
      
      const centerY = height / 2;
      const barWidth = width / barsRef.current.length;
      const maxBarHeight = height * 0.4;
      
      // Update bars
      if (isRecording && audioLevel > 0) {
        // Shift bars and add new level
        barsRef.current.shift();
        barsRef.current.push(audioLevel * maxBarHeight + Math.random() * 10);
      } else if (isProcessing) {
        // Animated processing bars
        barsRef.current = barsRef.current.map((_, i) => 
          Math.abs(Math.sin(Date.now() / 200 + i * 0.5)) * maxBarHeight * 0.6
        );
      } else {
        // Decay to zero
        barsRef.current = barsRef.current.map(h => h * 0.9);
      }
      
      // Draw bars
      barsRef.current.forEach((barHeight, i) => {
        const x = i * barWidth + 2;
        const y = centerY - barHeight / 2;
        
        // Color based on state
        let color = '#4ade80'; // Green for recording
        if (isProcessing) color = '#fbbf24'; // Yellow for processing
        if (!isRecording && !isProcessing) color = '#94a3b8'; // Gray for idle
        
        const gradient = ctx.createLinearGradient(0, y, 0, y + barHeight);
        gradient.addColorStop(0, color);
        gradient.addColorStop(1, color + '80');
        
        ctx.fillStyle = gradient;
        ctx.fillRect(x, y, barWidth - 4, barHeight);
      });
      
      animationRef.current = requestAnimationFrame(draw);
    };

    draw();

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [width, height, audioLevel, isRecording, isProcessing]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      className="waveform-canvas"
      aria-label={isRecording ? "Recording audio visualization" : isProcessing ? "Processing audio" : "Audio visualizer idle"}
    />
  );
};