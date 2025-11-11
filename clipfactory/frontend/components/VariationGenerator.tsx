/**
 * Variation Generator UI Component
 *
 * Features:
 * - Voice dictation for titles
 * - Dual-arrow music swiper
 * - Title style preview (TT3 vs AdLab)
 * - Real-time preview
 * - Processing animation
 */
import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FiMic, FiChevronLeft, FiChevronRight, FiPlay, FiCheck, FiAlertCircle, FiX } from 'react-icons/fi';

interface MusicTrack {
  id: string;
  name: string;
  vibe: string;
  context: string;
  color: string;
}

interface TitleVariant {
  text: string;
  variant_id: string;
  hook_style: string;
  predicted_ctr: number;
}

interface VariationGeneratorProps {
  clipId: string;
  onGenerate: (config: any) => void;
}

export default function VariationGenerator({ clipId, onGenerate }: VariationGeneratorProps) {
  // State
  const [isRecording, setIsRecording] = useState(false);
  const [voiceTranscript, setVoiceTranscript] = useState('');
  const [titleVariants, setTitleVariants] = useState<TitleVariant[]>([]);
  const [selectedTitleStyle, setSelectedTitleStyle] = useState<'TT3' | 'AdLab'>('TT3');
  const [musicTracks, setMusicTracks] = useState<MusicTrack[]>([]);
  const [currentMusicIndex, setCurrentMusicIndex] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string>('');
  const progressIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Cleanup interval on unmount
  useEffect(() => {
    return () => {
      if (progressIntervalRef.current) {
        clearInterval(progressIntervalRef.current);
        progressIntervalRef.current = null;
      }
    };
  }, []);

  // Load music tracks
  useEffect(() => {
    loadMusicTracks();
  }, []);

  const loadMusicTracks = async () => {
    try {
      const response = await fetch('/api/music/list');
      if (!response.ok) {
        throw new Error('Failed to load music tracks');
      }
      const data = await response.json();
      setMusicTracks(data.tracks);
    } catch (error: any) {
      const errorMessage = error.message || 'Failed to load music tracks. Please refresh the page.';
      setError(errorMessage);
    }
  };

  // Voice recording
  const startRecording = () => {
    setIsRecording(true);
    // TODO: Integrate Web Speech API or Whisper
    // For now, simulate
    setTimeout(() => {
      setVoiceTranscript("Sample title from voice input");
      setIsRecording(false);
      generateTitleVariants("Sample title from voice input");
    }, 2000);
  };

  const stopRecording = () => {
    setIsRecording(false);
  };

  // Generate title variants from voice
  const generateTitleVariants = async (transcript: string) => {
    try {
      const response = await fetch('/api/phase4/generate-titles', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          transcript,
          hook_score: 7.5,
          duration: 30,
          num_variants: 5
        })
      });

      if (!response.ok) {
        throw new Error('Failed to generate title variants');
      }

      const data = await response.json();
      setTitleVariants(data.titles);
    } catch (error: any) {
      const errorMessage = error.message || 'Failed to generate title variants. Please try again.';
      setError(errorMessage);
    }
  };

  // Music navigation
  const previousMusic = () => {
    setCurrentMusicIndex((prev) =>
      prev > 0 ? prev - 1 : musicTracks.length - 1
    );
  };

  const nextMusic = () => {
    setCurrentMusicIndex((prev) =>
      prev < musicTracks.length - 1 ? prev + 1 : 0
    );
  };

  // Generate variations
  const handleGenerate = async () => {
    setError(''); // Clear previous errors
    setIsProcessing(true);
    setProgress(0);

    const config = {
      clip_id: clipId,
      temporal_variations: ['base', '+4s', '+35s'],
      reframe_styles: ['original', 'flipped', 'blurry_bg'],
      title_style: selectedTitleStyle,
      music_id: musicTracks[currentMusicIndex]?.id,
      selected_title: titleVariants[0]?.text
    };

    // Clear any existing interval
    if (progressIntervalRef.current) {
      clearInterval(progressIntervalRef.current);
    }

    // Simulate progress
    progressIntervalRef.current = setInterval(() => {
      setProgress(prev => {
        if (prev >= 100) {
          if (progressIntervalRef.current) {
            clearInterval(progressIntervalRef.current);
            progressIntervalRef.current = null;
          }
          return 100;
        }
        return prev + 5;
      });
    }, 200);

    try {
      await onGenerate(config);

      // Complete
      if (progressIntervalRef.current) {
        clearInterval(progressIntervalRef.current);
        progressIntervalRef.current = null;
      }
      setProgress(100);

      // Play sound notification
      playNotificationSound();

    } catch (error: any) {
      const errorMessage = error.message || 'Generation failed. Please try again.';
      setError(errorMessage);
      setIsProcessing(false);
      if (progressIntervalRef.current) {
        clearInterval(progressIntervalRef.current);
        progressIntervalRef.current = null;
      }
    }
  };

  const playNotificationSound = () => {
    // TODO: Play notification sound
    const audio = new Audio('/sounds/done.mp3');
    audio.play().catch(() => {});
  };

  return (
    <div className="w-full max-w-4xl mx-auto p-6 space-y-8">
      {/* Header */}
      <div className="text-center">
        <h2 className="text-3xl font-bold text-gray-900">Variation Generator</h2>
        <p className="text-gray-600 mt-2">Create 9 variations from this clip</p>
      </div>

      {/* Error Display */}
      {error && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-red-50 border border-red-200 rounded-lg p-4"
        >
          <div className="flex items-start space-x-3">
            <FiAlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
            <div className="flex-1">
              <h4 className="font-semibold text-red-900">Error</h4>
              <p className="text-sm text-red-700 mt-1">{error}</p>
            </div>
            <button
              onClick={() => setError('')}
              className="text-red-600 hover:text-red-800"
              aria-label="Dismiss error message"
            >
              <FiX className="w-5 h-5" />
            </button>
          </div>
        </motion.div>
      )}

      {/* Voice Input */}
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h3 className="text-xl font-semibold mb-4">Title Input (Voice)</h3>

        <div className="flex items-center space-x-4">
          <button
            onClick={isRecording ? stopRecording : startRecording}
            aria-label={isRecording ? 'Stop voice recording' : 'Start voice recording'}
            className={`flex items-center space-x-2 px-6 py-3 rounded-lg font-semibold transition-all ${
              isRecording
                ? 'bg-red-500 hover:bg-red-600 text-white animate-pulse'
                : 'bg-blue-500 hover:bg-blue-600 text-white'
            }`}
          >
            <FiMic className="w-5 h-5" />
            <span>{isRecording ? 'Recording...' : 'Start Voice Input'}</span>
          </button>

          {voiceTranscript && (
            <div className="flex-1 text-gray-700 italic">
              "{voiceTranscript}"
            </div>
          )}
        </div>

        {/* Title Variants */}
        {titleVariants.length > 0 && (
          <div className="mt-6 space-y-2">
            <p className="text-sm text-gray-600 font-medium">Generated Variants:</p>
            {titleVariants.map((variant, idx) => (
              <div
                key={idx}
                className="p-3 bg-gray-50 rounded border border-gray-200 hover:bg-gray-100 cursor-pointer"
              >
                <div className="flex justify-between items-start">
                  <div className="flex-1">
                    <span className="font-medium text-gray-900">{variant.variant_id}:</span>
                    <span className="ml-2 text-gray-800">{variant.text}</span>
                  </div>
                  <div className="text-sm text-gray-500">
                    CTR: {(variant.predicted_ctr * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="mt-1 text-xs text-gray-500">
                  Style: {variant.hook_style}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Title Style Selection */}
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h3 className="text-xl font-semibold mb-4">Title Style</h3>

        <div className="grid grid-cols-2 gap-4">
          <button
            onClick={() => setSelectedTitleStyle('TT3')}
            aria-label="Select TikTok Cubed title style"
            aria-pressed={selectedTitleStyle === 'TT3'}
            className={`p-4 rounded-lg border-2 transition-all ${
              selectedTitleStyle === 'TT3'
                ? 'border-blue-500 bg-blue-50'
                : 'border-gray-200 hover:border-gray-300'
            }`}
          >
            <div className="font-semibold text-lg mb-2">TT³ (TikTok Cubed)</div>
            <div className="text-sm text-gray-600">Top bubble, black text</div>
            <div className="mt-3 bg-gray-900 text-white px-4 py-2 rounded-full text-center">
              Sample TT³ Title
            </div>
          </button>

          <button
            onClick={() => setSelectedTitleStyle('AdLab')}
            aria-label="Select AdLab Standard title style"
            aria-pressed={selectedTitleStyle === 'AdLab'}
            className={`p-4 rounded-lg border-2 transition-all ${
              selectedTitleStyle === 'AdLab'
                ? 'border-blue-500 bg-blue-50'
                : 'border-gray-200 hover:border-gray-300'
            }`}
          >
            <div className="font-semibold text-lg mb-2">AdLab Standard</div>
            <div className="text-sm text-gray-600">Bottom center, stroke</div>
            <div className="mt-3 bg-transparent text-black font-bold px-4 py-2 border-2 border-black rounded text-center">
              Sample AdLab Title
            </div>
          </button>
        </div>
      </div>

      {/* Music Selector */}
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h3 className="text-xl font-semibold mb-4">Music Selection</h3>

        {musicTracks.length > 0 && (
          <div className="flex items-center space-x-4">
            <button
              onClick={previousMusic}
              aria-label="Previous music track"
              className="p-3 rounded-full bg-gray-100 hover:bg-gray-200 transition-colors"
            >
              <FiChevronLeft className="w-6 h-6" />
            </button>

            <div className="flex-1">
              <motion.div
                key={currentMusicIndex}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                role="region"
                aria-label={`Music track: ${musicTracks[currentMusicIndex]?.name}`}
                className={`p-6 rounded-lg border-2`}
                style={{
                  backgroundColor: musicTracks[currentMusicIndex]?.color + '20',
                  borderColor: musicTracks[currentMusicIndex]?.color
                }}
              >
                <div className="font-bold text-xl mb-2">
                  {musicTracks[currentMusicIndex]?.name}
                </div>
                <div className="text-sm text-gray-700 mb-1">
                  Vibe: {musicTracks[currentMusicIndex]?.vibe}
                </div>
                <div className="text-sm text-gray-600">
                  {musicTracks[currentMusicIndex]?.context}
                </div>
              </motion.div>
            </div>

            <button
              onClick={nextMusic}
              aria-label="Next music track"
              className="p-3 rounded-full bg-gray-100 hover:bg-gray-200 transition-colors"
            >
              <FiChevronRight className="w-6 h-6" />
            </button>
          </div>
        )}

        <div className="mt-4 text-center text-sm text-gray-500">
          Track {currentMusicIndex + 1} of {musicTracks.length}
        </div>
      </div>

      {/* Generate Button */}
      <div className="bg-white rounded-lg shadow-lg p-6">
        <button
          onClick={handleGenerate}
          disabled={isProcessing || titleVariants.length === 0}
          aria-label="Generate 9 video variations"
          className={`w-full py-4 rounded-lg font-bold text-lg transition-all ${
            isProcessing || titleVariants.length === 0
              ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
              : 'bg-green-500 hover:bg-green-600 text-white'
          }`}
        >
          {isProcessing ? (
            <span className="flex items-center justify-center space-x-2">
              <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
              </svg>
              <span>Generating... {progress}%</span>
            </span>
          ) : (
            <span className="flex items-center justify-center space-x-2">
              <FiPlay className="w-5 h-5" />
              <span>Generate 9 Variations</span>
            </span>
          )}
        </button>

        {/* Progress Bar */}
        {isProcessing && (
          <div className="mt-4">
            <div className="w-full bg-gray-200 rounded-full h-2">
              <motion.div
                className="bg-green-500 h-2 rounded-full"
                initial={{ width: 0 }}
                animate={{ width: `${progress}%` }}
                transition={{ duration: 0.3 }}
              />
            </div>
          </div>
        )}

        {/* Processing Animation */}
        <AnimatePresence>
          {isProcessing && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="mt-6 p-4 bg-blue-50 rounded-lg border border-blue-200"
            >
              <div className="flex items-center space-x-3">
                <div className="relative w-16 h-16">
                  {/* Pixelated animation effect */}
                  <div className="absolute inset-0 bg-gradient-to-r from-blue-400 to-purple-500 rounded animate-pulse" />
                </div>
                <div>
                  <div className="font-semibold text-gray-900">Processing Matrix...</div>
                  <div className="text-sm text-gray-600">Creating reframes and variations</div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Success */}
        {progress === 100 && (
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="mt-6 p-4 bg-green-50 rounded-lg border border-green-200 flex items-center space-x-3"
          >
            <FiCheck className="w-6 h-6 text-green-600" />
            <div>
              <div className="font-semibold text-green-900">Complete!</div>
              <div className="text-sm text-green-700">9 variations generated successfully</div>
            </div>
          </motion.div>
        )}
      </div>
    </div>
  );
}
