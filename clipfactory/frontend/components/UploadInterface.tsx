/**
 * Upload Interface for Phase 2
 * - Upload video for council
 * - Export to Premiere XML
 * - Re-upload edited clips
 */
import React, { useState, useCallback, useEffect, useRef } from 'react';
import { useDropzone } from 'react-dropzone';
import { FiUpload, FiDownload, FiCheck, FiX, FiAlertCircle } from 'react-icons/fi';
import { motion } from 'framer-motion';
import axios from 'axios';

interface UploadInterfaceProps {
  onUploadComplete?: (videoId: string) => void;
}

const MAX_FILE_SIZE = 5 * 1024 * 1024 * 1024; // 5GB
const ACCEPTED_VIDEO_TYPES = ['video/mp4', 'video/quicktime', 'video/x-matroska', 'video/x-msvideo'];

export default function UploadInterface({ onUploadComplete }: UploadInterfaceProps) {
  const [uploadedVideo, setUploadedVideo] = useState<any>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [councilStatus, setCouncilStatus] = useState<'idle' | 'processing' | 'complete'>('idle');
  const [clipsFound, setClipsFound] = useState(0);
  const [error, setError] = useState<string>('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Cleanup polling interval on unmount
  useEffect(() => {
    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
        pollingIntervalRef.current = null;
      }
    };
  }, []);

  // Format file size for display
  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
  };

  // Dropzone for video upload
  const onDrop = useCallback(async (acceptedFiles: File[], rejectedFiles: any[]) => {
    setError(''); // Clear previous errors

    // Handle rejected files
    if (rejectedFiles.length > 0) {
      const rejection = rejectedFiles[0];
      if (rejection.errors[0]?.code === 'file-too-large') {
        setError(`File is too large. Maximum size is ${formatFileSize(MAX_FILE_SIZE)}`);
      } else if (rejection.errors[0]?.code === 'file-invalid-type') {
        setError('Invalid file type. Please upload MP4, MOV, MKV, or AVI files.');
      } else {
        setError('File upload rejected. Please try again.');
      }
      return;
    }

    const file = acceptedFiles[0];
    if (!file) return;

    // Validate file type
    if (!ACCEPTED_VIDEO_TYPES.includes(file.type)) {
      setError('Invalid file type. Please upload MP4, MOV, MKV, or AVI files.');
      return;
    }

    // Validate file size
    if (file.size > MAX_FILE_SIZE) {
      setError(`File is too large (${formatFileSize(file.size)}). Maximum size is ${formatFileSize(MAX_FILE_SIZE)}`);
      return;
    }

    setSelectedFile(file);
    setIsUploading(true);
    setUploadProgress(0);

    const formData = new FormData();
    formData.append('video', file);

    try {
      const response = await axios.post('/api/phase1/upload', formData, {
        onUploadProgress: (progressEvent) => {
          const progress = progressEvent.total
            ? Math.round((progressEvent.loaded * 100) / progressEvent.total)
            : 0;
          setUploadProgress(progress);
        },
      });

      setUploadedVideo(response.data);
      setCouncilStatus('processing');

      // Poll for council status
      pollCouncilStatus(response.data.video_id);

      onUploadComplete?.(response.data.video_id);
    } catch (error: any) {
      const errorMessage = error.response?.data?.message || error.message || 'Upload failed. Please try again.';
      setError(errorMessage);
      setIsUploading(false);
    }
  }, [onUploadComplete]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'video/*': ['.mp4', '.mov', '.mkv', '.avi']
    },
    maxFiles: 1,
    maxSize: MAX_FILE_SIZE,
  });

  // Poll council deliberation status
  const pollCouncilStatus = async (videoId: string) => {
    // Clear any existing interval
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current);
    }

    pollingIntervalRef.current = setInterval(async () => {
      try {
        const response = await axios.get(`/api/phase1/status/${videoId}`);
        const { status, clips_found, progress } = response.data;

        setClipsFound(clips_found);

        if (status === 'complete') {
          setCouncilStatus('complete');
          setIsUploading(false);
          if (pollingIntervalRef.current) {
            clearInterval(pollingIntervalRef.current);
            pollingIntervalRef.current = null;
          }
        }
      } catch (error: any) {
        const errorMessage = error.response?.data?.message || 'Failed to check processing status';
        setError(errorMessage);
        if (pollingIntervalRef.current) {
          clearInterval(pollingIntervalRef.current);
          pollingIntervalRef.current = null;
        }
      }
    }, 5000); // Check every 5 seconds
  };

  // Export to Premiere XML
  const exportToXML = async () => {
    if (!uploadedVideo) return;

    try {
      const response = await axios.post(
        `/api/phase2/export-xml/${uploadedVideo.video_id}`
      );

      // Download XML file
      const downloadUrl = response.data.download_url;
      window.location.href = downloadUrl;
    } catch (error: any) {
      const errorMessage = error.response?.data?.message || 'XML export failed. Please try again.';
      setError(errorMessage);
    }
  };

  return (
    <div className="w-full max-w-4xl mx-auto p-6 space-y-8">
      {/* Header */}
      <div className="text-center">
        <h2 className="text-3xl font-bold text-gray-900">Upload & Process</h2>
        <p className="text-gray-600 mt-2">Phase 1: Council Deliberation</p>
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

      {/* Upload Area */}
      <div className="bg-white rounded-lg shadow-lg p-6">
        <div
          {...getRootProps()}
          role="button"
          aria-label="Upload video file dropzone"
          className={`border-2 border-dashed rounded-lg p-12 text-center cursor-pointer transition-colors ${
            isDragActive
              ? 'border-blue-500 bg-blue-50'
              : 'border-gray-300 hover:border-gray-400'
          }`}
        >
          <input {...getInputProps()} aria-label="Video file input" />

          <FiUpload className="w-16 h-16 mx-auto text-gray-400 mb-4" />

          {isDragActive ? (
            <p className="text-lg text-blue-600 font-semibold">Drop video here...</p>
          ) : (
            <div>
              <p className="text-lg text-gray-700 font-semibold mb-2">
                Drop video here or click to browse
              </p>
              <p className="text-sm text-gray-500">
                Supports MP4, MOV, MKV, AVI • Max {formatFileSize(MAX_FILE_SIZE)}
              </p>
            </div>
          )}
        </div>

        {/* Selected File Info */}
        {selectedFile && !uploadedVideo && (
          <div className="mt-4 p-3 bg-gray-50 rounded-lg">
            <div className="flex justify-between text-sm">
              <span className="text-gray-700 font-medium">{selectedFile.name}</span>
              <span className="text-gray-500">{formatFileSize(selectedFile.size)}</span>
            </div>
          </div>
        )}

        {/* Upload Progress */}
        {isUploading && (
          <div className="mt-6">
            <div className="flex justify-between text-sm text-gray-600 mb-2">
              <span>Uploading...</span>
              <span>{uploadProgress}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <motion.div
                className="bg-blue-500 h-2 rounded-full"
                initial={{ width: 0 }}
                animate={{ width: `${uploadProgress}%` }}
                transition={{ duration: 0.3 }}
              />
            </div>
          </div>
        )}

        {/* Uploaded Video Info */}
        {uploadedVideo && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-6 p-4 bg-green-50 rounded-lg border border-green-200"
          >
            <div className="flex items-center space-x-3">
              <FiCheck className="w-6 h-6 text-green-600" />
              <div>
                <div className="font-semibold text-green-900">Upload Complete</div>
                <div className="text-sm text-green-700">{uploadedVideo.filename}</div>
              </div>
            </div>
          </motion.div>
        )}
      </div>

      {/* Council Status */}
      {councilStatus !== 'idle' && (
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h3 className="text-xl font-semibold mb-4">Council Deliberation</h3>

          {councilStatus === 'processing' && (
            <div className="space-y-4">
              <div className="flex items-center space-x-3">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500" />
                <div>
                  <div className="font-medium">AI Council is analyzing...</div>
                  <div className="text-sm text-gray-600">This may take 10-20 minutes</div>
                </div>
              </div>

              <div className="text-sm text-gray-600">
                Clips found so far: <span className="font-bold">{clipsFound}</span>
              </div>

              {/* Animated visualization */}
              <div className="grid grid-cols-3 gap-4 p-4 bg-gray-50 rounded-lg">
                {[1, 2, 3, 4, 5, 6].map((i) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0.3 }}
                    animate={{ opacity: [0.3, 1, 0.3] }}
                    transition={{
                      duration: 2,
                      repeat: Infinity,
                      delay: i * 0.2
                    }}
                    className="h-2 bg-blue-500 rounded"
                  />
                ))}
              </div>
            </div>
          )}

          {councilStatus === 'complete' && (
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
            >
              <div className="flex items-center space-x-3 mb-6">
                <FiCheck className="w-8 h-8 text-green-600" />
                <div>
                  <div className="text-xl font-bold text-green-900">Council Complete!</div>
                  <div className="text-gray-600">{clipsFound} clips selected</div>
                </div>
              </div>

              <button
                onClick={exportToXML}
                aria-label="Export video clips to Premiere Pro XML format"
                className="w-full flex items-center justify-center space-x-2 px-6 py-3 bg-purple-500 hover:bg-purple-600 text-white font-semibold rounded-lg transition-colors"
              >
                <FiDownload className="w-5 h-5" />
                <span>Export to Premiere Pro XML</span>
              </button>
            </motion.div>
          )}
        </div>
      )}

      {/* Re-upload Section */}
      {councilStatus === 'complete' && (
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h3 className="text-xl font-semibold mb-4">Phase 2: Re-upload Edited Clips</h3>

          <p className="text-gray-600 mb-4">
            After editing in Premiere Pro, drag and drop your edited clips here.
          </p>

          <ReuploadInterface videoId={uploadedVideo.video_id} />
        </div>
      )}
    </div>
  );
}

// Re-upload Component
function ReuploadInterface({ videoId }: { videoId: string }) {
  const [uploadedClips, setUploadedClips] = useState<File[]>([]);
  const [error, setError] = useState<string>('');
  const [isUploading, setIsUploading] = useState(false);

  const onDrop = useCallback((acceptedFiles: File[], rejectedFiles: any[]) => {
    setError(''); // Clear previous errors

    // Handle rejected files
    if (rejectedFiles.length > 0) {
      setError('Some files were rejected. Please upload only MP4 or MOV files.');
      return;
    }

    setUploadedClips(prev => [...prev, ...acceptedFiles]);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'video/*': ['.mp4', '.mov']
    },
    multiple: true,
  });

  const handleReupload = async () => {
    if (uploadedClips.length === 0) {
      setError('Please select at least one clip to upload');
      return;
    }

    setIsUploading(true);
    setError('');

    const formData = new FormData();
    uploadedClips.forEach(file => {
      formData.append('clips', file);
    });

    try {
      const response = await axios.post(
        `/api/phase2/reupload?video_id=${videoId}`,
        formData
      );

      // Success - could navigate or show success message
      console.log('Reupload complete:', response.data);
      setIsUploading(false);
    } catch (error: any) {
      const errorMessage = error.response?.data?.message || 'Reupload failed. Please try again.';
      setError(errorMessage);
      setIsUploading(false);
    }
  };

  return (
    <div>
      {/* Error Display */}
      {error && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-red-50 border border-red-200 rounded-lg p-3 mb-4"
        >
          <div className="flex items-start space-x-2">
            <FiAlertCircle className="w-4 h-4 text-red-600 flex-shrink-0 mt-0.5" />
            <div className="flex-1">
              <p className="text-sm text-red-700">{error}</p>
            </div>
            <button
              onClick={() => setError('')}
              className="text-red-600 hover:text-red-800"
              aria-label="Dismiss error message"
            >
              <FiX className="w-4 h-4" />
            </button>
          </div>
        </motion.div>
      )}

      <div
        {...getRootProps()}
        role="button"
        aria-label="Upload edited clips dropzone"
        className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer mb-4 ${
          isDragActive ? 'border-purple-500 bg-purple-50' : 'border-gray-300'
        }`}
      >
        <input {...getInputProps()} aria-label="Edited clips file input" />
        <p className="text-gray-700">
          {isDragActive ? 'Drop edited clips here...' : 'Drag & drop edited clips (multiple files OK)'}
        </p>
      </div>

      {uploadedClips.length > 0 && (
        <div className="space-y-2 mb-4">
          {uploadedClips.map((file, idx) => (
            <div key={idx} className="flex items-center justify-between p-2 bg-gray-50 rounded">
              <span className="text-sm">{file.name}</span>
              <button
                onClick={() => setUploadedClips(prev => prev.filter((_, i) => i !== idx))}
                className="text-red-500 hover:text-red-700"
                aria-label={`Remove ${file.name} from upload list`}
              >
                <FiX />
              </button>
            </div>
          ))}
        </div>
      )}

      {uploadedClips.length > 0 && (
        <button
          onClick={handleReupload}
          disabled={isUploading}
          aria-label={`Upload ${uploadedClips.length} edited clips`}
          className={`w-full px-6 py-3 font-semibold rounded-lg transition-colors ${
            isUploading
              ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
              : 'bg-green-500 hover:bg-green-600 text-white'
          }`}
        >
          {isUploading ? 'Uploading...' : `Upload ${uploadedClips.length} Edited Clips`}
        </button>
      )}
    </div>
  );
}
