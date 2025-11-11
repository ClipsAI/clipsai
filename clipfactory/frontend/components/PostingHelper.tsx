/**
 * Posting Helper - Phase 5
 * Upload screenshot → Generate title based on account type
 */
import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { FiImage, FiCopy, FiCheck, FiAlertCircle, FiX } from 'react-icons/fi';
import { motion } from 'framer-motion';
import axios from 'axios';

interface PostingHelperProps {
  accountType?: 'fan' | 'brand' | 'watermark';
}

const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB
const ACCEPTED_IMAGE_TYPES = ['image/png', 'image/jpeg', 'image/jpg'];

export default function PostingHelper({ accountType = 'fan' }: PostingHelperProps) {
  const [screenshot, setScreenshot] = useState<File | null>(null);
  const [screenshotPreview, setScreenshotPreview] = useState<string>('');
  const [generatedTitle, setGeneratedTitle] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [isCopied, setIsCopied] = useState(false);
  const [selectedAccountType, setSelectedAccountType] = useState(accountType);
  const [error, setError] = useState<string>('');

  // Format file size for display
  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
  };

  const onDrop = useCallback((acceptedFiles: File[], rejectedFiles: any[]) => {
    setError(''); // Clear previous errors

    // Handle rejected files
    if (rejectedFiles.length > 0) {
      const rejection = rejectedFiles[0];
      if (rejection.errors[0]?.code === 'file-too-large') {
        setError(`File is too large. Maximum size is ${formatFileSize(MAX_FILE_SIZE)}`);
      } else if (rejection.errors[0]?.code === 'file-invalid-type') {
        setError('Invalid file type. Please upload PNG, JPG, or JPEG images.');
      } else {
        setError('File upload rejected. Please try again.');
      }
      return;
    }

    const file = acceptedFiles[0];
    if (!file) return;

    // Validate file type
    if (!ACCEPTED_IMAGE_TYPES.includes(file.type)) {
      setError('Invalid file type. Please upload PNG, JPG, or JPEG images.');
      return;
    }

    // Validate file size
    if (file.size > MAX_FILE_SIZE) {
      setError(`File is too large (${formatFileSize(file.size)}). Maximum size is ${formatFileSize(MAX_FILE_SIZE)}`);
      return;
    }

    setScreenshot(file);

    // Create preview
    const reader = new FileReader();
    reader.onload = (e) => {
      setScreenshotPreview(e.target?.result as string);
    };
    reader.onerror = () => {
      setError('Failed to read file. Please try again.');
    };
    reader.readAsDataURL(file);

    // Auto-generate title
    generateTitle(file, selectedAccountType);
  }, [selectedAccountType]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg']
    },
    maxFiles: 1,
    maxSize: MAX_FILE_SIZE,
  });

  const generateTitle = async (file: File, accType: string) => {
    setIsGenerating(true);
    setError(''); // Clear previous errors

    const formData = new FormData();
    formData.append('screenshot', file);
    formData.append('account_type', accType);

    try {
      const response = await axios.post('/api/phase5/screenshot-to-title', formData);

      setGeneratedTitle(response.data.title);
    } catch (error: any) {
      const errorMessage = error.response?.data?.message || error.message || 'Title generation failed. Please try again.';
      setError(errorMessage);
      setGeneratedTitle('');
    } finally {
      setIsGenerating(false);
    }
  };

  const copyTitle = () => {
    if (!generatedTitle) {
      setError('No title to copy');
      return;
    }

    navigator.clipboard.writeText(generatedTitle)
      .then(() => {
        setIsCopied(true);
        setTimeout(() => setIsCopied(false), 2000);
      })
      .catch(() => {
        setError('Failed to copy title to clipboard');
      });
  };

  const handleAccountTypeChange = (type: 'fan' | 'brand' | 'watermark') => {
    setSelectedAccountType(type);
    if (screenshot) {
      generateTitle(screenshot, type);
    }
  };

  return (
    <div className="w-full max-w-4xl mx-auto p-6 space-y-8">
      {/* Header */}
      <div className="text-center">
        <h2 className="text-3xl font-bold text-gray-900">Posting Helper</h2>
        <p className="text-gray-600 mt-2">Upload screenshot → Generate title</p>
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

      {/* Account Type Selection */}
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h3 className="text-lg font-semibold mb-4">Account Type</h3>

        <div className="grid grid-cols-3 gap-4">
          <button
            onClick={() => handleAccountTypeChange('fan')}
            aria-label="Select Fan Account type"
            aria-pressed={selectedAccountType === 'fan'}
            className={`p-4 rounded-lg border-2 transition-all ${
              selectedAccountType === 'fan'
                ? 'border-blue-500 bg-blue-50'
                : 'border-gray-200 hover:border-gray-300'
            }`}
          >
            <div className="font-semibold">Fan Account</div>
            <div className="text-sm text-gray-600 mt-1">
              Casual, excited, emojis
            </div>
            <div className="text-xs text-gray-500 mt-2 italic">
              "bro was STRUGGLING 💀"
            </div>
          </button>

          <button
            onClick={() => handleAccountTypeChange('brand')}
            aria-label="Select Brand Account type"
            aria-pressed={selectedAccountType === 'brand'}
            className={`p-4 rounded-lg border-2 transition-all ${
              selectedAccountType === 'brand'
                ? 'border-blue-500 bg-blue-50'
                : 'border-gray-200 hover:border-gray-300'
            }`}
          >
            <div className="font-semibold">Brand Account</div>
            <div className="text-sm text-gray-600 mt-1">
              Professional, inspiring
            </div>
            <div className="text-xs text-gray-500 mt-2 italic">
              "Elite Training Techniques"
            </div>
          </button>

          <button
            onClick={() => handleAccountTypeChange('watermark')}
            aria-label="Select Watermark Account type"
            aria-pressed={selectedAccountType === 'watermark'}
            className={`p-4 rounded-lg border-2 transition-all ${
              selectedAccountType === 'watermark'
                ? 'border-blue-500 bg-blue-50'
                : 'border-gray-200 hover:border-gray-300'
            }`}
          >
            <div className="font-semibold">Watermark Account</div>
            <div className="text-sm text-gray-600 mt-1">
              Fan commentary style
            </div>
            <div className="text-xs text-gray-500 mt-2 italic">
              "the way he crushed this tho 🔥"
            </div>
          </button>
        </div>
      </div>

      {/* Screenshot Upload */}
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h3 className="text-lg font-semibold mb-4">Upload Screenshot</h3>

        <div
          {...getRootProps()}
          role="button"
          aria-label="Upload screenshot dropzone"
          className={`border-2 border-dashed rounded-lg p-12 text-center cursor-pointer transition-colors ${
            isDragActive
              ? 'border-purple-500 bg-purple-50'
              : 'border-gray-300 hover:border-gray-400'
          }`}
        >
          <input {...getInputProps()} aria-label="Screenshot file input" />

          <FiImage className="w-16 h-16 mx-auto text-gray-400 mb-4" />

          {isDragActive ? (
            <p className="text-lg text-purple-600 font-semibold">Drop screenshot here...</p>
          ) : (
            <div>
              <p className="text-lg text-gray-700 font-semibold mb-2">
                Drop screenshot or click to browse
              </p>
              <p className="text-sm text-gray-500">
                PNG, JPG, JPEG • Max {formatFileSize(MAX_FILE_SIZE)}
              </p>
            </div>
          )}
        </div>

        {/* Screenshot Preview */}
        {screenshotPreview && (
          <div className="mt-6">
            <img
              src={screenshotPreview}
              alt={screenshot ? `Preview of uploaded screenshot: ${screenshot.name}` : 'Screenshot preview'}
              className="w-full h-64 object-contain rounded-lg border border-gray-200"
            />
            {screenshot && (
              <div className="mt-2 text-sm text-gray-600 text-center">
                {screenshot.name} • {formatFileSize(screenshot.size)}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Generated Title */}
      {(generatedTitle || isGenerating) && (
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h3 className="text-lg font-semibold mb-4">Generated Title</h3>

          {isGenerating ? (
            <div className="flex items-center space-x-3 p-4 bg-gray-50 rounded-lg">
              <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-purple-500" />
              <span className="text-gray-600">Generating title...</span>
            </div>
          ) : (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
            >
              <div className="relative p-6 bg-gradient-to-r from-purple-50 to-blue-50 rounded-lg border-2 border-purple-200">
                <div className="text-xl font-semibold text-gray-900 mb-4">
                  {generatedTitle}
                </div>

                <div className="flex items-center space-x-4">
                  <button
                    onClick={copyTitle}
                    aria-label={isCopied ? 'Title copied to clipboard' : 'Copy title to clipboard'}
                    className="flex items-center space-x-2 px-4 py-2 bg-purple-500 hover:bg-purple-600 text-white font-medium rounded-lg transition-colors"
                  >
                    {isCopied ? (
                      <>
                        <FiCheck className="w-4 h-4" />
                        <span>Copied!</span>
                      </>
                    ) : (
                      <>
                        <FiCopy className="w-4 h-4" />
                        <span>Copy Title</span>
                      </>
                    )}
                  </button>

                  <button
                    onClick={() => screenshot && generateTitle(screenshot, selectedAccountType)}
                    disabled={!screenshot}
                    aria-label="Regenerate title from screenshot"
                    className={`px-4 py-2 font-medium rounded-lg transition-colors ${
                      screenshot
                        ? 'bg-gray-200 hover:bg-gray-300 text-gray-700'
                        : 'bg-gray-100 text-gray-400 cursor-not-allowed'
                    }`}
                  >
                    Regenerate
                  </button>
                </div>
              </div>

              <div className="mt-4 text-sm text-gray-600">
                <p><strong>Account Type:</strong> {selectedAccountType}</p>
                <p className="mt-1"><strong>Style:</strong> {
                  selectedAccountType === 'fan' ? 'Casual fan reaction' :
                  selectedAccountType === 'brand' ? 'Professional brand voice' :
                  'Fan commentary'
                }</p>
              </div>
            </motion.div>
          )}
        </div>
      )}

      {/* Quick Tips */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
        <h4 className="font-semibold text-blue-900 mb-2">💡 Quick Tips</h4>
        <ul className="text-sm text-blue-800 space-y-1">
          <li>• Screenshot the variation you want to post</li>
          <li>• Select the account type you're posting to</li>
          <li>• Generated title matches account style automatically</li>
          <li>• Copy and paste into your posting queue</li>
        </ul>
      </div>
    </div>
  );
}
