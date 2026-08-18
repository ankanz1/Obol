export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  language: string;
  timestamp: Date;
  audioUrl?: string;
  grounded?: boolean;
  refusalReason?: string;
  latencyMs?: number;
}

export interface PipelineResult {
  transcript: string;
  answer: string;
  audioBase64: string;
  language: string;
  grounded: boolean;
  refusalReason?: string;
  latencyMs: number;
  traceId: string;
}

export interface LanguageOption {
  code: string;
  name: string;
  nativeName: string;
}

export const SUPPORTED_LANGUAGES: LanguageOption[] = [
  { code: 'hi', name: 'Hindi', nativeName: 'हिंदी' },
  { code: 'bn', name: 'Bengali', nativeName: 'বাংলা' },
  { code: 'ta', name: 'Tamil', nativeName: 'தமிழ்' },
  { code: 'te', name: 'Telugu', nativeName: 'తెలుగు' },
  { code: 'mr', name: 'Marathi', nativeName: 'मराठी' },
  { code: 'gu', name: 'Gujarati', nativeName: 'ગુજરાતી' },
  { code: 'kn', name: 'Kannada', nativeName: 'ಕನ್ನಡ' },
  { code: 'ml', name: 'Malayalam', nativeName: 'മലയാളം' },
  { code: 'pa', name: 'Punjabi', nativeName: 'ਪੰਜਾਬੀ' },
  { code: 'or', name: 'Odia', nativeName: 'ଓଡ଼ିଆ' },
  { code: 'as', name: 'Assamese', nativeName: 'অসমীয়া' },
  { code: 'ur', name: 'Urdu', nativeName: 'اردو' },
  { code: 'ne', name: 'Nepali', nativeName: 'नेपाली' },
  { code: 'sa', name: 'Sanskrit', nativeName: 'संस्कृतम्' },
  { code: 'ks', name: 'Kashmiri', nativeName: 'کٲشُر' },
  { code: 'sd', name: 'Sindhi', nativeName: 'سنڌي' },
  { code: 'doi', name: 'Dogri', nativeName: 'डोगरी' },
  { code: 'sat', name: 'Santali', nativeName: 'ᱥᱟᱱᱛᱟᱲᱤ' },
  { code: 'en', name: 'English', nativeName: 'English' }
];

export interface SessionState {
  sessionId: string;
  language: string;
  messages: Message[];
  isRecording: boolean;
  isProcessing: boolean;
  wsConnected: boolean;
}