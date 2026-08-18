import React from 'react';
import { SUPPORTED_LANGUAGES, LanguageOption } from '../types';

interface LanguageSelectorProps {
  value: string;
  onChange: (lang: string) => void;
  disabled?: boolean;
}

export const LanguageSelector: React.FC<LanguageSelectorProps> = ({ 
  value, 
  onChange, 
  disabled = false 
}) => {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      disabled={disabled}
      className="language-selector"
      aria-label="Select language"
    >
      {SUPPORTED_LANGUAGES.map((lang: LanguageOption) => (
        <option key={lang.code} value={lang.code}>
          {lang.nativeName} ({lang.name})
        </option>
      ))}
    </select>
  );
};