import OpenAI from 'openai';
import type { CLIConfig } from './types';

export function createClient(config: CLIConfig): OpenAI {
  return new OpenAI({
    baseURL: config.baseUrl,
    apiKey: config.apiKey,
  });
}
