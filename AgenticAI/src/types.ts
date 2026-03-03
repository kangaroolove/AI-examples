import type OpenAI from 'openai';

export interface Skill {
  name: string;
  description: string;
  parameters: {
    type: 'object';
    properties: Record<string, {
      type: string;
      description: string;
      enum?: string[];
    }>;
    required: string[];
  };
  execute: (args: Record<string, unknown>) => Promise<SkillResult>;
}

export interface SkillResult {
  success: boolean;
  output: string;
  error?: string;
}

export interface CLIConfig {
  model: string;
  baseUrl: string;
  apiKey: string;
  temperature: number;
}

export type MessageHistory = OpenAI.Chat.ChatCompletionMessageParam[];
export type OpenAITool = OpenAI.Chat.ChatCompletionTool;
