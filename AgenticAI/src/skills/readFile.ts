import * as fs from 'fs';
import type { Skill, SkillResult } from '../types';

export const readFileSkill: Skill = {
  name: 'read_file',
  description: 'Read the content of a file at a given path and return it as text.',
  parameters: {
    type: 'object',
    properties: {
      file_path: {
        type: 'string',
        description: 'The absolute or relative path to the file to read.',
      },
    },
    required: ['file_path'],
  },
  async execute(args: Record<string, unknown>): Promise<SkillResult> {
    try {
      const filePath = typeof args['file_path'] === 'string' ? args['file_path'] : String(args['file_path']);
      if (!fs.existsSync(filePath)) {
        return { success: false, output: '', error: `File not found: ${filePath}` };
      }
      const content = fs.readFileSync(filePath, 'utf-8');
      return { success: true, output: content || '(file is empty)' };
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      return { success: false, output: '', error: message };
    }
  },
};
