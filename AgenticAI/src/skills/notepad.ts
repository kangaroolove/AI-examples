import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { execSync, spawn, ChildProcess } from 'child_process';
import type { Skill, SkillResult } from '../types';

const TEMP_FILE_PATH = path.join(os.tmpdir(), 'agentic_ai_notepad.txt');
let notepadProcess: ChildProcess | null = null;

function killNotepad(): void {
  if (notepadProcess) {
    try {
      notepadProcess.kill();
    } catch {
      // Process may already be gone
    }
    notepadProcess = null;
  }
}

function spawnNotepad(): void {
  notepadProcess = spawn('notepad.exe', [TEMP_FILE_PATH], {
    detached: true,
    stdio: 'ignore',
  });
  notepadProcess.unref();
}

export const openNotepadSkill: Skill = {
  name: 'open_notepad',
  description: 'Open Notepad with optional initial content. Creates a temporary file and opens it in Notepad.',
  parameters: {
    type: 'object',
    properties: {
      initial_content: {
        type: 'string',
        description: 'Optional initial text content to write to the file before opening Notepad.',
      },
    },
    required: [],
  },
  async execute(args: Record<string, unknown>): Promise<SkillResult> {
    try {
      const content = typeof args['initial_content'] === 'string' ? args['initial_content'] : '';
      fs.writeFileSync(TEMP_FILE_PATH, content, 'utf-8');
      killNotepad();
      spawnNotepad();
      return { success: true, output: `Notepad opened with file: ${TEMP_FILE_PATH}` };
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      return { success: false, output: '', error: message };
    }
  },
};

export const writeNotepadSkill: Skill = {
  name: 'write_notepad',
  description: 'Write content to the Notepad file. Replaces existing content. Notepad is restarted to show the new content.',
  parameters: {
    type: 'object',
    properties: {
      content: {
        type: 'string',
        description: 'The text content to write to the Notepad file.',
      },
    },
    required: ['content'],
  },
  async execute(args: Record<string, unknown>): Promise<SkillResult> {
    try {
      const content = typeof args['content'] === 'string' ? args['content'] : String(args['content']);
      killNotepad();
      fs.writeFileSync(TEMP_FILE_PATH, content, 'utf-8');
      spawnNotepad();
      return { success: true, output: `Written ${content.length} characters to Notepad.` };
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      return { success: false, output: '', error: message };
    }
  },
};

export const saveNotepadSkill: Skill = {
  name: 'save_notepad',
  description: 'Save the current Notepad file by sending Ctrl+S to the Notepad window.',
  parameters: {
    type: 'object',
    properties: {},
    required: [],
  },
  async execute(_args: Record<string, unknown>): Promise<SkillResult> {
    try {
      execSync(
        `powershell -NoProfile -Command "$wsh = New-Object -ComObject WScript.Shell; $wsh.AppActivate('Notepad'); Start-Sleep -Milliseconds 500; $wsh.SendKeys('^s')"`,
        { stdio: 'pipe' }
      );
      return { success: true, output: 'Ctrl+S sent to Notepad. File saved.' };
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      return { success: false, output: '', error: `Failed to save: ${message}` };
    }
  },
};

export const readNotepadSkill: Skill = {
  name: 'read_notepad',
  description: 'Read and return the current content of the Notepad temporary file.',
  parameters: {
    type: 'object',
    properties: {},
    required: [],
  },
  async execute(_args: Record<string, unknown>): Promise<SkillResult> {
    try {
      if (!fs.existsSync(TEMP_FILE_PATH)) {
        return { success: true, output: '(Notepad file does not exist yet — open Notepad first.)' };
      }
      const content = fs.readFileSync(TEMP_FILE_PATH, 'utf-8');
      return { success: true, output: content || '(file is empty)' };
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      return { success: false, output: '', error: message };
    }
  },
};
