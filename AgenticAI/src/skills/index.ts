import type { Skill, OpenAITool } from '../types';
import { openNotepadSkill, writeNotepadSkill, saveNotepadSkill, readNotepadSkill } from './notepad';

// To add a new skill: create src/skills/myskill.ts, export a Skill,
// import it here, and push it to SKILL_REGISTRY.
const SKILL_REGISTRY: Skill[] = [
  openNotepadSkill,
  writeNotepadSkill,
  saveNotepadSkill,
  readNotepadSkill,
];

export function getOpenAITools(): OpenAITool[] {
  return SKILL_REGISTRY.map((skill) => ({
    type: 'function' as const,
    function: {
      name: skill.name,
      description: skill.description,
      parameters: skill.parameters,
    },
  }));
}

export function getSkillNames(): string[] {
  return SKILL_REGISTRY.map((s) => s.name);
}

export async function executeSkill(name: string, args: Record<string, unknown>): Promise<string> {
  const skill = SKILL_REGISTRY.find((s) => s.name === name);
  if (!skill) {
    return JSON.stringify({ success: false, output: '', error: `Unknown skill: ${name}` });
  }
  const result = await skill.execute(args);
  return JSON.stringify(result);
}
