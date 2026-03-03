import OpenAI from 'openai';
import type { CLIConfig } from './types';
import { getSkillDescriptions, executeSkill } from './skills/index';

export interface WorkflowStep {
  skill: string;
  args: Record<string, unknown>;
  reason: string;
}

export interface WorkflowPlan {
  goal: string;
  steps: WorkflowStep[];
}

export interface StepResult {
  step: WorkflowStep;
  output: string;
  success: boolean;
}

/**
 * Phase 1 — Planner
 * Asks the model to review the skill registry and produce a JSON execution plan.
 */
export async function planTask(
  client: OpenAI,
  config: CLIConfig,
  userMessage: string
): Promise<WorkflowPlan> {
  const skillsDesc = getSkillDescriptions();

  const plannerPrompt =
    `You are a task planner for an AI agent that controls Windows Notepad.\n\n` +
    `Available skills:\n${skillsDesc}\n\n` +
    `User request: "${userMessage}"\n\n` +
    `Create a step-by-step execution plan. Rules:\n` +
    `- Match the request to the most relevant skills from the list above.\n` +
    `- To write content to Notepad: ALWAYS include open_notepad first, then write_notepad, then save_notepad.\n` +
    `- Put the actual generated content (poem, text, etc.) directly inside the write_notepad "content" arg.\n` +
    `- To only read Notepad: just use read_notepad.\n` +
    `- Only include skills actually needed for this request.\n` +
    `- Respond with ONLY a valid JSON object — no markdown fences, no explanation.\n\n` +
    `{\n` +
    `  "goal": "one sentence description of what will be done",\n` +
    `  "steps": [\n` +
    `    { "skill": "open_notepad", "args": {}, "reason": "Open Notepad before writing" },\n` +
    `    { "skill": "write_notepad", "args": { "content": "<full poem/text here>" }, "reason": "Write the poem" },\n` +
    `    { "skill": "save_notepad", "args": {}, "reason": "Save the file" }\n` +
    `  ]\n` +
    `}`;

  const response = await client.chat.completions.create({
    model: config.model,
    messages: [{ role: 'user', content: plannerPrompt }],
    temperature: 0.3, // low temperature for deterministic planning
  });

  const raw = response.choices[0]?.message?.content ?? '';

  // Extract JSON even if the model wraps it in markdown fences
  const jsonMatch = raw.match(/\{[\s\S]*\}/);
  if (!jsonMatch) {
    throw new Error(`Planner did not return valid JSON.\nGot: ${raw.slice(0, 300)}`);
  }

  const plan = JSON.parse(jsonMatch[0]) as WorkflowPlan;
  if (!Array.isArray(plan.steps) || plan.steps.length === 0) {
    throw new Error('Planner returned an empty or invalid plan.');
  }

  return plan;
}

/**
 * Phase 2 — Executor
 * Runs each step in order, stops on first failure.
 */
export async function executeWorkflow(
  plan: WorkflowPlan,
  onStep: (index: number, total: number, step: WorkflowStep) => void
): Promise<StepResult[]> {
  const results: StepResult[] = [];

  for (let i = 0; i < plan.steps.length; i++) {
    const step = plan.steps[i];
    onStep(i, plan.steps.length, step);

    const resultJson = await executeSkill(step.skill, step.args);
    const parsed = JSON.parse(resultJson) as { success: boolean; output: string; error?: string };

    results.push({
      step,
      output: parsed.success ? parsed.output : (parsed.error ?? 'unknown error'),
      success: parsed.success,
    });

    if (!parsed.success) break; // abort on failure
  }

  return results;
}
