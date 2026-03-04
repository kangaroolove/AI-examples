import OpenAI from 'openai';
import type { CLIConfig, MessageHistory } from './types';
import { planTask, executeWorkflow } from './workflow';
import { getOpenAITools, executeSkill } from './skills/index';

const NOTEPAD_KEYWORD = /notepad/i;

export async function checkOllamaConnection(client: OpenAI): Promise<void> {
  try {
    await client.models.list();
  } catch {
    console.error(
      '\nCould not connect to Ollama. Make sure it is running:\n' +
      '  ollama serve\n' +
      '  ollama pull llama3.2\n'
    );
    process.exit(1);
  }
}

export async function runAgentTurn(
  client: OpenAI,
  config: CLIConfig,
  history: MessageHistory,
  userMessage: string,
  onProgress?: (message: string) => void
): Promise<string> {
  history.push({ role: 'user', content: userMessage });

  // ── Workflow path: Notepad task ──────────────────────────────────────────
  if (NOTEPAD_KEYWORD.test(userMessage)) {
    onProgress?.('Discovering relevant skills...');

    let plan;
    try {
      plan = await planTask(client, config, userMessage);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      const response = `Could not create a plan: ${msg}`;
      history.push({ role: 'assistant', content: response });
      return response;
    }

    onProgress?.(`Goal: ${plan.goal}`);
    onProgress?.(`Plan: ${plan.steps.map((s) => s.skill).join(' → ')}`);

    const results = await executeWorkflow(plan, (i, total, step) => {
      onProgress?.(`[${i + 1}/${total}] ${step.skill} — ${step.reason}`);
    });

    const allOk = results.every((r) => r.success);
    const lines = results.map(
      (r, i) => `  ${i + 1}. ${r.step.skill}: ${r.success ? '✓' : '✗'} ${r.output}`
    );

    const finalResponse = allOk
      ? `Done! ${plan.goal}\n\n${lines.join('\n')}`
      : `Workflow stopped early.\n\n${lines.join('\n')}`;

    history.push({ role: 'assistant', content: finalResponse });
    return finalResponse;
  }

  // ── Plain chat path (with tool-call support) ─────────────────────────────
  const tools = getOpenAITools();

  // Agentic loop: keep going until the model stops calling tools
  for (;;) {
    const response = await client.chat.completions.create({
      model: config.model,
      messages: history,
      temperature: config.temperature,
      tools,
      tool_choice: 'auto',
    });

    const assistantMessage = response.choices[0]?.message;
    if (!assistantMessage) throw new Error('No response from model');
    history.push(assistantMessage);

    const toolCalls = assistantMessage.tool_calls;
    if (!toolCalls || toolCalls.length === 0) {
      // No tool calls — plain text response
      return assistantMessage.content ?? '';
    }

    // Execute each tool call and feed results back
    for (const call of toolCalls) {
      let args: Record<string, unknown> = {};
      try { args = JSON.parse(call.function.arguments); } catch { /* leave empty */ }
      const result = await executeSkill(call.function.name, args);
      history.push({
        role: 'tool',
        tool_call_id: call.id,
        content: result,
      });
    }
    // Loop back so the model can produce a final response after seeing tool results
  }
}
