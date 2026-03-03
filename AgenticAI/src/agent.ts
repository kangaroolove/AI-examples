import OpenAI from 'openai';
import type { CLIConfig, MessageHistory } from './types';
import { getOpenAITools, executeSkill } from './skills/index';

const MAX_TOOL_ITERATIONS = 10;

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
  userMessage: string
): Promise<string> {
  history.push({ role: 'user', content: userMessage });

  const tools = getOpenAITools();
  let iterations = 0;

  while (iterations < MAX_TOOL_ITERATIONS) {
    iterations++;

    const response = await client.chat.completions.create({
      model: config.model,
      messages: history,
      tools,
      tool_choice: 'auto',
      temperature: config.temperature,
    });

    const choice = response.choices[0];
    if (!choice) {
      throw new Error('No response from model');
    }

    const assistantMessage = choice.message;
    history.push(assistantMessage);

    if (choice.finish_reason === 'tool_calls' && assistantMessage.tool_calls) {
      for (const toolCall of assistantMessage.tool_calls) {
        const fnName = toolCall.function.name;
        let fnArgs: Record<string, unknown> = {};
        try {
          fnArgs = JSON.parse(toolCall.function.arguments) as Record<string, unknown>;
        } catch {
          fnArgs = {};
        }

        const resultContent = await executeSkill(fnName, fnArgs);

        history.push({
          role: 'tool',
          tool_call_id: toolCall.id,
          content: resultContent,
        });
      }
      // Continue loop to get the model's next response after tool results
      continue;
    }

    // finish_reason === 'stop' or other terminal reason
    return assistantMessage.content ?? '';
  }

  return 'Maximum tool iterations reached. The agent stopped to prevent an infinite loop.';
}
