import { Command } from 'commander';
import * as readline from 'readline';
import chalk from 'chalk';
import { createClient } from './client';
import { checkOllamaConnection, runAgentTurn } from './agent';
import { getSkillNames } from './skills/index';
import type { CLIConfig, MessageHistory } from './types';

const SYSTEM_PROMPT = `You are a helpful AI assistant. Always reply directly in this chat.

CRITICAL RULES — follow these without exception:
1. NEVER call any tool or skill unless the user's message contains an explicit Notepad instruction such as "open Notepad", "write to Notepad", "save Notepad", or "read Notepad".
2. For ALL other messages (greetings, questions, general conversation), respond with plain text only. Do NOT call any tool.
3. If the user says "Hi", "Hello", or asks a question unrelated to Notepad, just answer in text. No tool calls.

Notepad skills (only invoke when explicitly asked):
- open_notepad: Open Notepad with optional content
- write_notepad: Write content to Notepad
- save_notepad: Save Notepad (Ctrl+S)
- read_notepad: Read Notepad content`;

async function main(): Promise<void> {
  const program = new Command();

  program
    .name('agentic-ai')
    .description('TypeScript CLI agent with Ollama and Notepad skills')
    .option('-m, --model <model>', 'Ollama model to use', 'llama3.2')
    .option('-u, --url <url>', 'Ollama base URL', 'http://localhost:11434/v1')
    .option('-t, --temperature <number>', 'Sampling temperature', parseFloat, 0.7)
    .option('-k, --api-key <key>', 'API key (ignored by Ollama)', 'ollama')
    .parse(process.argv);

  const opts = program.opts<{
    model: string;
    url: string;
    temperature: number;
    apiKey: string;
  }>();

  const config: CLIConfig = {
    model: opts.model,
    baseUrl: opts.url,
    temperature: opts.temperature,
    apiKey: opts.apiKey,
  };

  const client = createClient(config);

  console.log(chalk.cyan('\nAgenticAI — TypeScript CLI Agent'));
  console.log(chalk.gray(`Model: ${config.model}  |  URL: ${config.baseUrl}`));
  console.log(chalk.gray(`Skills: ${getSkillNames().join(', ')}`));
  console.log(chalk.gray('Commands: exit, /clear, /history\n'));

  console.log(chalk.yellow('Connecting to Ollama...'));
  await checkOllamaConnection(client);
  console.log(chalk.green('Connected.\n'));

  const history: MessageHistory = [
    { role: 'system', content: SYSTEM_PROMPT },
  ];

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  const prompt = (): void => {
    rl.question(chalk.blue('You: '), async (input: string) => {
      const trimmed = input.trim();

      if (!trimmed) {
        prompt();
        return;
      }

      if (trimmed.toLowerCase() === 'exit') {
        console.log(chalk.gray('Goodbye.'));
        rl.close();
        return;
      }

      if (trimmed === '/clear') {
        // Keep system prompt, reset the rest
        history.splice(1);
        console.log(chalk.gray('Conversation history cleared.\n'));
        prompt();
        return;
      }

      if (trimmed === '/history') {
        const userMsgs = history.filter((m) => m.role === 'user' || m.role === 'assistant').length;
        console.log(chalk.gray(`Message history: ${userMsgs} messages (${history.length} total including system/tool).\n`));
        prompt();
        return;
      }

      try {
        const CLEAR = ' '.repeat(60) + '\r';
        process.stdout.write(chalk.gray('Thinking...\r'));

        const response = await runAgentTurn(
          client, config, history, trimmed,
          (msg) => {
            process.stdout.write(CLEAR);
            console.log(chalk.dim(`  > ${msg}`));
          }
        );

        process.stdout.write(CLEAR);
        console.log(chalk.green('AI: ') + response + '\n');
      } catch (err) {
        process.stdout.write(' '.repeat(60) + '\r');
        const message = err instanceof Error ? err.message : String(err);
        console.error(chalk.red(`Error: ${message}\n`));
      }

      prompt();
    });
  };

  prompt();
}

main().catch((err) => {
  console.error('Fatal error:', err);
  process.exit(1);
});
