import { Command } from 'commander';
import * as readline from 'readline';
import chalk from 'chalk';
import { createClient } from './client';
import { checkOllamaConnection, runAgentTurn } from './agent';
import { getSkillNames } from './skills/index';
import type { CLIConfig, MessageHistory } from './types';

const SYSTEM_PROMPT = `You are a helpful AI assistant with the ability to control Windows Notepad on the user's machine.

You have the following skills available:
- open_notepad: Open Notepad, optionally with initial content
- write_notepad: Write (replace) content in the Notepad file
- save_notepad: Save the Notepad file (sends Ctrl+S)
- read_notepad: Read the current content of the Notepad file

When the user asks you to work with Notepad, use these skills to fulfill their request.
You can chain multiple skill calls in sequence to accomplish complex tasks (e.g., write content then save it).
Always confirm what you have done after completing the task.`;

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
        process.stdout.write(chalk.gray('Thinking...\r'));
        const response = await runAgentTurn(client, config, history, trimmed);
        process.stdout.write('             \r'); // clear "Thinking..."
        console.log(chalk.green('AI: ') + response + '\n');
      } catch (err) {
        process.stdout.write('             \r');
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
