import { Command } from 'commander';
import * as readline from 'readline';
import chalk from 'chalk';
import { createClient } from './client';
import { checkOllamaConnection, runAgentTurn } from './agent';
import { getSkillNames } from './skills/index';
import type { CLIConfig, MessageHistory } from './types';

const SYSTEM_PROMPT = `You are a helpful AI assistant. Always reply directly in this chat.

CRITICAL RULES — follow these without exception:
1. For greetings, questions, and general conversation, respond with plain text only. Do NOT call any tool.
2. Only call a tool when the user explicitly asks you to perform that action.
3. When the user asks to read a file (e.g. "read file X", "show me the contents of X"), call the read_file tool with the given path.

Available skills (only invoke when explicitly asked):
- open_notepad: Open Notepad with optional content
- write_notepad: Write content to Notepad
- save_notepad: Save Notepad (Ctrl+S)
- read_notepad: Read Notepad content
- read_file(file_path): Read and return the contents of any file at the given path`;

// ── Slash command registry ────────────────────────────────────────────────────
const SLASH_COMMANDS = [
  { cmd: '/clear',   desc: 'Reset conversation history' },
  { cmd: '/help',    desc: 'Show available commands' },
  { cmd: '/history', desc: 'Show message count in this session' },
];

// ── Inline hint rendering (shows below cursor when user types "/") ─────────────
const PROMPT_LEN = 2; // visible length of "> "
let hintCount = 0;

function clearHints(): void {
  if (hintCount === 0) return;
  process.stdout.write('\x1B[s');                   // save cursor (ANSI SCO)
  for (let i = 0; i < hintCount; i++) {
    process.stdout.write('\x1B[1B\r\x1B[2K');       // cursor DOWN (no scroll) + col 0 + erase
  }
  process.stdout.write('\x1B[u');                   // restore cursor
  hintCount = 0;
}

function renderHints(input: string): void {
  clearHints();
  if (!input.startsWith('/')) return;
  const matches = SLASH_COMMANDS.filter(c => c.cmd.startsWith(input));
  if (matches.length === 0) return;

  const n = matches.length;
  // Print blank lines to guarantee rows exist below (scrolls if at bottom).
  // On Windows, \n = CR+LF, so the column resets to 0 after each newline.
  process.stdout.write('\n'.repeat(n));
  // Go back up to the prompt row (cursor is now at col 0 due to Windows CR+LF).
  process.stdout.write(`\x1B[${n}A`);
  // Advance to the column where the cursor actually is (after "> " + typed input).
  const col = PROMPT_LEN + input.length;
  if (col > 0) process.stdout.write(`\x1B[${col}C`);
  // Save the correct cursor position.
  process.stdout.write('\x1B[s');
  // Draw each hint on the rows we just created.
  for (const m of matches) {
    process.stdout.write('\x1B[1B\r\x1B[2K');
    process.stdout.write(chalk.cyan(m.cmd.padEnd(12)) + chalk.gray(m.desc));
  }
  // Return cursor to the input line.
  process.stdout.write('\x1B[u');
  hintCount = n;
}

async function main(): Promise<void> {
  const program = new Command();

  program
    .name('agentic-ai')
    .description('TypeScript CLI agent with Ollama and Notepad skills')
    .option('-m, --model <model>', 'Ollama model to use', 'gpt-oss:120b-cloud')
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
  console.log(chalk.gray('Type / to see commands\n'));

  console.log(chalk.yellow('Connecting to Ollama...'));
  await checkOllamaConnection(client);
  console.log(chalk.green('Connected.\n'));

  const history: MessageHistory = [
    { role: 'system', content: SYSTEM_PROMPT },
  ];

  // Tab completion for slash commands
  const completer = (line: string): [string[], string] => {
    const hits = SLASH_COMMANDS.map(c => c.cmd).filter(c => c.startsWith(line));
    return [hits, line];
  };

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    completer,
  });

  // Show/update hints on every keypress when input starts with "/"
  readline.emitKeypressEvents(process.stdin, rl);
  process.stdin.on('keypress', () => {
    // 1. Clear old hints NOW (readline redraws its line before our listener fires,
    //    so hints from the previous keypress are already orphaned on screen).
    clearHints();
    // 2. After readline finishes updating rl.line, draw fresh hints.
    setImmediate(() => {
      const line = (rl as unknown as { line: string }).line ?? '';
      renderHints(line);
    });
  });

  const prompt = (): void => {
    rl.question(chalk.blue('> '), async (input: string) => {
      clearHints(); // remove hint lines before printing anything
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

      if (trimmed === '/help') {
        console.log(chalk.cyan('\nAvailable commands:'));
        for (const c of SLASH_COMMANDS) {
          console.log(`  ${chalk.cyan(c.cmd.padEnd(12))} ${chalk.gray(c.desc)}`);
        }
        console.log(chalk.gray(`  ${'exit'.padEnd(12)} Quit the program\n`));
        prompt();
        return;
      }

      if (trimmed === '/clear') {
        history.splice(1); // keep system prompt, reset the rest
        console.log(chalk.gray('Conversation history cleared.\n'));
        prompt();
        return;
      }

      if (trimmed === '/history') {
        const turns = history.filter(m => m.role === 'user' || m.role === 'assistant').length;
        console.log(chalk.gray(`Message history: ${turns} messages (${history.length} total).\n`));
        prompt();
        return;
      }

      try {
        const CLEAR_LINE = ' '.repeat(60) + '\r';
        process.stdout.write(chalk.gray('Thinking...\r'));

        const response = await runAgentTurn(
          client, config, history, trimmed,
          (msg) => {
            process.stdout.write(CLEAR_LINE);
            console.log(chalk.dim(`  > ${msg}`));
          }
        );

        process.stdout.write(CLEAR_LINE);
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
