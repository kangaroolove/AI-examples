import { spawn, ChildProcess } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';
import type { Skill, SkillResult } from './types';

// ── Config types ──────────────────────────────────────────────────────────────

export interface MCPServerConfig {
  command: string;
  args?: string[];
  env?: Record<string, string>;
  disabled?: boolean;
}

export interface MCPConfig {
  mcpServers: Record<string, MCPServerConfig>;
}

export interface MCPServerStatus {
  name: string;
  status: 'connected' | 'failed' | 'disabled';
  tools: string[];
  error?: string;
}

// ── Internal MCP protocol types ───────────────────────────────────────────────

interface MCPTool {
  name: string;
  description?: string;
  inputSchema: {
    type: string;
    properties?: Record<string, { type: string; description?: string; enum?: string[] }>;
    required?: string[];
  };
}

// ── MCP stdio JSON-RPC client ─────────────────────────────────────────────────

class MCPClient {
  private proc: ChildProcess;
  private buffer = '';
  private pending = new Map<
    number,
    { resolve: (v: unknown) => void; reject: (e: Error) => void }
  >();
  private nextId = 0;

  constructor(config: MCPServerConfig) {
    const env = { ...process.env, ...(config.env ?? {}) };
    // On Windows, commands like `npx` are `.cmd` scripts and need cmd.exe to resolve.
    // Use `cmd /c` instead of shell:true to avoid the deprecation warning about
    // unsanitised argument concatenation.
    const isWindows = process.platform === 'win32';
    const command = isWindows ? 'cmd' : config.command;
    const args = isWindows
      ? ['/c', config.command, ...(config.args ?? [])]
      : (config.args ?? []);

    this.proc = spawn(command, args, {
      stdio: ['pipe', 'pipe', 'pipe'],
      env,
      shell: false,
    });

    this.proc.stdout!.on('data', (chunk: Buffer) => {
      this.buffer += chunk.toString();
      const lines = this.buffer.split('\n');
      this.buffer = lines.pop() ?? '';
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) continue;
        try {
          this.handleMessage(JSON.parse(trimmed));
        } catch { /* ignore non-JSON lines */ }
      }
    });

    // Suppress stderr so it doesn't pollute the terminal
    this.proc.stderr!.resume();
  }

  private handleMessage(msg: Record<string, unknown>): void {
    const id = msg['id'] as number | undefined;
    if (id === undefined) return;
    const p = this.pending.get(id);
    if (!p) return;
    this.pending.delete(id);
    if (msg['error']) {
      const err = msg['error'] as Record<string, unknown>;
      p.reject(new Error((err['message'] as string) ?? JSON.stringify(err)));
    } else {
      p.resolve(msg['result']);
    }
  }

  private send(msg: object): void {
    this.proc.stdin!.write(JSON.stringify(msg) + '\n');
  }

  private request(method: string, params: object = {}, timeoutMs = 15_000): Promise<unknown> {
    const id = ++this.nextId;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`MCP request timeout: ${method}`));
      }, timeoutMs);

      this.pending.set(id, {
        resolve: (v) => { clearTimeout(timer); resolve(v); },
        reject: (e) => { clearTimeout(timer); reject(e); },
      });

      this.send({ jsonrpc: '2.0', id, method, params });
    });
  }

  private notify(method: string, params: object = {}): void {
    this.send({ jsonrpc: '2.0', method, params });
  }

  async initialize(): Promise<void> {
    // Allow up to 60s for the first call — npx may need to download the package
    await this.request('initialize', {
      protocolVersion: '2024-11-05',
      capabilities: { tools: {} },
      clientInfo: { name: 'agentic-ai', version: '1.0.0' },
    }, 60_000);
    this.notify('notifications/initialized');
  }

  async listTools(): Promise<MCPTool[]> {
    const result = (await this.request('tools/list')) as Record<string, unknown> | null;
    return ((result?.['tools'] ?? []) as MCPTool[]);
  }

  async callTool(name: string, args: Record<string, unknown>): Promise<string> {
    const result = (await this.request('tools/call', { name, arguments: args })) as Record<string, unknown> | null;
    const content = (result?.['content'] ?? []) as Array<Record<string, unknown>>;
    return content
      .map((c) => (c['type'] === 'text' ? String(c['text']) : JSON.stringify(c)))
      .join('\n') || JSON.stringify(result);
  }

  kill(): void {
    try { this.proc.kill(); } catch { /* ignore */ }
  }
}

// ── Convert MCP tool → Skill ──────────────────────────────────────────────────

function mcpToolToSkill(serverName: string, tool: MCPTool, client: MCPClient): Skill {
  const schema = tool.inputSchema ?? { type: 'object', properties: {}, required: [] };
  // Prefix with server name to avoid collisions; use __ separator
  const skillName = `mcp__${serverName}__${tool.name}`;

  return {
    name: skillName,
    description: tool.description
      ? `[MCP:${serverName}] ${tool.description}`
      : `[MCP:${serverName}] ${tool.name}`,
    parameters: {
      type: 'object',
      properties: (schema.properties ?? {}) as Skill['parameters']['properties'],
      required: schema.required ?? [],
    },
    execute: async (args): Promise<SkillResult> => {
      try {
        const output = await client.callTool(tool.name, args);
        return { success: true, output };
      } catch (err) {
        const error = err instanceof Error ? err.message : String(err);
        return { success: false, output: '', error };
      }
    },
  };
}

// ── Config loading ────────────────────────────────────────────────────────────

const CONFIG_PATH = path.join(process.cwd(), 'mcp.json');

export function getMCPConfigPath(): string {
  return CONFIG_PATH;
}

function loadConfig(): MCPConfig {
  if (!fs.existsSync(CONFIG_PATH)) {
    return { mcpServers: {} };
  }
  try {
    return JSON.parse(fs.readFileSync(CONFIG_PATH, 'utf-8')) as MCPConfig;
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    throw new Error(`Failed to parse mcp.json: ${msg}`);
  }
}

// ── Public API ────────────────────────────────────────────────────────────────

const activeClients: MCPClient[] = [];

export async function initMCPServers(): Promise<{
  statuses: MCPServerStatus[];
  skills: Skill[];
}> {
  let config: MCPConfig;
  try {
    config = loadConfig();
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    return {
      statuses: [{ name: '(config)', status: 'failed', tools: [], error: msg }],
      skills: [],
    };
  }

  const entries = Object.entries(config.mcpServers);
  if (entries.length === 0) {
    return { statuses: [], skills: [] };
  }

  const statuses: MCPServerStatus[] = [];
  const skills: Skill[] = [];

  for (const [name, serverCfg] of entries) {
    if (serverCfg.disabled) {
      statuses.push({ name, status: 'disabled', tools: [] });
      continue;
    }

    const client = new MCPClient(serverCfg);
    try {
      await client.initialize();
      const tools = await client.listTools();
      activeClients.push(client);
      const toolNames = tools.map((t) => t.name);
      statuses.push({ name, status: 'connected', tools: toolNames });
      for (const tool of tools) {
        skills.push(mcpToolToSkill(name, tool, client));
      }
    } catch (err) {
      client.kill();
      const error = err instanceof Error ? err.message : String(err);
      statuses.push({ name, status: 'failed', tools: [], error });
    }
  }

  // Ensure MCP server processes are killed on exit
  process.once('exit', () => {
    for (const c of activeClients) c.kill();
  });

  return { statuses, skills };
}
