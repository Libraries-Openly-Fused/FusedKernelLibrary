#!/usr/bin/env node

/**
 * FusedKernelLibrary MCP Server
 * 
 * This MCP server exposes the FusedKernelLibrary functionality to AI agents
 * for testing and integration purposes.
 */

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ErrorCode,
  ListResourcesRequestSchema,
  ListToolsRequestSchema,
  McpError,
  ReadResourceRequestSchema,
} from '@modelcontextprotocol/sdk/types.js';
import { execSync, spawn } from 'child_process';
import { readFileSync, existsSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const projectRoot = join(__dirname, '..');

class FusedKernelLibraryServer {
  constructor() {
    this.server = new Server(
      {
        name: 'fused-kernel-library-mcp',
        version: '1.0.0',
      },
      {
        capabilities: {
          resources: {},
          tools: {},
        },
      }
    );

    this.setupToolHandlers();
    this.setupResourceHandlers();
  }

  setupToolHandlers() {
    this.server.setRequestHandler(ListToolsRequestSchema, async () => {
      return {
        tools: [
          {
            name: 'build_library',
            description: 'Build the FusedKernelLibrary using CMake',
            inputSchema: {
              type: 'object',
              properties: {
                buildType: {
                  type: 'string',
                  enum: ['Debug', 'Release'],
                  default: 'Release',
                  description: 'Build configuration type'
                },
                enableCuda: {
                  type: 'boolean',
                  default: true,
                  description: 'Enable CUDA support'
                },
                enableCpu: {
                  type: 'boolean',
                  default: true,
                  description: 'Enable CPU support'
                }
              }
            }
          },
          {
            name: 'run_tests',
            description: 'Run the FusedKernelLibrary test suite',
            inputSchema: {
              type: 'object',
              properties: {
                testType: {
                  type: 'string',
                  enum: ['all', 'unit', 'standard', 'benchmark'],
                  default: 'all',
                  description: 'Type of tests to run'
                },
                verbose: {
                  type: 'boolean',
                  default: false,
                  description: 'Enable verbose output'
                }
              }
            }
          },
          {
            name: 'get_library_info',
            description: 'Get information about the FusedKernelLibrary capabilities',
            inputSchema: {
              type: 'object',
              properties: {}
            }
          },
          {
            name: 'check_cuda_support',
            description: 'Check if CUDA is available and supported',
            inputSchema: {
              type: 'object',
              properties: {}
            }
          },
          {
            name: 'list_examples',
            description: 'List available code examples in the library',
            inputSchema: {
              type: 'object',
              properties: {}
            }
          },
          {
            name: 'validate_code_example',
            description: 'Validate a code example against the library API',
            inputSchema: {
              type: 'object',
              properties: {
                examplePath: {
                  type: 'string',
                  description: 'Path to the example file to validate'
                }
              },
              required: ['examplePath']
            }
          }
        ]
      };
    });

    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      const { name, arguments: args } = request.params;

      try {
        switch (name) {
          case 'build_library':
            return await this.buildLibrary(args);
          case 'run_tests':
            return await this.runTests(args);
          case 'get_library_info':
            return await this.getLibraryInfo();
          case 'check_cuda_support':
            return await this.checkCudaSupport();
          case 'list_examples':
            return await this.listExamples();
          case 'validate_code_example':
            return await this.validateCodeExample(args);
          default:
            throw new McpError(
              ErrorCode.MethodNotFound,
              `Unknown tool: ${name}`
            );
        }
      } catch (error) {
        throw new McpError(
          ErrorCode.InternalError,
          `Tool execution failed: ${error.message}`
        );
      }
    });
  }

  // Secure command execution helper
  execCommand(command, args, cwd, timeout = 30000) {
    return new Promise((resolve, reject) => {
      const child = spawn(command, args, { 
        cwd,
        stdio: ['ignore', 'pipe', 'pipe']
      });
      
      let stdout = '';
      let stderr = '';
      
      child.stdout.on('data', (data) => {
        stdout += data.toString();
      });
      
      child.stderr.on('data', (data) => {
        stderr += data.toString();
      });
      
      const timeoutId = setTimeout(() => {
        child.kill();
        reject(new Error('Command timeout'));
      }, timeout);
      
      child.on('close', (code) => {
        clearTimeout(timeoutId);
        if (code === 0) {
          resolve(stdout);
        } else {
          reject(new Error(`Command failed with code ${code}: ${stderr}`));
        }
      });
      
      child.on('error', (error) => {
        clearTimeout(timeoutId);
        reject(error);
      });
    });
  }

  setupResourceHandlers() {
    this.server.setRequestHandler(ListResourcesRequestSchema, async () => {
      return {
        resources: [
          {
            uri: 'fusedkernel://readme',
            mimeType: 'text/markdown',
            name: 'FusedKernelLibrary README',
            description: 'Main documentation for the library'
          },
          {
            uri: 'fusedkernel://cmake-config',
            mimeType: 'text/plain',
            name: 'CMake Configuration',
            description: 'CMake build configuration'
          },
          {
            uri: 'fusedkernel://examples',
            mimeType: 'application/json',
            name: 'Code Examples',
            description: 'Available code examples and their descriptions'
          }
        ]
      };
    });

    this.server.setRequestHandler(ReadResourceRequestSchema, async (request) => {
      const { uri } = request.params;

      switch (uri) {
        case 'fusedkernel://readme':
          return {
            contents: [
              {
                uri,
                mimeType: 'text/markdown',
                text: readFileSync(join(projectRoot, 'README.md'), 'utf-8')
              }
            ]
          };
        case 'fusedkernel://cmake-config':
          return {
            contents: [
              {
                uri,
                mimeType: 'text/plain',
                text: readFileSync(join(projectRoot, 'CMakeLists.txt'), 'utf-8')
              }
            ]
          };
        case 'fusedkernel://examples':
          return {
            contents: [
              {
                uri,
                mimeType: 'application/json',
                text: JSON.stringify(await this.getExamplesList(), null, 2)
              }
            ]
          };
        default:
          throw new McpError(
            ErrorCode.InvalidRequest,
            `Unknown resource: ${uri}`
          );
      }
    });
  }

  async buildLibrary(args = {}) {
    const buildType = args.buildType || 'Release';
    const enableCuda = args.enableCuda !== false;
    const enableCpu = args.enableCpu !== false;

    try {
      // Create build directory
      const buildDir = join(projectRoot, 'build');
      
      // Sanitize paths and build configuration
      const safeBuildDir = buildDir.replace(/[^a-zA-Z0-9_\-\/\.]/g, '');
      const safeProjectRoot = projectRoot.replace(/[^a-zA-Z0-9_\-\/\.]/g, '');
      const safeBuildType = ['Debug', 'Release'].includes(buildType) ? buildType : 'Release';
      
      const cmakeArgs = [
        'cmake',
        '-B', safeBuildDir,
        `-DCMAKE_BUILD_TYPE=${safeBuildType}`,
        `-DENABLE_CUDA=${enableCuda ? 'ON' : 'OFF'}`,
        `-DENABLE_CPU=${enableCpu ? 'ON' : 'OFF'}`,
        safeProjectRoot
      ];

      const buildArgs = ['cmake', '--build', safeBuildDir, '--config', safeBuildType];

      const configOutput = await this.execCommand(cmakeArgs[0], cmakeArgs.slice(1), safeProjectRoot, 60000);
      const buildOutput = await this.execCommand(buildArgs[0], buildArgs.slice(1), safeProjectRoot, 120000);

      return {
        content: [
          {
            type: 'text',
            text: `Build completed successfully!\n\nConfiguration:\n${configOutput}\n\nBuild:\n${buildOutput}`
          }
        ]
      };
    } catch (error) {
      return {
        content: [
          {
            type: 'text',
            text: `Build failed: ${error.message}`
          }
        ]
      };
    }
  }

  async runTests(args = {}) {
    const testType = args.testType || 'all';
    const verbose = args.verbose || false;

    try {
      const buildDir = join(projectRoot, 'build');
      
      if (!existsSync(buildDir)) {
        throw new Error('Build directory not found. Please build the library first.');
      }

      const ctestArgs = [
        'ctest',
        '--build-config', 'Release'
      ];
      
      if (verbose) {
        ctestArgs.push('--verbose');
      }
      
      if (testType !== 'all') {
        ctestArgs.push('--tests-regex', testType);
      }

      const safeBuildDir = buildDir.replace(/[^a-zA-Z0-9_\-\/\.]/g, '');
      const output = await this.execCommand(ctestArgs[0], ctestArgs.slice(1), safeBuildDir, 300000);

      return {
        content: [
          {
            type: 'text',
            text: `Tests completed!\n\n${output}`
          }
        ]
      };
    } catch (error) {
      return {
        content: [
          {
            type: 'text',
            text: `Tests failed: ${error.message}`
          }
        ]
      };
    }
  }

  async getLibraryInfo() {
    try {
      const cmakeContent = readFileSync(join(projectRoot, 'CMakeLists.txt'), 'utf-8');
      const readmeContent = readFileSync(join(projectRoot, 'README.md'), 'utf-8');
      
      // Extract version from CMakeLists.txt
      const versionMatch = cmakeContent.match(/PROJECT_VERSION_MAJOR\s+(\d+).*PROJECT_VERSION_MINOR\s+(\d+).*PROJECT_VERSION_REV\s+(\d+)/s);
      const version = versionMatch ? `${versionMatch[1]}.${versionMatch[2]}.${versionMatch[3]}` : 'Unknown';

      return {
        content: [
          {
            type: 'text',
            text: `FusedKernelLibrary Information:

Version: ${version}
Description: C++17 implementation of GPU kernel fusion methodology
Supported Backends: CPU, CUDA
Fusion Types: Vertical, Horizontal, Backwards Vertical, Divergent Horizontal

Key Features:
- Automatic kernel fusion for GPU libraries
- CPU and CUDA backends support
- Template-based operation chaining
- Memory-efficient kernel execution
- Compatible with existing CUDA code

Build Options:
- ENABLE_CPU: CPU backend support
- ENABLE_CUDA: CUDA backend support  
- BUILD_TEST: Standard tests
- BUILD_UTEST: Unit tests
- ENABLE_BENCHMARK: Benchmarking tests`
          }
        ]
      };
    } catch (error) {
      return {
        content: [
          {
            type: 'text',
            text: `Failed to get library info: ${error.message}`
          }
        ]
      };
    }
  }

  async checkCudaSupport() {
    try {
      let cudaInfo = '';
      
      // Check for CUDA compiler
      try {
        const nvccOutput = execSync('nvcc --version', { encoding: 'utf-8', timeout: 5000 });
        cudaInfo += `NVCC found:\n${nvccOutput}\n`;
      } catch {
        cudaInfo += 'NVCC compiler not found\n';
      }

      // Check for nvidia-smi
      try {
        const smiOutput = execSync('nvidia-smi', { encoding: 'utf-8', timeout: 5000 });
        cudaInfo += `\nNVIDIA GPU information:\n${smiOutput}`;
      } catch {
        cudaInfo += '\nnvidia-smi not available (no GPU detected or drivers not installed)';
      }

      return {
        content: [
          {
            type: 'text',
            text: `CUDA Support Status:\n\n${cudaInfo}`
          }
        ]
      };
    } catch (error) {
      return {
        content: [
          {
            type: 'text',
            text: `Failed to check CUDA support: ${error.message}`
          }
        ]
      };
    }
  }

  async listExamples() {
    try {
      const examplesList = await this.getExamplesList();
      
      return {
        content: [
          {
            type: 'text',
            text: `Available Examples:\n\n${JSON.stringify(examplesList, null, 2)}`
          }
        ]
      };
    } catch (error) {
      return {
        content: [
          {
            type: 'text',
            text: `Failed to list examples: ${error.message}`
          }
        ]
      };
    }
  }

  async getExamplesList() {
    const examples = [];
    
    // Check tests directory for examples
    try {
      const testsDir = join(projectRoot, 'tests');
      const testFiles = execSync('find . -name "*.h" -o -name "*.cpp" -o -name "*.cu"', { 
        cwd: testsDir, 
        encoding: 'utf-8' 
      }).split('\n').filter(Boolean);
      
      examples.push(...testFiles.map(file => ({
        path: file,
        type: 'test',
        directory: 'tests'
      })));
    } catch (error) {
      // Ignore errors if tests directory doesn't exist or is inaccessible
    }

    // Add README example as a reference
    examples.push({
      path: 'README.md',
      type: 'documentation_example',
      directory: '.',
      description: 'Main example showing image processing pipeline'
    });

    return examples;
  }

  async validateCodeExample(args) {
    const { examplePath } = args;
    
    if (!examplePath) {
      throw new Error('examplePath is required');
    }

    try {
      const fullPath = join(projectRoot, examplePath);
      
      if (!existsSync(fullPath)) {
        return {
          content: [
            {
              type: 'text',
              text: `Example file not found: ${examplePath}`
            }
          ]
        };
      }

      const content = readFileSync(fullPath, 'utf-8');
      
      // Basic validation - check for FKL includes and patterns
      const hasInclude = content.includes('#include <fused_kernel/') || content.includes('fused_kernel');
      const hasNamespace = content.includes('using namespace fk') || content.includes('fk::');
      const hasOperations = content.includes('.then(') || content.includes('executeOperations');
      
      const validation = {
        file: examplePath,
        valid: hasInclude && (hasNamespace || hasOperations),
        checks: {
          hasIncludes: hasInclude,
          hasNamespace: hasNamespace,
          hasOperations: hasOperations
        },
        size: content.length
      };

      return {
        content: [
          {
            type: 'text',
            text: `Code Example Validation:\n\n${JSON.stringify(validation, null, 2)}\n\nContent Preview:\n${content.substring(0, 500)}${content.length > 500 ? '...' : ''}`
          }
        ]
      };
    } catch (error) {
      return {
        content: [
          {
            type: 'text',
            text: `Validation failed: ${error.message}`
          }
        ]
      };
    }
  }

  async run() {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    console.error('FusedKernelLibrary MCP server running on stdio');
  }
}

const server = new FusedKernelLibraryServer();
server.run().catch(console.error);