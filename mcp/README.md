# FusedKernelLibrary MCP Integration

This directory contains the Model Context Protocol (MCP) server integration for the FusedKernelLibrary, enabling AI agents to interact with and test the library through a standardized interface.

## Overview

The MCP integration provides AI agents with:
- **Tools** to build, test, and validate the library
- **Resources** to access documentation and configuration
- **Examples** to understand usage patterns and API

## Quick Start

### Prerequisites

- Node.js 18.0+ 
- npm or yarn
- FusedKernelLibrary source code
- Optional: CUDA toolkit for GPU functionality

### Installation

```bash
cd mcp
npm install
```

### Running the MCP Server

```bash
npm start
```

The server runs on stdio and communicates using the MCP protocol.

### Testing the Integration

```bash
npm test
```

## Available Tools

### `build_library`
Builds the FusedKernelLibrary using CMake with configurable options.

**Parameters:**
- `buildType`: "Debug" or "Release" (default: "Release")
- `enableCuda`: boolean (default: true)
- `enableCpu`: boolean (default: true)

### `run_tests`
Executes the library test suite.

**Parameters:**
- `testType`: "all", "unit", "standard", or "benchmark" (default: "all")
- `verbose`: boolean (default: false)

### `get_library_info`
Retrieves comprehensive information about the library capabilities.

### `check_cuda_support`
Verifies CUDA availability and configuration on the system.

### `list_examples`
Lists all available code examples and patterns in the library.

### `validate_code_example`
Validates code examples against the library API.

**Parameters:**
- `examplePath`: string (required) - Path to the example file

## Available Resources

### `fusedkernel://readme`
Access to the main README.md documentation.

### `fusedkernel://cmake-config`
Access to the CMakeLists.txt configuration.

### `fusedkernel://examples`
JSON listing of available code examples.

## AI Agent Scenarios

The MCP integration supports several AI agent testing scenarios:

### 1. Library Discovery
- AI agent discovers library capabilities
- Reads documentation and examples
- Understands fusion techniques and API

### 2. Build and Test Workflow
- AI agent checks system requirements (CUDA)
- Builds library with appropriate configuration
- Runs comprehensive test suite

### 3. Code Generation and Validation
- AI agent examines existing examples
- Generates new fusion code patterns
- Validates generated code against library API

## Configuration

The `mcp-config.json` file contains:
- Server configuration
- Resource definitions
- Tool descriptions
- AI agent scenario definitions

## Usage Examples

### Basic MCP Client Interaction

```javascript
// Example MCP request to build the library
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "build_library",
    "arguments": {
      "buildType": "Release",
      "enableCuda": true
    }
  }
}
```

### Resource Access

```javascript
// Example MCP request to read documentation
{
  "jsonrpc": "2.0", 
  "id": 2,
  "method": "resources/read",
  "params": {
    "uri": "fusedkernel://readme"
  }
}
```

## Integration with AI Systems

This MCP server can be integrated with:

- **Claude Desktop** - Add to MCP server configuration
- **VS Code with MCP extensions**
- **Custom AI agents** using MCP client libraries
- **GitHub Copilot** for enhanced code assistance

### Claude Desktop Configuration

Add to your Claude Desktop configuration:

```json
{
  "mcpServers": {
    "fused-kernel-library": {
      "command": "node",
      "args": ["./mcp/server.js"],
      "cwd": "/path/to/FusedKernelLibrary"
    }
  }
}
```

## Development

### Adding New Tools

1. Add tool definition to `setupToolHandlers()` in `server.js`
2. Implement the tool handler method
3. Update the configuration in `mcp-config.json`
4. Add tests to `test/mcp-client-test.js`

### Adding New Resources

1. Add resource definition to `setupResourceHandlers()` in `server.js`
2. Implement the resource reader method
3. Update configuration and documentation

## Troubleshooting

### Common Issues

**Server won't start:**
- Check Node.js version (requires 18.0+)
- Verify all dependencies are installed: `npm install`
- Check for port conflicts

**Build failures:**
- Ensure CMake is available in PATH
- Check CUDA toolkit installation for GPU builds
- Verify compiler compatibility

**Test failures:**
- Build the library first: use `build_library` tool
- Check system requirements (CUDA for GPU tests)
- Review verbose test output

### Debug Mode

Run the server with debug logging:

```bash
NODE_ENV=development node server.js
```

## Contributing

When adding new functionality:

1. Follow the existing code patterns
2. Add appropriate error handling
3. Update documentation
4. Add test cases
5. Ensure MCP protocol compliance

## License

This MCP integration follows the same license as the FusedKernelLibrary project.