#!/usr/bin/env node
/**
 * Example JavaScript Script Template for Skills
 *
 * This template demonstrates best practices for writing Node.js scripts
 * that can be included in Custom Skills.
 */

const fs = require('fs').promises;
const path = require('path');

/**
 * Parse command line arguments
 * @returns {Object} Parsed arguments
 */
function parseArguments() {
    const args = process.argv.slice(2);
    const parsed = {
        input: null,
        output: null,
        verbose: false
    };

    for (let i = 0; i < args.length; i++) {
        switch (args[i]) {
            case '--input':
                parsed.input = args[++i];
                break;
            case '--output':
                parsed.output = args[++i];
                break;
            case '--verbose':
                parsed.verbose = true;
                break;
            case '--help':
                printHelp();
                process.exit(0);
            default:
                console.error(`Unknown argument: ${args[i]}`);
                printHelp();
                process.exit(1);
        }
    }

    if (!parsed.input || !parsed.output) {
        console.error('Error: --input and --output arguments are required');
        printHelp();
        process.exit(1);
    }

    return parsed;
}

/**
 * Print help message
 */
function printHelp() {
    console.log(`
Usage: node example-script.js [options]

Options:
  --input <path>    Input file path (required)
  --output <path>   Output file path (required)
  --verbose         Enable verbose output
  --help            Show this help message
    `.trim());
}

/**
 * Validate that input file exists
 * @param {string} inputPath - Path to input file
 * @returns {Promise<boolean>} True if valid
 */
async function validateInput(inputPath) {
    try {
        const stats = await fs.stat(inputPath);
        if (!stats.isFile()) {
            console.error(`Error: '${inputPath}' is not a file`);
            return false;
        }
        return true;
    } catch (error) {
        console.error(`Error: Input file '${inputPath}' does not exist`);
        return false;
    }
}

/**
 * Process the input data
 * @param {Object} data - Input data
 * @returns {Object} Processed data
 */
function processData(data) {
    // Add your processing logic here
    return {
        status: 'success',
        inputKeys: Object.keys(data),
        processed: true,
        timestamp: new Date().toISOString()
    };
}

/**
 * Main execution function
 */
async function main() {
    const args = parseArguments();

    // Validate input
    if (!await validateInput(args.input)) {
        process.exit(1);
    }

    try {
        // Read input file
        const inputContent = await fs.readFile(args.input, 'utf8');
        const inputData = JSON.parse(inputContent);

        if (args.verbose) {
            console.log(`Processing ${args.input}...`);
        }

        // Process data
        const result = processData(inputData);

        // Write output
        await fs.writeFile(
            args.output,
            JSON.stringify(result, null, 2),
            'utf8'
        );

        if (args.verbose) {
            console.log(`Results written to ${args.output}`);
            console.log(`Status: ${result.status}`);
        }

        process.exit(0);

    } catch (error) {
        if (error instanceof SyntaxError) {
            console.error(`Error: Invalid JSON in input file: ${error.message}`);
        } else {
            console.error(`Error: ${error.message}`);
        }
        process.exit(1);
    }
}

// Run main function
main().catch(error => {
    console.error(`Unexpected error: ${error.message}`);
    process.exit(1);
});
