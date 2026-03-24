---
name: your-advanced-skill-name
description: Brief description of what this Skill does and when Claude should use it. Include specific triggers and keywords. Use third person. Max 1024 characters. Example - "Processes CSV files and generates statistical reports with visualizations. Use when analyzing data, creating reports, working with spreadsheets, or when user mentions CSV, data analysis, statistics, or reports."
metadata:
  version: 1.0.0
  dependencies: python>=3.8, pandas>=1.5.0, requests>=2.28.0
# Optional: Restrict which tools Claude can use (CLAUDE CODE ONLY - omit if targeting both platforms)
# allowed-tools: Read, Bash  # For data analysis without file modifications
---

# IMPORTANT: Dual-Platform Compatibility
# This template works on both Claude Code and claude.ai
#
# FRONTMATTER FORMAT:
# - version and dependencies MUST be under "metadata" field (claude.ai requirement)
# - Claude Code accepts this format without issues
# - For dual-platform: omit "allowed-tools" field (Claude Code only)
#
# REQUIREMENTS:
# - Name: lowercase-with-hyphens, max 64 chars, gerund form preferred
# - Description: specific triggers, third person, max 1024 chars
# - File: must be named SKILL.md (uppercase)
# - Keep SKILL.md under 500 lines - move details to REFERENCE.md
#
# DEPENDENCIES:
# - Claude Code: Auto-installs from PyPI/npm (or asks permission)
# - claude.ai: Can install when needed from PyPI/npm
#
# INSTALLATION:
# - Claude Code: ~/.claude/skills/ (personal) or .claude/skills/ (project)
# - claude.ai: ZIP directory and upload via Settings > Capabilities

# Your Advanced Skill Name

## Overview
Explain what this Skill does and problems it solves. Be concise—assume Claude is intelligent. Only include information Claude doesn't already have.

## When to Use This Skill
Be specific about triggers and contexts:
- When user mentions [specific keywords or phrases]
- When processing [specific data types or file formats]
- When integrating with [specific systems or APIs]
- When asked to [specific action or workflow]

## Prerequisites
List any setup requirements:
- Required software or tools
- API keys or credentials (via MCP, never hardcoded)
- Expected file formats or structures
- Environmental setup

## Instructions

**For complex workflows, provide a checklist:**
```
Workflow Progress:
- [ ] Phase 1: Preparation
- [ ] Phase 2: Processing
- [ ] Phase 3: Output
- [ ] Phase 4: Validation
```

### Phase 1: Preparation
1. Validate inputs meet required format
2. Check dependencies are available
3. Load necessary resources

### Phase 2: Processing
1. Execute main workflow steps
2. Handle errors gracefully
3. Log progress and results

### Phase 3: Output
1. Format results appropriately
2. Provide summary of actions taken
3. Suggest next steps if applicable

### Phase 4: Validation (Feedback Loop)
1. Run validation script: `python scripts/validate.py output.json`
2. If validation fails:
   - Review error messages
   - Fix issues
   - Run validation again
3. Only proceed when validation passes

## Scripts

Scripts should solve problems, not punt to Claude. Include explicit error handling and avoid "voodoo constants."

### process_data.py
**Execute this script** to process data files:
```bash
python scripts/process_data.py --input data.csv --output results.json
```

- **Purpose**: What it does
- **Inputs**: What it expects
- **Outputs**: What it produces
- **Error handling**: How it handles failures

### validate.py
**Execute this script** to validate outputs:
```bash
python scripts/validate.py results.json
```

Returns "OK" or lists specific validation errors.

### analyze_format.js
**Execute this script** for format analysis:
```bash
node scripts/analyze_format.js input.json
```

- **Purpose**: What it does
- **Inputs**: What it expects
- **Outputs**: What it produces

**Note**: Always use forward slashes in paths, never backslashes.

## Resources

### Reference Files
- `REFERENCE.md`: Detailed technical specifications
- `resources/examples/`: Sample inputs and outputs
- `resources/templates/`: Templates for common use cases
- `resources/data/`: Reference data files

### External Resources
- Documentation: [Link to relevant docs]
- API Reference: [Link to API docs]
- Related tools: [Link to related resources]

## Guidelines

### Quality Standards
- Output must meet [specific criteria]
- Follow [specific style guide]
- Validate results against [specific requirements]

### Error Handling
- If [condition], do [action]
- For errors in [scenario], [fallback approach]
- Always provide clear error messages

### Performance Considerations
- Optimize for [specific metric]
- Cache [specific data] when possible
- Limit [resource] usage to [threshold]

## Examples

### Example 1: Basic Usage
**Input:**
```
User request: "..."
Data: {...}
```

**Process:**
1. Step by step what happens
2. Including any script calls
3. And resource references

**Output:**
```
Expected result format
```

### Example 2: Advanced Usage
**Input:**
```
Complex scenario description
Multiple data sources
```

**Process:**
1. How this case is handled differently
2. Which scripts are invoked
3. How resources are utilized

**Output:**
```
Expected comprehensive result
```

### Example 3: Edge Case
**Input:**
```
Edge case scenario
```

**Handling:**
- How this is detected
- Special processing applied
- Expected behavior

## Troubleshooting

### Common Issues

#### Issue 1: [Problem description]
**Symptoms:** What the user sees
**Cause:** Why this happens
**Solution:** How to fix it

#### Issue 2: [Problem description]
**Symptoms:** What the user sees
**Cause:** Why this happens
**Solution:** How to fix it

## Security Considerations
- ⚠️ **Never hardcode sensitive data** (API keys, passwords, tokens)
- Use MCP connections for external services (format: `ServerName:tool_name`)
- Validate all user inputs in scripts
- Sanitize outputs before displaying
- Follow principle of least privilege
- Document security requirements clearly

## Best Practices Checklist

Before publishing this Skill:

### Core Quality (Both Platforms)
- [ ] Name uses lowercase-with-hyphens (max 64 chars)
- [ ] Name uses gerund form (processing-pdfs, analyzing-data)
- [ ] Description includes what it does AND when to use it (max 1024 chars)
- [ ] Description written in third person
- [ ] Description includes specific trigger keywords
- [ ] File named SKILL.md (uppercase SKILL)
- [ ] SKILL.md body under 500 lines
- [ ] Detailed content moved to reference.md
- [ ] Examples are concrete, not abstract
- [ ] Consistent terminology throughout
- [ ] File references are one level deep
- [ ] No Windows-style paths (use forward slashes)
- [ ] No time-sensitive information

### Platform-Specific
- [ ] **If Claude Code only**: Placed in ~/.claude/skills/ or .claude/skills/
- [ ] **If Claude Code only**: allowed-tools specified if restriction needed
- [ ] **If claude.ai only**: Removed allowed-tools field
- [ ] **If claude.ai**: Created ZIP with correct structure (skill-name/SKILL.md)
- [ ] **If both platforms**: Omitted Claude Code-only features

### Code and Scripts
- [ ] Scripts solve problems rather than punt to Claude
- [ ] Error handling is explicit and helpful
- [ ] No "voodoo constants" (all values justified with comments)
- [ ] Required packages listed in dependencies
- [ ] Scripts have clear documentation
- [ ] Feedback loops for validation

### Testing
- [ ] Created at least three evaluations
- [ ] Tested with Haiku, Sonnet, and Opus
- [ ] Tested with real usage scenarios
- [ ] Description accurately triggers Skill (autonomous invocation works)
- [ ] Tested with queries matching description keywords
- [ ] Verified Skill appears in `"What Skills are available?"` query
- [ ] Verified dependencies install correctly
- [ ] **If Claude Code**: Tested after restarting Claude Code
- [ ] **If Claude Code**: Tested allowed-tools restrictions (if specified)
- [ ] **If Claude Code**: Verified in correct directory (~/.claude/skills/ or .claude/skills/)
- [ ] **If claude.ai**: Uploaded ZIP and enabled in Settings > Capabilities
- [ ] **If claude.ai**: Verified ZIP structure is correct

## Version History
- **1.0.0**: Initial release with [features]

## Notes
Additional context. Keep concise—only what Claude doesn't already know. For detailed technical specifications, create REFERENCE.md.
