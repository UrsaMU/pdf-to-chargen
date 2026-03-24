---
name: your-skill-name
description: Brief description of what this Skill does and when Claude should use it. Include specific triggers and keywords. Use third person. Max 1024 characters. Example - "Processes CSV files and generates reports. Use when analyzing data, working with spreadsheets, or when user mentions CSV, data analysis, or reports."
metadata:
  version: 1.0.0
  # Optional: Add dependencies if needed
  # dependencies: python>=3.8, pandas>=1.5.0
# Optional: Restrict which tools Claude can use (CLAUDE CODE ONLY - omit if targeting both platforms)
# allowed-tools: Read, Grep, Glob  # For read-only Skills
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
#
# INSTALLATION:
# - Claude Code: ~/.claude/skills/ (personal) or .claude/skills/ (project)
# - claude.ai: ZIP directory and upload via Settings > Capabilities

# Your Skill Name

## Overview
Explain what this Skill does and its purpose. Be concise—assume Claude is intelligent. Only include information Claude doesn't already have.

## When to Use This Skill
Be specific about triggers and contexts:
- When user mentions [specific keywords]
- When working with [specific file types or contexts]
- When asked about [specific topic or action]

## Instructions
Provide clear instructions. Match detail level to task fragility:
- **High freedom**: Multiple valid approaches → General guidance
- **Low freedom**: Fragile operations → Specific steps

1. **Step 1**: Do this first
2. **Step 2**: Then do this
3. **Step 3**: Finally do this

**For complex workflows, add a checklist:**
```
Progress:
- [ ] Step 1
- [ ] Step 2
- [ ] Step 3
```

## Guidelines
Any specific rules or guidelines Claude should follow:
- Guideline 1
- Guideline 2
- Guideline 3

## Examples

Show concrete examples with actual input/output pairs:

### Example 1: Basic Case
**Input:** User asks: "..."
**Output:**
```
Claude should respond with: "..."
```

### Example 2: Variation
**Input:** User provides: "..."
**Output:**
```
Claude should: "..."
```

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
- [ ] Examples are concrete, not abstract
- [ ] Consistent terminology throughout
- [ ] No Windows-style paths (use forward slashes)
- [ ] No time-sensitive information

### Platform-Specific
- [ ] **If Claude Code only**: Placed in ~/.claude/skills/ or .claude/skills/
- [ ] **If Claude Code only**: allowed-tools specified if restriction needed
- [ ] **If claude.ai only**: Removed allowed-tools field
- [ ] **If claude.ai**: Created ZIP with correct structure
- [ ] **If both platforms**: Omitted Claude Code-only features

### Testing
- [ ] Tested with target models (Haiku, Sonnet, Opus)
- [ ] Description accurately triggers Skill (autonomous invocation works)
- [ ] Tested with queries matching description keywords
- [ ] Verified Skill appears in `"What Skills are available?"` query
- [ ] **If Claude Code**: Tested after restarting Claude Code
- [ ] **If claude.ai**: Uploaded and enabled in Settings > Capabilities

## Notes
Additional context. Keep concise—only what Claude doesn't already know.
