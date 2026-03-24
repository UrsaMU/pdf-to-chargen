# Skill Weaver - Technical Reference

This document provides detailed technical information for creating Skills for both Claude Code and claude.ai. It's referenced from SKILL.md but contains supplemental details that aren't needed for every skill creation task.

## Platform Comparison

### Overview

Skills work on two platforms: **Claude Code** (desktop agent) and **claude.ai** (web application). Most features are common, but some are platform-specific.

### Feature Comparison Matrix

| Feature | Claude Code | claude.ai | Notes |
|---------|-------------|-----------|-------|
| **SKILL.md format** | ✅ Yes | ✅ Yes | Identical format |
| **YAML frontmatter** | ✅ Yes | ✅ Yes | Same fields |
| **Progressive disclosure** | ✅ Yes | ✅ Yes | Same behavior |
| **Scripts (Python/JS)** | ✅ Yes | ✅ Yes | Both support |
| **Dependencies** | Auto-install | Install when needed | Both use PyPI/npm |
| **Storage** | Filesystem | ZIP upload | Different methods |
| **Distribution** | Plugin/git | Manual ZIP | Different methods |
| **allowed-tools** | ✅ Yes | ❌ No | Claude Code only |
| **Debug mode** | `claude --debug` | Review thinking | Different approaches |
| **Skill updates** | Restart required | Re-upload ZIP | Different processes |

### When to Use Each Platform

**Claude Code is best for:**
- Team collaboration via git (Project Skills)
- Plugin distribution
- Tool restrictions needed (allowed-tools)
- Local development workflows
- Frequent Skill iteration

**claude.ai is best for:**
- Individual users
- Quick Skill distribution (share ZIP)
- No installation required
- Cross-platform access (any browser)

**Both platforms if:**
- Maximum compatibility desired
- Targeting diverse user base
- Skip Claude Code-only features (allowed-tools)

## Skills Architecture (Common to Both Platforms)

**What are Skills?**
Skills package expertise into discoverable capabilities. Each Skill consists of a SKILL.md file with instructions that Claude reads when relevant, plus optional supporting files like scripts and templates.

**Model-Invoked**: Skills are autonomously triggered by Claude based on your request and the Skill's description. This differs from slash commands (user-invoked).

**Discovery**: Claude uses the `name` and `description` fields to decide when to use a Skill. The description is critical for proper discovery.

**Progressive Disclosure**: Claude loads Skill information in stages to manage context efficiently:
1. Metadata (always loaded): name, description
2. SKILL.md body (when Skill used): core instructions
3. Additional files (as needed): reference docs, templates, scripts

## Storage and Distribution

### Claude Code Storage Locations

Skills in Claude Code can be stored in three locations:

#### Personal Skills (`~/.claude/skills/`)
- Available across all your projects
- For individual workflows and preferences
- Experimental Skills you're developing
- Not shared with team members

#### Project Skills (`.claude/skills/`)
- Shared with your team via git
- For team workflows and conventions
- Automatically available to team members
- Checked into version control

#### Plugin Skills
- Bundled with Claude Code plugins
- Automatically available when plugin installed
- Recommended for team distribution
- Managed through plugin system

### claude.ai Distribution

Skills for claude.ai are distributed as ZIP files:

**Creation:**
```bash
zip -r skill-name.zip skill-name/
```

**Expected structure:**
```
skill-name.zip
  └── skill-name/
      ├── SKILL.md
      ├── reference.md
      └── scripts/
```

**Upload process:**
1. Navigate to Settings > Capabilities
2. Upload ZIP file
3. Enable the Skill
4. Start using it

**Team sharing:**
- Share ZIP via email, file sharing, or git repository
- Each user manually uploads to their account
- No automatic updates (re-upload for changes)

## YAML Frontmatter Specification

### Required Fields

#### name
- **Type**: String
- **Max length**: 64 characters
- **Format**: lowercase-with-hyphens only
- **Restrictions**: Cannot contain "anthropic" or "claude"
- **Recommendation**: Use gerund form (verb + -ing)
- **Examples**:
  - ✅ "processing-pdfs"
  - ✅ "analyzing-spreadsheets"
  - ❌ "Brand Guidelines" (not lowercase-with-hyphens)
  - ❌ "BrandGuidelinesForMarketingTeamAndSalesTeam" (too long, wrong format)

#### description
- **Type**: String
- **Max length**: 1024 characters
- **Purpose**: Claude uses this to determine when to invoke the skill
- **Format**: Third person, includes both what it does AND when to use it
- **Must include**: Specific triggers and keywords
- **Best practices**:
  - Always use third person ("Processes files..." not "I can help...")
  - Be specific about scenarios
  - Mention key domains or contexts
  - Include trigger terms users would mention
- **Examples**:
  - ✅ "Extracts text and tables from PDF files, fills forms, merges documents. Use when working with PDF files or when the user mentions PDFs, forms, or document extraction."
  - ✅ "Analyzes Excel spreadsheets, creates pivot tables, generates charts. Use when analyzing Excel files, spreadsheets, tabular data, or .xlsx files."
  - ❌ "Helps with stuff" (too vague, no triggers)
  - ❌ "I can help you with documents" (first person, vague)

### Optional Fields

**Important for Dual-Platform Compatibility:**

claude.ai enforces strict frontmatter validation and ONLY accepts these top-level keys: `name`, `description`, `license`, `allowed-tools`, `metadata`. Therefore, `version` and `dependencies` must be nested under the `metadata` field to work on claude.ai.

Claude Code is more permissive and will accept the `metadata` field without issues (it may ignore unknown fields). While historical Claude Code documentation showed `version` as a top-level field, using the `metadata` structure ensures compatibility with both platforms.

**Recommended Approach:**
- Use `metadata` field for frontmatter version/dependencies (works on both platforms)
- Include version information in markdown body as well (## Version section)
- This provides redundancy and ensures visibility regardless of platform

#### metadata
- **Type**: Object containing version and dependencies
- **Purpose**: Namespace for version tracking information
- **Required for claude.ai**: Yes, if you need version/dependencies in frontmatter
- **Claude Code**: Accepted but may be ignored; information in markdown body is also parsed
- **Format**:
  ```yaml
  metadata:
    version: 1.0.0
    dependencies: package>=version
  ```

#### version
- **Type**: String
- **Format**: Semantic versioning (MAJOR.MINOR.PATCH)
- **Example**: "1.0.0", "2.1.3"
- **Location**: Must be under `metadata` for claude.ai
- **Recommendation**: Always include for tracking changes

#### dependencies
- **Type**: String (comma-separated list)
- **Format**: `package>=version, another-package>=version`
- **Location**: Must be under `metadata` for claude.ai
- **Examples**:
  - Python: `python>=3.8, pandas>=1.5.0, numpy>=1.21.0`
  - JavaScript: `node>=16.0.0, axios>=1.0.0`
- **Claude Code behavior**:
  - Automatically installs required packages (or asks for permission)
  - Packages must be available from PyPI (Python) or npm (Node.js)
  - Installation happens when Skill is first used

#### allowed-tools
- **Type**: String (comma-separated list of tool names)
- **Purpose**: Restricts which tools Claude can use when this Skill is active
- **Claude Code only**: This field is only supported in Claude Code Agent Skills
- **Behavior**:
  - When specified: Claude can only use listed tools without asking permission
  - When not specified: Claude follows standard permission model
- **Use cases**:
  - Read-only Skills that shouldn't modify files
  - Skills with limited scope (e.g., only data analysis)
  - Security-sensitive workflows
- **Available tools**: Read, Write, Edit, Bash, Grep, Glob, WebFetch, WebSearch, TodoWrite, and others
- **Examples**:
  - Read-only: `allowed-tools: Read, Grep, Glob`
  - Data analysis: `allowed-tools: Read, Bash`
  - Code review: `allowed-tools: Read, Grep, Glob`
- **Example frontmatter**:
```yaml
---
name: safe-file-reader
description: Read files without making changes. Use when you need read-only file access.
allowed-tools: Read, Grep, Glob
---
```

## File Naming Conventions

### Skill Directory
- **Format**: lowercase-with-hyphens
- **Match**: Should match the skill name (normalized)
- **Examples**:
  - Skill name: "Brand Guidelines" → Directory: `brand-guidelines`
  - Skill name: "Data Analysis" → Directory: `data-analysis`

### Required Files
- `Skill.md` - Exact capitalization, always required

### Optional Files
- `REFERENCE.md` - Uppercase, for supplemental information
- `README.md` - Uppercase, for documentation
- `resources/` - Lowercase directory for assets
- `scripts/` - Lowercase directory for executable code

## Progressive Disclosure System

Claude loads skill information in stages:

### Stage 1: Metadata (Always Loaded)
```yaml
name: Brand Guidelines
description: Apply brand guidelines to documents
```
- Loaded for all skills
- Used to decide if skill is relevant
- Keep minimal and focused

### Stage 2: Skill.md Body (Loaded When Skill Used)
- Core instructions and workflows
- Essential examples
- Key guidelines
- When to use this skill

### Stage 3: REFERENCE.md (Loaded On-Demand)
- Detailed technical specifications
- Edge cases and advanced scenarios
- Comprehensive API documentation
- Historical context

### Stage 4: Resources (Loaded As Needed)
- Example files
- Templates
- Data files
- Images and assets

**Why This Matters:**
- Faster skill loading
- Lower token usage
- Better performance
- More maintainable

## Script Integration Patterns

### Python Scripts

#### Basic Pattern
```python
#!/usr/bin/env python3
import argparse
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    # Your logic here

    return 0

if __name__ == '__main__':
    sys.exit(main())
```

#### With Dependencies
```yaml
# In Skill.md frontmatter
dependencies: python>=3.8, pandas>=1.5.0, requests>=2.28.0
```

```python
# In script
import pandas as pd
import requests
```

#### Error Handling
```python
try:
    # Operations
    result = process_data(input_data)
except FileNotFoundError as e:
    print(f"Error: {e}", file=sys.stderr)
    return 1
except ValueError as e:
    print(f"Invalid data: {e}", file=sys.stderr)
    return 2
```

### JavaScript Scripts

#### Basic Pattern
```javascript
#!/usr/bin/env node
const fs = require('fs').promises;

async function main() {
    const args = process.argv.slice(2);
    // Parse args and process

    try {
        // Your logic here
        process.exit(0);
    } catch (error) {
        console.error(`Error: ${error.message}`);
        process.exit(1);
    }
}

main().catch(error => {
    console.error(error);
    process.exit(1);
});
```

#### With Dependencies
```yaml
# In Skill.md frontmatter
dependencies: node>=16.0.0, axios>=1.0.0, cheerio>=1.0.0
```

```javascript
// In script
const axios = require('axios');
const cheerio = require('cheerio');
```

## Resource Organization Patterns

### Small Skills (< 5 files)
```
skill-name/
├── Skill.md
└── resources/
    └── examples/
        ├── example-1.txt
        └── example-2.txt
```

### Medium Skills (5-15 files)
```
skill-name/
├── Skill.md
├── REFERENCE.md
├── resources/
│   ├── examples/
│   ├── templates/
│   └── data/
└── scripts/
    └── process.py
```

### Large Skills (15+ files)
```
skill-name/
├── Skill.md
├── REFERENCE.md
├── README.md
├── resources/
│   ├── examples/
│   │   ├── basic/
│   │   └── advanced/
│   ├── templates/
│   │   ├── minimal.md
│   │   └── complete.md
│   └── data/
│       ├── reference.json
│       └── lookup.csv
└── scripts/
    ├── __init__.py
    ├── process.py
    ├── validate.py
    └── utils.py
```

## Testing Strategies

### Unit Testing (Before Upload)
1. **Syntax Validation**
   - YAML frontmatter is valid
   - All referenced files exist
   - Scripts have correct shebang
   - Scripts are executable

2. **Content Validation**
   - Description is specific and clear
   - Instructions are actionable
   - Examples are complete
   - No hardcoded secrets

3. **Script Testing**
   ```bash
   # Test Python script
   python scripts/process.py --help
   python scripts/process.py --input test.json --output result.json

   # Test JavaScript script
   node scripts/process.js --help
   node scripts/process.js --input test.json --output result.json
   ```

### Integration Testing (After Upload)
1. **Invocation Testing**
   - Try prompts that should trigger the skill
   - Try prompts that shouldn't trigger it
   - Verify Claude loads the skill when expected

2. **Execution Testing**
   - Test with typical inputs
   - Test with edge cases
   - Test with invalid inputs
   - Verify error handling

3. **Thinking Analysis**
   - Review Claude's reasoning
   - Check if skill was considered
   - Verify why it was/wasn't used
   - Refine description if needed

## Common Patterns and Anti-Patterns

### ✅ Good Patterns

#### Focused Skills
```yaml
# Good: One clear purpose
name: CSV Data Validator
description: Validate CSV files against schema and data quality rules
```

#### Clear Instructions
```markdown
## Instructions
1. Load the CSV file using pandas
2. Check column names match schema
3. Validate data types for each column
4. Report any validation errors
```

#### Helpful Examples
```markdown
## Examples
### Example 1: Valid File
Input: users.csv with columns: id, name, email
Output: "✓ All validations passed. 150 rows processed."

### Example 2: Invalid File
Input: users.csv missing 'email' column
Output: "✗ Validation failed: Missing required column 'email'"
```

### ❌ Anti-Patterns

#### Unfocused Skills
```yaml
# Bad: Tries to do too much
name: Data Management
description: Handles all data tasks including validation, processing, reporting, and visualization
```

#### Vague Instructions
```markdown
## Instructions
1. Do the thing with the data
2. Make it work properly
3. Output the results
```

#### Missing Examples
```markdown
## Examples
(No examples provided)
```

## Packaging Checklist

Before creating the ZIP file:

- [ ] Directory name matches skill name (normalized)
- [ ] Skill.md exists with valid YAML frontmatter
- [ ] Name is ≤ 64 characters
- [ ] Description is ≤ 200 characters
- [ ] All referenced files exist at correct paths
- [ ] Scripts have proper shebangs
- [ ] No hardcoded secrets
- [ ] Dependencies are listed in frontmatter
- [ ] Examples are complete and accurate
- [ ] README.md exists (recommended)
- [ ] Version is specified

Creating the ZIP:
```bash
# From parent directory
zip -r skill-name.zip skill-name/

# Verify structure
unzip -l skill-name.zip
```

Expected output:
```
skill-name/Skill.md
skill-name/README.md
skill-name/resources/...
skill-name/scripts/...
```

## Troubleshooting Guide for Claude Code Agent Skills

### Skill Not Being Discovered

**Symptom**: Claude doesn't use the Skill when expected (autonomous invocation failure)

**Claude Code specific checks**:
1. **Verify file location**:
   ```bash
   # Personal Skills
   ls ~/.claude/skills/your-skill-name/SKILL.md

   # Project Skills
   ls .claude/skills/your-skill-name/SKILL.md
   ```

2. **Check YAML syntax**:
   ```bash
   # View frontmatter
   cat ~/.claude/skills/your-skill-name/SKILL.md | head -n 15
   ```
   - Must start with `---` on line 1
   - Must end with `---` before Markdown content
   - No tabs (use spaces for indentation)
   - Quoted strings if they contain special characters

3. **Run debug mode**:
   ```bash
   claude --debug
   ```
   This shows Skill loading errors and discovery decisions.

4. **Restart Claude Code**:
   - Changes to Skills require restarting Claude Code
   - Plugin Skills require plugin reinstall/update

**Diagnosis**:
1. Check if description is too vague
2. Review if trigger terms match user's language
3. Verify Skill is in correct directory

**Solutions**:
- Make description more specific with clear triggers
- Include exact keywords users would mention
- Test with phrases that match description
- Example: If description says "PDF files", ask "Can you help with this PDF?"

### Skill Loads But Doesn't Work

**Symptom**: Skill is invoked but doesn't produce correct results

**Diagnosis**:
1. Review instructions for clarity
2. Check examples are accurate
3. Verify file paths in references

**Solutions**:
- Add more detailed step-by-step instructions
- Include more examples covering edge cases
- Clarify expected inputs/outputs
- Add error handling guidance

### Scripts Won't Execute

**Symptom**: Scripts fail when Claude tries to run them

**Diagnosis**:
1. Check dependencies are listed in frontmatter
2. Verify script has correct shebang
3. Test script independently
4. Review error messages

**Solutions**:
- Add missing dependencies to frontmatter
- Fix script syntax errors
- Ensure file paths are correct
- Add better error messages in script

### Performance Issues

**Symptom**: Skill loads slowly or uses too many tokens

**Diagnosis**:
1. Check Skill.md size
2. Review how much content is in main file vs REFERENCE.md
3. Check if all content is necessary

**Solutions**:
- Move detailed content to REFERENCE.md
- Split large skills into multiple focused skills
- Remove redundant information
- Optimize examples to be concise

## Version Control Best Practices

### Semantic Versioning
- **MAJOR** (1.0.0 → 2.0.0): Breaking changes, incompatible updates
- **MINOR** (1.0.0 → 1.1.0): New features, backward compatible
- **PATCH** (1.0.0 → 1.0.1): Bug fixes, minor improvements

### Documenting Changes
Keep a changelog in README.md:
```markdown
## Version History
- **1.2.0**: Added support for XML input format
- **1.1.1**: Fixed bug in date parsing
- **1.1.0**: Added validation for email fields
- **1.0.0**: Initial release
```

### Testing After Updates
When updating a skill:
1. Test that existing functionality still works
2. Test new features thoroughly
3. Verify description still accurately reflects behavior
4. Update examples if behavior changed
5. Bump version number appropriately

## Advanced Techniques

### Skill Composition
Skills can work together automatically. Design skills to complement each other:

**Example: Data Workflow**
- Skill 1: "CSV Validator" - Validates input data
- Skill 2: "Data Analyzer" - Analyzes valid data
- Skill 3: "Report Generator" - Creates reports from analysis

Claude can chain these together automatically when the user asks for a complete workflow.

### Conditional Logic
Use clear conditional instructions:
```markdown
## Instructions
1. Check the input format:
   - If CSV: Use pandas to read
   - If JSON: Use json module
   - If XML: Use xml.etree.ElementTree

2. If data size > 10MB:
   - Process in chunks
   - Show progress updates

3. If errors found:
   - Log to error file
   - Continue processing remaining data
```

### Context Awareness
Help Claude understand context:
```markdown
## When to Use This Skill
- When user mentions "sales data" or "revenue reports"
- When user provides CSV files with sales metrics
- When user asks to "analyze sales performance"
- NOT for marketing data (use Marketing Analytics skill instead)
```

## Security Deep Dive

### Secret Management
Never hardcode secrets. Instead:

**Bad:**
```python
# NEVER hardcode secrets like this!
SECRET_KEY = "hardcoded-secret-value"  # BAD PRACTICE
```

**Good:**
```python
import os
SECRET_KEY = os.environ.get('SECRET_KEY')
if not SECRET_KEY:
    raise ValueError("SECRET_KEY environment variable not set")
```

**Better:**
Use MCP connections for external services.

### Input Validation
Always validate user inputs:

```python
def validate_input(data):
    # Check type
    if not isinstance(data, dict):
        raise TypeError("Input must be a dictionary")

    # Check required fields
    required = ['name', 'email']
    for field in required:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")

    # Sanitize strings
    data['name'] = data['name'].strip()[:100]

    return data
```

### File Operations
Be careful with file paths:

```python
from pathlib import Path

def safe_file_read(filepath):
    # Convert to Path object
    path = Path(filepath).resolve()

    # Check path is within expected directory
    base_dir = Path('/allowed/directory').resolve()
    if not path.is_relative_to(base_dir):
        raise ValueError("Access denied: Path outside allowed directory")

    # Verify file exists and is readable
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {filepath}")

    return path.read_text()
```

## Performance Optimization

### Token Efficiency
- Keep Skill.md focused and concise
- Move detailed specs to REFERENCE.md
- Use clear, direct language
- Avoid repetition

### Loading Speed
- Minimize file size where possible
- Structure content hierarchically
- Use progressive disclosure
- Reference external files only when needed

### Script Performance
- Cache expensive operations
- Process data in chunks for large files
- Use efficient libraries (pandas, numpy)
- Add progress indicators for long operations

## Evaluation and Iteration

### Build Evaluations First
Create evaluations BEFORE writing extensive documentation to ensure your Skill solves real problems.

**Evaluation-driven development:**
1. **Identify gaps**: Run Claude on tasks without the Skill, document failures
2. **Create evaluations**: Build 3+ scenarios testing these gaps
3. **Establish baseline**: Measure Claude's performance without the Skill
4. **Write minimal instructions**: Create just enough to pass evaluations
5. **Iterate**: Execute evaluations, compare, refine

**Evaluation structure example:**
```json
{
  "skills": ["pdf-processing"],
  "query": "Extract all text from this PDF and save to output.txt",
  "files": ["test-files/document.pdf"],
  "expected_behavior": [
    "Successfully reads the PDF using appropriate tool",
    "Extracts text from all pages without missing content",
    "Saves to output.txt in readable format"
  ]
}
```

### Develop Skills Iteratively with Claude
Use one Claude instance ("Claude A") to create Skills, another ("Claude B") to test them.

**Creating a new Skill:**
1. **Complete task without Skill**: Work through problem with Claude A using normal prompting
2. **Identify reusable pattern**: Notice what context you repeatedly provide
3. **Ask Claude A to create Skill**: "Create a Skill that captures this pattern"
4. **Review for conciseness**: Remove unnecessary explanations
5. **Improve architecture**: Ask Claude A to organize content effectively
6. **Test on similar tasks**: Use Skill with Claude B on related use cases
7. **Iterate based on observation**: Return to Claude A with specifics about Claude B's behavior

**Iterating on existing Skills:**
1. **Use Skill in real workflows**: Give Claude B actual tasks
2. **Observe behavior**: Note where it struggles or succeeds
3. **Return to Claude A**: Share current Skill and describe observations
4. **Review suggestions**: Claude A suggests improvements
5. **Apply and test**: Update Skill, test with Claude B again
6. **Repeat based on usage**: Continue observe-refine-test cycle

**Why this works**: Claude A understands agent needs, you provide domain expertise, Claude B reveals gaps through real usage.

### Observe How Claude Navigates Skills
Pay attention to how Claude uses Skills in practice:
- **Unexpected exploration paths**: Structure might not be intuitive
- **Missed connections**: Links need to be more explicit
- **Overreliance on sections**: Content might belong in main Skill.md
- **Ignored content**: File might be unnecessary or poorly signaled

The `name` and `description` in metadata are critical—Claude uses these to decide whether to trigger the Skill.

## Content Guidelines

### Avoid Time-Sensitive Information
Don't include information that will become outdated:

**Bad (time-sensitive):**
```markdown
If you're doing this before August 2025, use the old API.
```

**Good (use "old patterns" section):**
```markdown
## Current method
Use the v2 API endpoint: `api.example.com/v2/messages`

## Old patterns
<details>
<summary>Legacy v1 API (deprecated 2025-08)</summary>
The v1 API used: `api.example.com/v1/messages`
This endpoint is no longer supported.
</details>
```

### Use Consistent Terminology
Choose one term and use it throughout:

**Good - Consistent:**
- Always "API endpoint"
- Always "field"
- Always "extract"

**Bad - Inconsistent:**
- Mix "API endpoint", "URL", "API route", "path"
- Mix "field", "box", "element", "control"

### Structure Longer Reference Files
For files longer than 100 lines, include table of contents at the top:

```markdown
# API Reference

## Contents
- Authentication and setup
- Core methods (create, read, update, delete)
- Advanced features (batch operations, webhooks)
- Error handling patterns
- Code examples

## Authentication and setup
...
```

## Engagement Detection Patterns

When creating Skills using the adaptive workflow, accurately detecting user engagement level is critical for providing the right experience. This section provides detailed patterns for recognizing and responding to engagement signals.

### Understanding Engagement Levels

**Low Engagement** indicates the user trusts your judgment and wants quick results without detailed collaboration. They're busy, know what they want, and prefer efficiency over involvement.

**Medium Engagement** indicates the user has specific requirements but doesn't want extensive back-and-forth. They know what adjustments they need and want you to implement them efficiently.

**High Engagement** indicates the user wants to understand, influence, or learn from the process. They value collaboration and want to explore options together.

### Language Pattern Recognition

#### Low Engagement Patterns

**Approval signals:**
- "looks good"
- "ship it"
- "that works"
- "perfect"
- "great, thanks"
- "let's go with that"
- "👍" or other positive emoji-only responses
- "yep"
- "sounds right"
- Single-word approvals: "yes", "ok", "good", "fine"

**Urgency signals:**
- "just do it"
- "go ahead"
- "let's move forward"
- "ready when you are"
- "package it up"

**Trust signals:**
- "I trust your judgment"
- "you know best"
- "whatever you think"
- "your call"

**Response strategy for low engagement:**
```
User: "looks good"

Your response:
✓ Package immediately
✓ Provide installation instructions
✓ Offer brief usage examples
✗ Don't ask follow-up questions
✗ Don't suggest additional features
✗ Don't request more feedback
```

#### Medium Engagement Patterns

**Specific change requests:**
- "can you add X?"
- "change the description to..."
- "make it Y instead"
- "tweak the Z section"
- "update the name to..."
- "add error handling for..."

**Focused feedback:**
- "the description seems too vague"
- "I'd prefer X over Y"
- "let's use Z instead"
- "can we target both platforms?"
- "add a template for..."

**Bounded questions:**
- "what format should I use?" (answerable directly)
- "where does this file go?" (straightforward answer)
- "should this be in REFERENCE.md?" (yes/no decision)

**Response strategy for medium engagement:**
```
User: "Can you add type hint checking to the review checklist?"

Your response:
1. Make the specific change requested
2. Present updated section briefly
3. Ask one checkpoint question: "Ready to package, or any other changes?"
4. If approved → package immediately
5. If more changes → repeat cycle
6. After 2-3 iterations → offer to package or continue

✓ Make changes efficiently
✓ Show what changed
✓ Ask if ready or need more
✗ Don't expand scope unprompted
✗ Don't dive into deep explanations
✗ Don't suggest major restructuring
```

#### High Engagement Patterns

**Open-ended questions:**
- "how does X work?"
- "what are the tradeoffs between Y and Z?"
- "should we handle this scenario?"
- "what about edge case A?"
- "is there a better way to..."

**Uncertainty signals:**
- "hmm, not sure about..."
- "I'm wondering if..."
- "would it be better to..."
- "what do you think about..."
- "I'm concerned that..."

**Exploratory language:**
- "let's think about..."
- "what if we..."
- "could we also..."
- "have you considered..."
- "alternatively..."

**Learning interest:**
- "why did you choose X?"
- "can you explain Y?"
- "help me understand Z"
- "what's the reasoning behind..."
- "how does this compare to..."

**Multiple questions in one message:**
- Signals desire for detailed discussion
- Shows investment in understanding
- Indicates collaborative preference

**Response strategy for high engagement:**
```
User: "Should we include async code patterns? What about testing coverage? I'm also wondering if we need separate skills for different Python versions."

Your response:
1. Switch to collaborative checkpoint mode
2. Address questions with options and tradeoffs
3. Ask for input on priorities
4. Break remaining work into smaller decisions
5. Explain reasoning for recommendations
6. Invite continued discussion

✓ Explain options thoroughly
✓ Present tradeoffs clearly
✓ Ask for preferences before proceeding
✓ Offer to explore alternatives
✓ Provide context and reasoning
✗ Don't make big decisions unilaterally
✗ Don't rush to completion
✗ Don't limit discussion
```

### Detecting Mixed Signals

Sometimes user feedback contains conflicting signals. Use these strategies:

#### Approval + New Request Pattern
```
User: "Looks great! Can you also add validation for email fields?"
```

**Analysis:**
- "Looks great" = low engagement signal
- "Can you also..." = medium engagement signal (specific request)

**Classification:** Medium engagement
**Response:** Treat as medium engagement—make the change, then ask if ready to package.

#### Approval + Question Pattern
```
User: "This works well. How does the progressive disclosure actually work in practice?"
```

**Analysis:**
- "This works well" = low engagement signal
- "How does..." = high engagement signal (learning interest)

**Classification:** High engagement (learning interest overrides approval)
**Response:** Provide detailed explanation, ask if they want to explore further, then return to packaging.

#### Minimal Text + Multiple Small Requests
```
User: "change name, add python 3.11, fix typo in line 23"
```

**Analysis:**
- Brief communication = appears low engagement
- Multiple specific items = actually medium engagement

**Classification:** Medium engagement
**Response:** Make all changes efficiently, present results, ask if ready or need more.

#### Enthusiastic Approval + Future Ideas
```
User: "Perfect! This will be so useful. Eventually we should probably add support for async patterns too."
```

**Analysis:**
- "Perfect" = low engagement signal for current work
- "Eventually we should..." = future feature, not immediate request

**Classification:** Low engagement (package now)
**Response:** Package the current Skill, mention that async support could be a future enhancement.

### Transition Strategies

#### From Low to Medium Engagement
Rarely happens, but if user returns with specific feedback after approval:

```
User: "looks good" → [you start packaging]
User: "Actually, can you change the description?"

Response: Stop packaging, make the change, revert to medium engagement mode.
```

#### From Medium to High Engagement
Watch for questions that indicate deeper interest:

```
User: "change description to X" → [you make change]
You: "Updated. Ready to package, or any other changes?"
User: "Why did you structure it with REFERENCE.md instead of putting everything in SKILL.md?"

Response: User is asking "why" questions—shift to high engagement mode, explain reasoning.
```

#### From High to Medium Engagement
User starts giving directives instead of asking questions:

```
User: [multiple open-ended questions, discussing options]
You: [collaborative responses]
User: "OK, let's go with option B and add the validation script."

Response: User has made decision—shift to medium engagement, implement efficiently.
```

#### From High to Low Engagement
User signals they're satisfied and ready to finish:

```
User: [detailed discussion]
You: [collaborative responses]
User: "Great explanation, that makes sense. Let's finalize it."

Response: User is ready to finish—shift to low engagement, package immediately.
```

### Real Conversation Examples

#### Example 1: Clear Low Engagement
```
AI: "I've created a code-reviewer Skill for both platforms. Here's what I built:

Key Decisions:
- Platform: Both (maximum flexibility)
- Scope: Python best practices, security, style
- Description: Triggers on 'Python', 'code review', 'PR'

What would you like to adjust?"

User: "looks good"

AI: [Immediately packages and provides installation instructions]
✓ Correct: Recognized low engagement, moved to completion
```

#### Example 2: Clear Medium Engagement
```
AI: "I've created a data-analyzer Skill. [details]
What would you like to adjust?"

User: "Can you add support for Excel files in addition to CSV?"

AI: [Makes the change]
"Updated to support both CSV and Excel formats via pandas.read_excel(). Ready to package, or any other changes?"

User: "That's it, package it up"

AI: [Packages immediately]
✓ Correct: Recognized medium engagement, made change efficiently, moved to completion
```

#### Example 3: Clear High Engagement
```
AI: "I've created a report-generator Skill. [details]
What would you like to adjust?"

User: "How does this handle different report formats? Should we support both PDF and HTML, or is that too broad for one Skill? I'm also wondering about templating—would it be better to use Jinja2 or just string formatting?"

AI: "Great questions! Let's think through these:

**Report Formats:**
- Single Skill with both: Better for simple reports, users get flexibility
- Separate Skills: Better if formats need very different handling
- Your use case: If reports are similar structure, I'd recommend single Skill with format parameter

**Templating:**
- Jinja2: More powerful, better for complex layouts, requires dependency
- String formatting: Simpler, no dependencies, good for straightforward reports
- Your use case: How complex are your report templates?

What's most important for your workflow?"

User: "Our reports are pretty straightforward, mostly tables and summaries."

AI: "Then I'd recommend:
- Single Skill supporting both PDF and HTML
- String formatting (keeps it simple)
- Add Jinja2 support later if needed

Should I proceed with this approach?"

✓ Correct: Recognized high engagement, provided options/tradeoffs, sought input before proceeding
```

#### Example 4: Mixed Signal Resolution
```
AI: "I've created a csv-validator Skill. [details]
What would you like to adjust?"

User: "Looks good. What happens if the CSV has encoding issues?"

AI: [Recognizes: approval (low) + technical question (medium/high)]
[Interprets as: satisfied with current state, but wants to understand edge case]

"Glad you like it! For encoding issues, the Skill currently uses UTF-8 with error handling that:
1. Tries UTF-8 first
2. Falls back to latin-1 if that fails
3. Reports encoding used in validation output

This covers most cases. Would you like me to add explicit encoding detection (using chardet library), or is the current fallback approach sufficient for your needs?"

User: "Current approach is fine, let's go with it."

AI: [Packages immediately]
✓ Correct: Answered question but detected user was ready to proceed, didn't force more discussion
```

### Common Pitfalls to Avoid

#### Pitfall 1: Over-Collaborating with Low Engagement Users
```
❌ Bad:
User: "looks good"
AI: "Great! Should we add X? What about Y? Have you considered Z?"

✓ Good:
User: "looks good"
AI: [Packages immediately with installation instructions]
```

#### Pitfall 2: Under-Collaborating with High Engagement Users
```
❌ Bad:
User: "How does this work? What are the tradeoffs? Should we...?"
AI: "I'll make that change. Ready to package?"

✓ Good:
User: "How does this work? What are the tradeoffs? Should we...?"
AI: [Detailed explanation, options, invites continued discussion]
```

#### Pitfall 3: Misreading Brevity as Low Engagement
```
User: "add async support"

❌ Bad interpretation: Low engagement, they're terse
✓ Good interpretation: Medium engagement, specific request

Response: Make the change, show results, ask if ready or need more
```

#### Pitfall 4: Treating Questions as Blocking Issues
```
User: "what does progressive disclosure mean?"

❌ Bad: Assume they need extensive education before proceeding
✓ Good: Brief explanation, then: "The Skill uses this already. Want to proceed with packaging or discuss further?"
```

#### Pitfall 5: Forcing a Mode Change Too Early
```
User: [asks one question]

❌ Bad: "I'm switching to collaborative checkpoint mode..."
✓ Good: Answer the question, see if more questions follow before changing approach
```

### Engagement Detection Checklist

When analyzing user feedback, ask:

**Volume indicators:**
- [ ] Is the response short (< 10 words)?
- [ ] Is it a single sentence or multiple paragraphs?
- [ ] Does it contain multiple questions?

**Content indicators:**
- [ ] Pure approval words? → Low
- [ ] Specific change requests? → Medium
- [ ] Open-ended questions? → High
- [ ] "Why/How" questions? → High
- [ ] "Should we" questions? → High

**Tone indicators:**
- [ ] Decisive and directive? → Medium/Low
- [ ] Exploratory and uncertain? → High
- [ ] Trusting and delegating? → Low

**Context indicators:**
- [ ] First piece of feedback after review? → Be conservative, default to medium
- [ ] Follow-up after changes? → Maintain current level or adjust based on new signals
- [ ] Late in conversation? → More likely to shift toward low (fatigue/satisfaction)

### Adaptive Response Templates

**For Low Engagement:**
```markdown
Template:
[Package immediately]
✓ All set! Here's your Skill package: [filename]

**Installation:**
[Brief instructions for their platform]

**Quick test:**
Try: "[example prompt that triggers the Skill]"
```

**For Medium Engagement:**
```markdown
Template:
✓ Updated [specific change made]

[Brief before/after if helpful]

Ready to package, or any other changes?
```

**For High Engagement:**
```markdown
Template:
Great questions! Let's explore these:

**[First topic]:**
- Option A: [brief description, tradeoff]
- Option B: [brief description, tradeoff]
- Recommendation: [with reasoning]

**[Second topic]:**
- [similar structure]

What's most important for your use case? [or] Should I proceed with [recommended approach]?
```

### Summary: Quick Reference

| Signal Type | Engagement Level | Response Action |
|-------------|-----------------|-----------------|
| "looks good", "ship it", 👍 | Low | Package immediately |
| "add X", "change Y to Z" | Medium | Make change → ask if ready |
| "why/how...", "should we...", "what about..." | High | Explain options → seek input |
| Approval + question | High | Answer → offer more discussion |
| Approval + small request | Medium | Make change → ask if ready |
| Multiple questions | High | Collaborative mode |
| Brief directives | Medium | Efficient implementation |
| Single question | Answer → wait for follow-up before mode shift |

**Golden Rule:** When unsure, default to medium engagement and adjust based on the next response.

## Workflow Mode Deep Dive

This section provides advanced guidance on the three workflow modes: Quick Create, Guided Create, and Default (Adaptive). Use this when you need detailed implementation patterns beyond the Skill.md overview.

### Quick Create Mode: Advanced Patterns

**Philosophy:** Maximize autonomy, minimize interaction. Trust your training and make intelligent defaults.

#### Decision-Making in Quick Create

When creating skills in Quick Create mode, use these heuristics:

**Platform selection:**
- Default to "both platforms" unless context clearly indicates otherwise
- If user mentions "Claude Code" or "plugins" → Claude Code only
- If user mentions "upload" or "claude.ai" → claude.ai only

**Scope boundaries:**
- Focus on core use case mentioned
- Don't add "nice to have" features
- Skip edge cases unless critical
- Prefer simple over comprehensive

**Structure choices:**
- Single SKILL.md for most skills
- Only add REFERENCE.md if core instructions would exceed 400 lines
- Skip example files unless essential for functionality
- No scripts unless specifically mentioned

**Description writing:**
- Use most obvious trigger keywords
- Keep to 2-3 sentences maximum
- Don't overthink—user can refine later

#### Quick Create Response Patterns

**Minimal acknowledgment:**
```
✓ Creating [skill-name] skill...

[Progress indicators]

✓ All set! [package details]
```

**Skip explanatory text:**
- Don't explain why you made choices
- Don't describe features in detail
- Don't suggest improvements
- Just deliver the working skill

**Installation instructions format:**
```
**For claude.ai:**
1. Upload [filename]
2. Enable in Settings
3. Try: "[example prompt]"

**For Claude Code:**
1. Extract to ~/.claude/skills/
2. Restart Claude Code
3. Try: "[example prompt]"
```

#### Quick Create Error Recovery

If you need critical information that's missing:

**Ask once, proceed with best guess:**
```
"Creating for both platforms (specify if you need Claude Code only).

[Continue with creation]"
```

**Don't:**
- Ask multiple clarifying questions
- Present options and wait for choice
- Explain tradeoffs

### Guided Create Mode: Advanced Patterns

**Philosophy:** Be a teacher and collaborator. Prioritize understanding over speed.

#### Educational Patterns

**Introduce concepts progressively:**
```
AI: "First, let's understand how Skills work. Claude loads them in stages:
1. Metadata (always)
2. SKILL.md (when invoked)
3. Other files (as needed)

This means we want to keep SKILL.md focused. Does that make sense?"
```

**Explain the "why" behind recommendations:**
```
AI: "I recommend targeting both platforms because:
- More users can use your skill
- No platform-specific code needed
- You can always create platform-specific versions later

However, if you need Claude Code features like 'allowed-tools', we should focus on Claude Code only. What's more important for your use case?"
```

**Connect to user's goals:**
```
AI: "Since you mentioned this is for your team, let's think about distribution:
- Plugin: Best for team-wide deployment
- Project skills: Good for project-specific workflows
- Personal skills: Just for you

Which distribution method fits your team's workflow?"
```

#### Question Patterns in Guided Mode

**Option presentation format:**
```
**[Decision Point]:**

Option A: [Name]
- Pros: [2-3 key benefits]
- Cons: [1-2 limitations]
- Best for: [use case]

Option B: [Name]
- Pros: [2-3 key benefits]
- Cons: [1-2 limitations]
- Best for: [use case]

Recommendation: [Your suggestion with reasoning]

What fits your needs best?
```

**Building on user's answers:**
```
User: "I need it for API testing"

AI: "Great! API testing skills can be structured a few ways:

**Approach 1: Request/Response Validation**
Test that APIs return expected data structures and status codes.

**Approach 2: Integration Testing**
Test full workflows across multiple API calls.

**Approach 3: Contract Testing**
Verify API matches its specification (OpenAPI/Swagger).

Which testing approach matches what you need?"
```

#### Checkpoint Patterns

**Major decision checkpoints:**
```
AI: "So far we've decided:
✓ Platform: Both
✓ Focus: REST API testing
✓ Validation: Response structure and status codes

Before I continue, does this direction feel right, or should we adjust anything?"
```

**Work-in-progress checkpoints:**
```
AI: "I've drafted the description:

'Tests REST APIs by validating requests, checking responses, and verifying status codes. Use when testing APIs, endpoints, or when user mentions REST, HTTP, or API testing.'

This will trigger when you're working with APIs. Want to adjust the wording or triggers?"
```

### Default (Adaptive) Mode: Advanced Patterns

**Philosophy:** Start neutral, adapt quickly. Watch for signals, respond fluidly.

#### The Initial Review Pattern

The initial review after autonomous creation is critical—it sets the tone and provides engagement signals. **Use a concise summary format, not comprehensive review.**

**Structure of effective initial review (concise format):**

1. **Opening (1 line with checkmark):**
   ```
   "✓ Created your [skill-name] Skill! Here's what I built:"
   ```

2. **Files Created (2-3 bullets):**
   ```
   **Files created:**
   - SKILL.md (main instructions)
   - REFERENCE.md (detailed patterns)
   - templates/[template-name].md
   ```

3. **Key Features Summary (3-5 bullets):**
   ```
   **Key features:**
   - [Feature 1 description]
   - [Feature 2 description]
   - [Feature 3 description]
   - [Feature 4 description]
   ```

4. **Download Link (prominent):**
   ```
   **Download:** [Download skill-name.zip]
   ```

5. **Offer to explain + feedback invitation:**
   ```
   "I can explain any decisions if you'd like. What would you like to adjust?"
   ```

**Why this works:**
- Digestible summary prevents information overload
- Files created gives visibility into what was built
- Key features focus on value, not implementation details
- Download link is prominent and easy to find
- Offer to explain makes decision details available on request
- User's response clearly indicates engagement level

#### Reading Initial Responses

**Length indicators:**
| Response Length | Likely Engagement | Next Action |
|----------------|-------------------|-------------|
| < 5 words | Low | Package immediately |
| 5-15 words | Medium | Make change, checkpoint |
| 15-30 words | Medium-High | Assess content, not just length |
| 30+ words | High | Switch to collaborative |
| Question(s) | Medium-High | Answer, then assess |

**Content overrides length:**
- "looks good thanks!" (13 words) → Low engagement (approval)
- "add X, Y, Z" (3 words) → Medium engagement (directives)
- "why X?" (2 words) → High engagement (learning question)

#### Adapting Mid-Conversation

**Detecting engagement shifts:**

Early signal → late signal pattern:
```
User's first response: "looks good" [low]
User's second response: "actually, can you add error handling?" [medium]

Interpretation: User reviewed more carefully, found something to improve.
Action: Treat as medium engagement, make change efficiently.
```

Question → directive pattern:
```
User's first response: "how does progressive disclosure work?" [high]
You: [detailed explanation]
User's second response: "got it, let's add that to the skill" [medium]

Interpretation: User learned what they needed, now ready to execute.
Action: Shift to medium engagement, implement efficiently.
```

Multiple iterations pattern:
```
Iteration 1: "add X" [medium]
You: [add X]
Iteration 2: "change Y" [medium]
You: [change Y]
Iteration 3: "tweak Z" [medium]

After 3+ iterations: "Ready to package, or keep refining?"
```

#### Artifact Creation Patterns

**Create artifacts for real-time visibility during skill building.**

Users prefer seeing files as they're created rather than just progress text. Use the Write tool for each file to open artifact panes.

**What to create as artifacts:**
- SKILL.md (always)
- REFERENCE.md (if included)
- Templates (*.md files in templates/)
- Scripts (*.py, *.js files in scripts/)

**When to create them:**
During Phase 1 (Autonomous Create), as you build each component.

**Example workflow:**
```
1. Show progress: "Creating Skill structure... ✓"
2. Write SKILL.md using Write tool → opens artifact pane
3. Show progress: "Writing description and metadata... ✓"
4. Write REFERENCE.md using Write tool → opens artifact pane
5. Show progress: "Adding templates... ✓"
6. Write template files using Write tool → opens artifact panes
7. Continue until all files created
```

**Why this works:**
- Users see exactly what's being created in real-time
- Artifact panes provide easy review without scrolling
- Visual feedback builds confidence
- Users can inspect files before the final summary

**Pattern:**
```python
# Instead of just saying "creating SKILL.md"
# Actually create it with Write tool:

Write(
  file_path="/path/to/skill-name/SKILL.md",
  content="[full skill content]"
)

# This opens an artifact pane showing the file
# User can see it being created in real-time
```

#### Download Link Management

**Keep download links visible and accessible throughout the conversation.**

Download links get lost in scrollback after detailed explanations. Re-emphasize them to improve UX.

**When to repeat download links:**

1. **After answering questions:**
   ```
   [User asks: "How does validation work?"]

   Your response:
   "Validation uses a three-step process:
   1. Schema checking
   2. Data type validation
   3. Custom rule evaluation

   [Detailed explanation...]

   **Download:** [Download api-testing.zip]

   Any other questions, or ready to use it?"
   ```

2. **After making changes:**
   ```
   [User: "Can you add GraphQL support?"]

   Your response:
   "✓ Updated to include GraphQL support

   I've added:
   - GraphQL query validation
   - Schema introspection support
   - Example GraphQL tests

   **Download:** [Download api-testing.zip]

   Ready to go, or any other changes?"
   ```

3. **After detailed explanations:**
   ```
   [User asks multiple questions, you provide in-depth answers]

   At the end:
   "...and that's how the progressive disclosure works.

   **Download:** [Download skill-name.zip]

   Does that clarify things? Any other questions?"
   ```

**Format consistency:**
Always use the same format for download links:
```
**Download:** [Download skill-name.zip]
```

**Why this works:**
- Links don't get buried in conversation
- Users don't have to scroll back to find the artifact
- Reduces friction and frustration
- Shows consideration for user experience

**Don't:**
- Provide link only once at the beginning
- Hide links in dense paragraphs
- Assume users remember where the link is

**Do:**
- Repeat link after substantial responses
- Make it visually distinct with bold formatting
- Include it before asking next question

### Mode Selection Decision Tree

Use this tree when user's request is ambiguous:

```
User requests skill creation
    |
    ├─ Mentions "quick" or "fast" or "just do it"
    │  └─> Quick Create Mode
    |
    ├─ Mentions "walk me through" or "help me learn"
    │  └─> Guided Create Mode
    |
    ├─ Asks clarifying questions before you start
    │  └─> Guided Create Mode (user wants collaboration)
    |
    ├─ Provides detailed requirements upfront
    │  └─> Default Mode (they know what they want)
    |
    └─ Standard request, no mode indicators
       └─> Default Mode (adapt from feedback)
```

### Common Mode Selection Mistakes

#### Mistake 1: Over-interpreting single words

```
❌ Bad:
User: "Create a skill for testing"
AI: "I notice you didn't say 'quick', so I'll walk you through every decision..."

✓ Good:
User: "Create a skill for testing"
AI: [Uses default mode, adapts from feedback]
```

#### Mistake 2: Ignoring explicit mode requests

```
❌ Bad:
User: "Just quickly make a skill for API testing"
AI: "Let's think through the requirements together. First, what APIs..."

✓ Good:
User: "Just quickly make a skill for API testing"
AI: "Creating api-testing skill...
[Quick autonomous creation]"
```

#### Mistake 3: Staying in wrong mode after shift signal

```
❌ Bad:
User: "Walk me through creating a data skill"
AI: [Guided mode, asking questions]
User: "Actually, just make the standard decisions and show me"
AI: "Sure! Now let's discuss the next choice: should we..."

✓ Good:
User: "Walk me through creating a data skill"
AI: [Guided mode, asking questions]
User: "Actually, just make the standard decisions and show me"
AI: "Understood! Switching to autonomous creation...
[Proceeds with Quick Create approach]"
```

#### Mistake 4: Forcing mode declaration

```
❌ Bad:
User: "Create a skill for code reviews"
AI: "Would you like Quick Create or Guided Create mode?"

✓ Good:
User: "Create a skill for code reviews"
AI: [Creates autonomously, adapts from feedback—Default Mode]
```

### Hybrid Patterns

Sometimes you'll need to blend approaches:

**Quick Create with one clarification:**
```
User: "Just make a skill for processing documents"

AI: "Creating document-processing skill for both platforms...

One quick question: PDF, Word, or both?
(I'll default to both if you don't specify)"

[If no response in context, proceed with both]
```

**Guided Create with time limits:**
```
User: "Walk me through creating an API skill, but I only have 10 minutes"

AI: "Got it! I'll guide you through the key decisions quickly.

Quick decision 1: Platform?
- Both (recommended, 2 min to set up)
- Claude Code only (3 min, more options)

[Fast-paced guided mode]"
```

**Default Mode with learning moments:**
```
[Default mode creation]
User: "what's REFERENCE.md for?"

AI: "REFERENCE.md is for detailed content that doesn't need to be in the main SKILL.md. Claude loads it on-demand when needed, keeping the primary skill file focused.

For your api-testing skill, I put the main workflow in SKILL.md and detailed HTTP status code reference in REFERENCE.md.

[Returns to adaptive mode without forcing deeper discussion]"
```

### Summary: Mode Selection Quick Reference

| User Signal | Mode | Key Behaviors |
|------------|------|---------------|
| "just make it", "quickly" | Quick Create | Minimal questions, autonomous, immediate packaging |
| "walk me through", "help me learn" | Guided Create | Frequent questions, educational, collaborative |
| Standard request | Default (Adaptive) | Autonomous create, comprehensive review, adapt from feedback |
| "actually, just..." mid-conversation | Switch to Quick | Honor the mode change immediately |
| Asks "why" or "how" questions | Switch to Guided | Provide explanations, invite more questions |
| Multiple approvals | Stay in current | Don't overthink, respect user's pattern |

**Remember:** Modes are guidelines, not rigid rules. Prioritize user's immediate signals over mode classification.

## Advanced: Skills with Executable Code

### Solve, Don't Punt
When writing scripts, handle error conditions rather than punting to Claude.

**Good (handles errors):**
```python
def process_file(path):
    """Process a file, creating if it doesn't exist."""
    try:
        with open(path) as f:
            return f.read()
    except FileNotFoundError:
        print(f"File {path} not found, creating default")
        with open(path, 'w') as f:
            f.write('')
        return ''
    except PermissionError:
        print(f"Cannot access {path}, using default")
        return ''
```

**Bad (punts to Claude):**
```python
def process_file(path):
    # Just fail and let Claude figure it out
    return open(path).read()
```

Document configuration parameters (avoid "voodoo constants"):

**Good:**
```python
# HTTP requests typically complete within 30 seconds
# Longer timeout accounts for slow connections
REQUEST_TIMEOUT = 30

# Three retries balances reliability vs speed
# Most intermittent failures resolve by second retry
MAX_RETRIES = 3
```

**Bad:**
```python
TIMEOUT = 47  # Why 47?
RETRIES = 5   # Why 5?
```

### Provide Utility Scripts
Pre-made scripts offer advantages over generated code:
- More reliable
- Save tokens (no code in context)
- Save time (no generation)
- Ensure consistency

Make clear whether Claude should:
- **Execute script** (most common): "Run analyze_form.py to extract fields"
- **Read as reference** (for complex logic): "See analyze_form.py for the algorithm"

### Use Visual Analysis
When inputs can be rendered as images, have Claude analyze them:

```markdown
## Form layout analysis
1. Convert PDF to images: `python scripts/pdf_to_images.py form.pdf`
2. Analyze each page image to identify form fields
3. Claude can see field locations and types visually
```

### Create Verifiable Intermediate Outputs
For complex, open-ended tasks, use "plan-validate-execute" pattern:

**Example: Updating 50 form fields**
Without validation, Claude might reference non-existent fields or miss required fields.

**Solution: Add intermediate validation**
1. Analyze form
2. Create plan file (changes.json)
3. Validate plan with script
4. Execute changes
5. Verify output

**Why this works:**
- Catches errors early
- Machine-verifiable
- Reversible planning
- Clear debugging

Make validation scripts verbose with specific error messages.

### Package Dependencies
List required packages in Skill.md. Skills run in code execution environment:
- **claude.ai**: Can install from npm and PyPI, pull from GitHub
- **Anthropic API**: No network access, no runtime installation

### MCP Tool References
Always use fully qualified tool names to avoid "tool not found" errors.

**Format:** `ServerName:tool_name`

**Example:**
```markdown
Use the BigQuery:bigquery_schema tool to retrieve table schemas.
Use the GitHub:create_issue tool to create issues.
```

Without the server prefix, Claude may fail to locate the tool.

### Avoid Assuming Tools Are Installed
Don't assume packages are available:

**Bad:**
```markdown
"Use the pdf library to process the file."
```

**Good:**
```markdown
"Install required package: `pip install pypdf`

Then use it:
```python
from pypdf import PdfReader
reader = PdfReader("file.pdf")
```
```

### Runtime Environment
Skills run in code execution environment with filesystem access, bash commands, and code execution.

**How this affects authoring:**
- **Metadata pre-loaded**: name and description loaded at startup into system prompt
- **Files read on-demand**: Claude uses bash/Read tools to access Skill.md and other files when needed
- **Scripts executed efficiently**: Scripts can run via bash without loading full contents into context
- **No context penalty for large files**: Reference files don't consume tokens until read
- **File paths matter**: Claude navigates like a filesystem—use forward slashes
- **Name files descriptively**: Use `form_validation_rules.md`, not `doc2.md`
- **Organize for discovery**: Structure by domain/feature (good: `reference/finance.md`, bad: `docs/file1.md`)

## Checklist for Effective Skills

Before sharing a Skill, verify:

### Core Quality
- [ ] Description is specific and includes key terms
- [ ] Description includes both what Skill does and when to use it
- [ ] Name uses lowercase-with-hyphens format
- [ ] Name is ≤ 64 characters
- [ ] Description is ≤ 1024 characters
- [ ] Skill.md body is under 500 lines
- [ ] Additional details are in separate files (if needed)
- [ ] No time-sensitive information (or in "old patterns" section)
- [ ] Consistent terminology throughout
- [ ] Examples are concrete, not abstract
- [ ] File references are one level deep
- [ ] Progressive disclosure used appropriately
- [ ] Workflows have clear steps

### Code and Scripts
- [ ] Scripts solve problems rather than punt to Claude
- [ ] Error handling is explicit and helpful
- [ ] No "voodoo constants" (all values justified)
- [ ] Required packages listed and verified as available
- [ ] Scripts have clear documentation
- [ ] No Windows-style paths (all forward slashes)
- [ ] Validation/verification steps for critical operations
- [ ] Feedback loops included for quality-critical tasks

### Testing
- [ ] At least three evaluations created
- [ ] Tested with Haiku, Sonnet, and Opus
- [ ] Tested with real usage scenarios
- [ ] Team feedback incorporated (if applicable)
- [ ] Description accurately triggers Skill
- [ ] Claude loads Skill when expected
- [ ] Skill produces correct results

## Support and Resources

- **Official Documentation**: https://docs.claude.com/claude/docs/skills
- **Example Skills**: https://github.com/anthropics/skills
- **Best Practices Guide**: https://docs.claude.com/claude/docs/skill-authoring-best-practices
- **Community Forum**: Check Anthropic's community resources

## Glossary

**Frontmatter**: YAML metadata at the start of Skill.md

**Progressive Disclosure**: Loading information in stages as needed

**Skill Composition**: Multiple skills working together automatically

**MCP**: Model Context Protocol for external service integration

**Semantic Versioning**: Version numbering scheme (MAJOR.MINOR.PATCH)

**Token**: Unit of text processed by Claude (affects performance and cost)

**Evaluation-Driven Development**: Creating tests before writing documentation to ensure Skills solve real problems

**Utility Scripts**: Pre-made scripts bundled with Skills for reliable, token-efficient operations
