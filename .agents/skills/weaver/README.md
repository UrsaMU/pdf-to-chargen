# Skill Weaver - Custom Skill for Creating Skills

A meta-skill that helps you create new Custom Skills for Claude following best practices and proper structure.

> **Note**: Anthropic provides default skills including "skill-builder" and "skill-creator". If you prefer to use Skill Weaver, you can be explicit in your requests (e.g., "use weaver to create a skill for...") or manage enabled skills in settings.

## Overview

Skill Weaver guides you through the entire process of building a Custom Skill, from initial planning to final packaging. It ensures your skills follow the correct structure, include proper metadata, and adhere to best practices.

## What This Skill Does

When activated, Skill Weaver will:

1. **Gather Requirements**: Ask about your skill's purpose, scope, and needs
2. **Create Structure**: Set up the proper directory and file organization
3. **Generate Files**: Create Skill.md with correct YAML frontmatter and content
4. **Add Templates**: Provide starter templates for resources and scripts
5. **Guide Testing**: Help you test and validate your new skill
6. **Provide Best Practices**: Ensure your skill follows recommended patterns

## When to Use

Use this skill when you want to:
- Create a new Custom Skill from scratch
- Build a specialized workflow for Claude
- Package organizational knowledge or guidelines
- Automate repetitive tasks with scripts
- Structure complex instructions for Claude

## Quick Start

Simply ask Claude to create a skill:
```
"Create a new skill for [your purpose]"
"Help me build a skill that does [task]"
"I need a custom skill for [workflow]"
```

### Workflow Modes

Skill Weaver adapts to your preferred interaction style:

**Quick Create** - Fast and autonomous:
```
"Just quickly create a skill for API testing"
"Make a basic code review skill, no need to ask questions"
```

**Guided Create** - Collaborative and educational:
```
"Walk me through creating a data analysis skill"
"Help me learn how to build a skill for testing"
```

**Default (Adaptive)** - Automatically adapts to your engagement:
```
"Create a skill for processing documents"
[Skill is created, then adapts based on your feedback]
```

The skill-creator will:
- Create complete, functional skills autonomously
- Show you what it built with clear decision explanations
- Detect your engagement level from your feedback
- Adapt its approach accordingly (hands-off or hands-on)

## Templates Included

This skill includes several templates to help you get started:

### 1. Basic Skill Template (`templates/basic-skill-template.md`)
Use for simple, instruction-based skills that don't need resources or scripts.

**Best for:**
- Style guides and guidelines
- Simple workflows
- Question-answering skills
- Documentation helpers

### 2. Advanced Skill Template (`templates/advanced-skill-template.md`)
Use for complex skills with resources, scripts, and multiple components.

**Best for:**
- Data processing workflows
- Multi-step automation
- Skills requiring external resources
- Complex integrations

### 3. Example Python Script (`templates/example-script.py`)
Template for Python scripts with proper error handling and argument parsing.

**Best for:**
- Data analysis
- File processing
- API interactions
- Mathematical computations

### 4. Example JavaScript Script (`templates/example-script.js`)
Template for Node.js scripts with async/await patterns.

**Best for:**
- JSON processing
- Web scraping
- File transformations
- Modern async workflows

## Directory Structure

```
weaver-create/
├── Skill.md                    # Main skill file
├── README.md                   # This file
└── templates/
    ├── basic-skill-template.md      # Simple skill template
    ├── advanced-skill-template.md   # Complex skill template
    ├── example-script.py            # Python script template
    └── example-script.js            # JavaScript script template
```

## Key Features

### Progressive Disclosure
The skill teaches Claude's progressive disclosure system:
- Metadata (always loaded)
- Skill.md core content (loaded when skill is used)
- REFERENCE.md (loaded only if needed)
- Additional resources (loaded on-demand)

### Best Practices Enforcement
Ensures your skills follow:
- Focused, single-purpose design
- Clear, specific descriptions
- Proper version control
- Security considerations
- Composability with other skills

### Complete Guidance
Covers all aspects:
- YAML frontmatter requirements
- Markdown structure
- Resource organization
- Script integration
- Testing procedures
- Packaging for distribution

## Example Usage

### Creating a Simple Skill
```
User: "Create a skill for applying our company's coding standards"

Skill Weaver will:
1. Ask about your coding standards
2. Create a focused skill directory
3. Generate Skill.md with your guidelines
4. Provide testing instructions
```

### Creating an Advanced Skill
```
User: "Build a skill that processes CSV data and generates reports"

Skill Weaver will:
1. Gather requirements about data format and report types
2. Set up directory with scripts folder
3. Create Skill.md with proper dependencies
4. Add Python script template for processing
5. Include example inputs/outputs
6. Guide testing and validation
```

## Testing Your New Skill

After creating a skill:

1. **Review Structure**: Verify all files are in place
2. **Check Metadata**: Ensure description is clear and specific
3. **Validate References**: Confirm all file paths are correct
4. **Test Prompts**: Try various prompts that should trigger it
5. **Iterate**: Refine based on how Claude responds

## Tips for Success

### Write Clear Descriptions
The description field is crucial - Claude uses it to decide when to invoke your skill.

**Good:** "Apply Acme Corp brand guidelines to presentations and documents"
**Bad:** "Helps with branding stuff"

### Keep Skills Focused
One skill = one workflow. Create multiple skills instead of one mega-skill.

**Good:** Separate skills for "Generate Reports" and "Validate Data"
**Bad:** One "Data Management" skill that does everything

### Start Simple, Iterate
Begin with basic Markdown instructions. Add complexity only when needed.

1. Start: Basic Skill.md with instructions
2. Add: Example files in resources folder
3. Expand: Scripts for automation
4. Refine: Based on testing and use

### Use Examples Liberally
Examples help Claude understand what success looks like.

Include:
- Example inputs
- Expected outputs
- Edge cases
- Common variations

## Security Reminders

When creating skills with scripts:

⚠️ **Never** hardcode API keys or passwords
⚠️ **Always** use MCP connections for external services
⚠️ **Validate** all inputs in scripts
⚠️ **Document** security requirements

## Version History

- **0.2.2**: Fixed CI workflows to work with new directory structure
  - Updated all GitHub Actions workflows to use weaver-create paths
  - Updated release names from "Skill Creator" to "Skill Weaver"
  - Updated package names from skill-creator-*.zip to weaver-create-*.zip
- **0.2.1**: Renamed directory from "skill-creator" to "weaver-create" to fully avoid conflicts
  - Renamed directory from "skill-creator/" to "weaver-create/"
  - Updated all documentation references to new directory name
  - Added awareness note about both Anthropic "skill-builder" and "skill-creator"
  - Updated all installation paths and examples
- **0.2.0**: Renamed skill from "skill-creator" to "weaver" to avoid conflicts with Anthropic's default skill-builder
  - Changed skill name to "weaver" in frontmatter
  - Updated description to use "weaves" terminology
  - Added awareness note about Anthropic's skill-builder
  - Updated all documentation references
- **0.1.1-beta.1**: UX improvements based on user feedback (beta)
  - Changed Phase 2 from comprehensive review to concise summary
  - Added artifact creation patterns for real-time file visibility
  - Implemented download link repetition pattern to prevent loss in scrollback
  - Updated Initial Review Pattern to focus on digestible summaries
  - Added "offer to explain" pattern instead of upfront detailed explanations
- **0.1.0**: First production release with complete adaptive workflow system
  - Adaptive skill creation workflow (Quick Create, Guided Create, Default modes)
  - Comprehensive engagement detection patterns
  - Progress indicators and decision explanations
  - Mode switching and hybrid patterns
  - Full dual-platform support (Claude Code and claude.ai)
- **0.1.0-alpha.5**: Added adaptive skill creation workflow with engagement detection and progress indicators
- **0.1.0-alpha.4**: Enhanced dual-platform compatibility with version redundancy and improved documentation
- **0.1.0-alpha.3**: Fixed auto-release workflow to extract version from metadata field
- **0.1.0-alpha.2**: Fixed frontmatter structure for claude.ai compatibility, updated documentation
- **0.1.0-alpha.1**: Initial alpha release of Skill Weaver

## Resources

- [Official Skills Documentation](https://docs.claude.com/claude/docs/skills)
- [Example Skills Repository](https://github.com/anthropics/skills)
- [Skill Authoring Best Practices](https://docs.claude.com/claude/docs/skill-authoring-best-practices)

## CI/CD and Releases

This repository includes automated workflows for validation, testing, and releases.

### Automated Validation

Every push and pull request automatically validates the skill structure:

- ✓ YAML frontmatter validation (required fields, length limits)
- ✓ Semantic versioning check
- ✓ Security scanning (hardcoded secrets detection)
- ✓ Python script syntax validation
- ✓ JavaScript script syntax validation
- ✓ Documentation completeness check
- ✓ File reference validation

**Workflow**: `.github/workflows/validate.yml`

### Automated Releases

Releases are automatically created when you push a version tag:

```bash
# Create and push a new version tag
git tag v1.0.0
git push origin v1.0.0
```

The release workflow will:

1. **Validate** the skill structure
2. **Update** version in Skill.md
3. **Package** the skill into a ZIP file
4. **Generate** release notes automatically
5. **Create** a GitHub release with the package
6. **Upload** artifacts for download

**Workflow**: `.github/workflows/release.yml`

### Manual Packaging

You can also package the skill locally using the provided script:

```bash
# Package with default settings
./scripts/package-skill.sh

# Package with custom name
./scripts/package-skill.sh skill-creator skill-creator-1.0.0
```

The script will:
- Validate the skill structure
- Clean up development files
- Create a properly structured ZIP
- Show package information
- Provide next steps

### CI/CD Scripts

The repository includes several helper scripts:

#### Validate Skill
```bash
python .github/scripts/validate-skill.py skill-creator
```
Validates skill structure, frontmatter, and references.

#### Update Version
```bash
python .github/scripts/update-version.py skill-creator 1.2.0
```
Updates the version field in Skill.md frontmatter.

#### Generate Release Notes
```bash
python .github/scripts/generate-release-notes.py skill-creator 1.2.0
```
Generates formatted release notes from skill metadata.

### Release Process

**For maintainers:**

1. **Make your changes** to the skill
2. **Update version** in Skill.md manually or let CI do it
3. **Commit and push** your changes
4. **Create a tag**:
   ```bash
   git tag v1.2.0 -m "Release version 1.2.0"
   git push origin v1.2.0
   ```
5. **GitHub Actions** automatically creates the release
6. **Download** the packaged skill from the release page

**For users:**

1. Go to the [Releases](../../releases) page
2. Download the latest `skill-creator-*.zip` file
3. Upload to Claude via Settings > Capabilities
4. Enable and start using the skill

### Versioning Guidelines

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR** (1.0.0 → 2.0.0): Breaking changes
- **MINOR** (1.0.0 → 1.1.0): New features, backward compatible
- **PATCH** (1.0.0 → 1.0.1): Bug fixes, minor improvements

Examples:
- New template added: MINOR version bump (1.0.0 → 1.1.0)
- Fixed typo in docs: PATCH version bump (1.0.0 → 1.0.1)
- Changed Skill.md structure: MAJOR version bump (1.0.0 → 2.0.0)

## Contributing

To improve this skill:
1. Test with various skill creation scenarios
2. Add more template examples
3. Enhance error handling guidance
4. Expand troubleshooting section

## License

This skill is provided as an example and template for creating Custom Skills for Claude.
