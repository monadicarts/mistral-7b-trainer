# Contributing to Mistral 7b v0.3 LLM Model Trainer

First off, thank you for considering contributing to the **Mistral 7b v0.3 LLM Model Trainer**! Your help is appreciated, and your contributions will make this project even better.

This document provides guidelines for contributing to the project. Please read it carefully to ensure a smooth and effective collaboration process.

## Table of Contents

- [Contributing to Mistral 7b v0.3 LLM Model Trainer](#contributing-to-mistral-7b-v03-llm-model-trainer)
  - [Table of Contents](#table-of-contents)
  - [How Can I Contribute?](#how-can-i-contribute)
    - [Reporting Bugs](#reporting-bugs)
    - [Suggesting Enhancements](#suggesting-enhancements)
    - [Code Contributions](#code-contributions)
    - [Documentation Improvements](#documentation-improvements)
  - [Development Setup](#development-setup)
  - [Pull Request Process](#pull-request-process)
  - [Coding Style](#coding-style)
  - [Commit Message Guidelines](#commit-message-guidelines)
  - [Code of Conduct](#code-of-conduct)
  - [Getting Help](#getting-help)

## How Can I Contribute?

### Reporting Bugs

If you find a bug, please ensure the bug was not already reported by searching on GitHub under [Issues](https://github.com/monadicarts/mistral-7b-trainer/issues).

If you're unable to find an open issue addressing the problem, [open a new one](https://github.com/monadicarts/mistral-7b-trainer/issues/new). Be sure to include a **title and clear description**, as much relevant information as possible, and a **code sample or an executable test case** demonstrating the expected behavior that is not occurring.

### Suggesting Enhancements

If you have an idea for an enhancement or a new feature, please open an issue to discuss it. This allows us to coordinate efforts and ensure that the proposed changes align with the project's goals.

Provide a clear and detailed explanation of the feature, why it would be beneficial, and any potential implementation ideas.

### Code Contributions

1.  **Fork the repository** on GitHub.
2.  **Clone your fork** locally: `git clone https://github.com/[your-user-name]]/mistral-7b-trainer.git`
3.  **Create a new branch** for your feature or bug fix: `git checkout -b feature/your-feature-name` or `git checkout -b fix/your-bug-fix-name`.
4.  **Set up your development environment** as described in the Development Setup section and in the main `README.md`.
5.  **Make your changes.** Ensure your code follows the project's Coding Style.
6.  **Add or update tests** for your changes.
7.  **Ensure all tests pass:** `python -m pytest`
8.  **Commit your changes** with a clear and descriptive commit message.
9.  **Push your branch** to your fork on GitHub: `git push origin feature/your-feature-name`.
10. **Open a Pull Request (PR)** to the `main` branch of the original [repository](https://github.com/monadicarts/mistral-7b-trainer). Provide a clear description of your changes in the PR.

### Documentation Improvements

Good documentation is *crucial*. If you find areas where the documentation can be improved (e.g., README, code comments, docstrings), please feel free to make a PR with your suggestions.

## Development Setup

Please refer to the Setup section in the main `README.md` file for instructions on setting up the development environment, including creating a virtual environment and installing dependencies.

## Pull Request Process

1.  Ensure any install or build dependencies are removed before the end of the layer when doing a build.
    *   Clarification: Ensure your pull request does not include temporary build artifacts or unnecessary files (e.g., `__pycache__/`, `.pytest_cache/`, local `build/` or `dist/` directories).
2.  Update the [`README.md`](README.md) with details of changes to the interface, this includes new environment variables, exposed ports, useful file locations, and container parameters.
3.  Increase the version numbers in any examples and the `README.md` to the new version that this Pull Request would represent. The versioning scheme we use is SemVer.
4.  You may merge the Pull Request in once you have the sign-off of two other developers, or if you do not have permission to do that, you may request the second reviewer to merge it for you.
5.  Ensure all tests pass (`python -m pytest`) and that new functionality is covered by tests.
6.  Link your PR to any relevant issues.

## Coding Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) -- Style Guide for Python Code.
- We recommend using [Ruff](https://ruff.rs/) for code formatting and linting to maintain consistency.
- Use clear and descriptive variable and function names.
- Write comprehensive docstrings for all modules, classes, and functions.
- Add comments to explain complex or non-obvious parts of the code.
- Ensure your code is well-tested.

## Commit Message Guidelines

Please follow a conventional commit message format to help keep the history clean and automate changelog generation. For example:
- `feat: Add new feature X`
- `fix: Resolve bug Y in module Z`
- `docs: Update README with new instructions`
- `style: Apply Black formatting`
- `refactor: Improve performance of function A`
- `test: Add unit tests for B`
- `chore: Update dependencies`

## Code of Conduct

This project and everyone participating in it is governed by the [Code of Conduct](https://github.com/monadicarts/monadicarts/blob/main/CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to justinfrost@duck.com.

## Getting Help

If you have questions or get stuck while contributing, feel free to open an issue on GitHub and tag it with "question" or "help wanted".

---
✌️ Thank you for contributing!
