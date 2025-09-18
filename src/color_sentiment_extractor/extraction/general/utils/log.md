# Logger Utility

This module provides a lightweight debug logger for the project, controlled entirely by an environment variable.  
It is designed to be minimal, flexible, and safe to use across different parts of the codebase.

## Features
- 🔍 **Topic-based filtering**: log only messages from the topics you want (e.g. `extraction`, `config`).
- 🌐 **Environment variable control**:  
  - Set `CHATBOT_DEBUG_TOPICS=all` to log everything.  
  - Set `CHATBOT_DEBUG_TOPICS=extraction,config` to restrict output to these topics.  
  - Default (empty) → no filtering, everything is logged.
- ⏱️ **Timestamps**: every message is prefixed with the current time (`HH:MM:SS`).
- 🏷️ **Log levels**: simple `level` argument (defaults to `DEBUG`) for tagging messages.
- 🔄 **Dynamic reload**: call `reload_topics()` to pick up changes to `CHATBOT_DEBUG_TOPICS` without restarting the process.
- ⚡ **Flexible output**: messages are written to `sys.stderr` by default, but can be redirected via the `stream` argument.

## Functions
- `_load_topics()` → Internal helper to parse `CHATBOT_DEBUG_TOPICS` env var into a set of topics.
- `reload_topics()` → Reload topics from the environment variable at runtime.
- `debug()` → Print a timestamped line with topic and level if enabled by `CHATBOT_DEBUG_TOPICS`.

## Example Usage

```bash
# Log everything
export CHATBOT_DEBUG_TOPICS=all

# Log only extraction and config
export CHATBOT_DEBUG_TOPICS=extraction,config


from Chatbot.extraction.general.utils.log import debug, reload_topics

debug("starting extraction", topic="extraction")   # will show if extraction enabled
debug("loaded config.json", topic="config")       # will show if config enabled
debug("this is hidden", topic="other")            # hidden unless "other" or "all" enabled

# Update env and reload topics dynamically
import os
os.environ["CHATBOT_DEBUG_TOPICS"] = "all"
reload_topics()
debug("now everything is visible")
