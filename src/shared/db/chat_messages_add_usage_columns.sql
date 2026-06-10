-- Manual migration for existing agent042 databases created before
-- prompt/completion token usage tracking was added to chat_messages.
--
-- This file is intentionally idempotent so it can be applied safely on
-- environments that already have the columns.

ALTER TABLE chat_messages
    ADD COLUMN IF NOT EXISTS prompt_tokens INTEGER,
    ADD COLUMN IF NOT EXISTS completion_tokens INTEGER;
