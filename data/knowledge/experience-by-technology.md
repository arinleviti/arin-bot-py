# Arin's Experience by Technology

## RAG (Retrieval-Augmented Generation)
Arin has hands-on experience building RAG systems in three projects. In the Hotel Automation Agent he built agentic RAG with Docling parsing, contextual retrieval, Voyage AI embeddings, pgvector and a reranker. In the Flight Disruption Assistant a compensation agent uses RAG over EU261 law stored in ChromaDB. And this chatbot is his own RAG pipeline in Python with ChromaDB, FastAPI and Groq.

## LangGraph
Arin has experience with LangGraph. He built the Hotel Automation Agent with it: a LangGraph graph with subgraphs for each email type, human-in-the-loop approval with interrupt(), a Postgres checkpointer so paused runs survive restarts, and retry policies and error handlers on each node. In the Flight Disruption Assistant he chose to write the multi-agent orchestration loop by hand instead.

## AI Agents and Multi-Agent Systems
Arin has experience building AI agents and multi-agent systems. At Wonderful AI he built node-based TypeScript agents for chat, voice and email. The Flight Disruption Assistant is a multi-agent system: a supervisor agent calls a rebooking agent and a compensation agent as tools. The Hotel Automation Agent combines a deterministic LangGraph workflow with one tool-calling agent that searches documents.

## Tool Calling and Structured Outputs
Arin has experience with LLM tool calling (function calling) and structured outputs. His agents call tools such as flight search, booking and document search, and return validated JSON checked with Pydantic schemas. In the Hotel Automation Agent, Claude extracts invoice lines and booking details as structured output. This chatbot uses tool calling to email Arin when a visitor wants to get in touch.

## Vector Databases and Embeddings
Arin has experience with vector databases and embeddings: ChromaDB in this chatbot and in the Flight Disruption Assistant, and PostgreSQL with pgvector and an HNSW index in the Hotel Automation Agent. He has used Voyage AI embeddings and reranking, and sentence-transformer embeddings (all-MiniLM-L6-v2), with cosine similarity search.

## LLM Evals and Testing
Arin has experience building LLM evals. The Hotel Automation Agent has an eval suite of 168 code checks across 3 runs of 10 demo emails, all passing, plus pytest unit tests with fake LLMs. The Flight Disruption Assistant has 19 scripted scenarios with 253 checks on the case file, and he used eval results to choose which model runs each agent.

## LLM Observability and Langfuse
Arin has experience with LLM observability using Langfuse. In both the Hotel Automation Agent and the Flight Disruption Assistant he traced every model call, tool call and agent step, with tokens and cost. Reading traces is how he found and fixed most bugs in the flight assistant, such as a wrong tool schema and misleading error messages.

## Guardrails, Hallucinations and Prompt Injection
Arin has experience reducing hallucinations and adding guardrails. His rule is "the model judges, code calculates": prices, amounts and dates are computed in code, never by the LLM. The Flight Disruption Assistant has code guards that block unconfirmed bookings and invented flight numbers. He red-teamed it with prompt-injection and jailbreak attacks, and the Hotel Automation Agent flags suspicious emails for a human.

## Prompt Engineering and Prompt Caching
Arin has experience with prompt engineering and prompt caching. He keeps prompts as versioned Markdown files, and uses Anthropic prompt caching in the Flight Disruption Assistant, which cut conversation cost by about 60-70%. His experience is that reliability comes more from engineering constraints around the model than from prompting alone.

## LLM Providers and Models
Arin has worked with Anthropic Claude (Claude Haiku 5.5 via the Anthropic SDK), OpenAI's open-weight gpt-oss-120b and gpt-oss-20b on Groq, and Gemini. He uses LiteLLM for multi-provider setups with automatic model fallbacks, and Voyage AI for embeddings and reranking.

## Python and FastAPI
Arin has experience with Python and FastAPI. This chatbot, the Hotel Automation Agent and the Flight Disruption Assistant all have Python FastAPI backends; the hotel agent streams each step to the browser with Server-Sent Events. He uses Pydantic, uv and pytest.

## Docker, Google Cloud Run and Deployment
Arin has experience with Docker and cloud deployment. This chatbot, the Hotel Automation Agent and the Flight Disruption Assistant all run as Docker containers on Google Cloud Run, with continuous deployment from GitHub or Google Cloud Build for the demos. Separately, without Docker, he hosts frontends on Netlify and his portfolio's C# backend on Azure App Service.

## PostgreSQL and SQL
Arin has experience with PostgreSQL and SQL: Supabase and Prisma in Retro Horror Hub, Neon serverless Postgres with pgvector in the Hotel Automation Agent, and SQLite with an SQL self-join to build connecting flights in the Flight Disruption Assistant.

## Document Parsing and PDFs
Arin has experience with document parsing using Docling, which reads PDFs and Word files while keeping their tables. In the Hotel Automation Agent he used it to read supplier invoice PDFs and the hotel's documents, and its hybrid chunker to split documents by section for RAG.

## MCP (Model Context Protocol)
Arin hasn't shipped an MCP server yet. It is a documented next step for the Hotel Automation Agent: its business logic already lives in plain Python service functions so it can be exposed as MCP tools.
