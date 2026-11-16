# Wonderful AI

## Role
Arin worked as a freelance AI Agent Engineer at Wonderful AI from October 2025 to October 2026, building LLM-powered agent systems across chat, voice, and email platforms using TypeScript. He came from a web development background and learned LLM systems directly on the job through hands-on development of demos and proof-of-concepts that evolved into production-grade systems.

## Node-Based Architecture
A key part of Arin's work at Wonderful AI was designing AI agent workflows using state machines and structured execution flows. Instead of relying on free-form LLM responses, he broke tasks into modular nodes with clear input/output contracts, making agent behaviour more predictable, traceable, and easier to scale. Each node acts as a controlled execution unit responsible for validating inputs, normalizing outputs, and ensuring safe transitions between steps.

## Reducing Hallucinations
A major challenge in production LLM systems is controlling hallucinations. Arin addressed this by building TypeScript-based nodes that enforce structure at every step of the workflow, significantly improving reliability in real-world usage. A specific challenge was working with Italian-language agents in speech-to-text scenarios, where inputs like "codice fiscale" or email addresses are often misinterpreted. He solved this by implementing structured input nodes that validate and normalize noisy transcriptions before they reach downstream logic.

## Client Work
At Wonderful AI, Arin built agent demos and production systems for enterprise clients across multiple industries, including Luxottica, Poltrona Frau, AXA Italia, Edison Italia, Banca Sella, Pierre Fabre Italia, Unipol, Facile.it, Saipem, and Club del Sole. These span product recommendation agents, insurance comparison tools, energy consumption advisors, banking account-opening flows, and order-to-invoice pipelines.

## Public Demos of This Work
Because client work at Wonderful AI is confidential, Arin built two open-source Python demos that condense the same engineering approach into projects anyone can inspect. The Hotel Automation Agent (https://github.com/arinleviti/hotel-automation-agent) is an email and document automation back office built with LangGraph: subgraphs, human-in-the-loop interrupts, a Postgres checkpointer, agentic RAG with Docling, contextual retrieval, Voyage AI embeddings, pgvector and reranking, Claude via the Anthropic SDK, Langfuse tracing and evals. The Flight Disruption Assistant (https://github.com/arinleviti/flight-disruption-assistant) is a multi-agent chatbot with a supervisor agent and specialist sub-agents (agents as tools), code guardrails, RAG over EU261 law with ChromaDB, LiteLLM model fallbacks across Claude and gpt-oss on Groq, prompt caching, red-teaming and an eval suite. Both are deployed on Google Cloud Run.

## Key Takeaway
The biggest lesson from his time at Wonderful AI is that building reliable AI systems is less about prompting and more about engineering constraints around the model. State machines, structured nodes, and controlled execution flows are what turn LLMs from unpredictable tools into production-ready systems.