# Portfolio Website

## About the Site
Arin's portfolio website is available at https://arinleviti.site. It was designed and developed entirely by Arin himself, and serves as a showcase of his work, skills, and progression as a developer and AI engineer. The site itself is a product — not a template, not a theme, but a custom-built application reflecting his technical abilities and design sensibility.

## Tech Stack
The frontend is built with Angular, hosted on Netlify. The backend is built with C# and ASP.NET Core, hosted on Azure App Service. The hero section uses Angular i18n for multilingual translation support. Images are managed through Cloudinary.

## What's on the Site
The site includes a hero section introducing Arin, a projects section showcasing his work including Retro Horror Hub, Hero Burger, and his AI agent work at Wonderful AI, a blog with posts covering his projects and technical learnings, and a contact section. The floating chat widget — this assistant — is integrated directly into the site as a live demonstration of Arin's AI engineering capabilities.

## The Blog
Arin's blog covers his projects and technical experiences in depth. Published posts include: Building Retro Horror Hub (May 2026), Building Reliable AI Agents at Wonderful AI (May 2026), Hero Burger Website: Motion, Brand and Experience (August 2025), and Building a Game Frontend with Angular (April 2025).

## This Chatbot
The chat assistant on this site was built by Arin entirely from scratch in Python. It uses FastAPI as the server framework, ChromaDB as a vector database, and the Groq API (with Llama 3.3 70B) as the language model. It implements a full RAG pipeline: knowledge base documents are chunked by section, embedded using a local model via onnxruntime, stored in ChromaDB, and retrieved by semantic similarity at query time to ground the LLM's responses. Conversation memory is maintained per session. The whole system is containerised with Docker and deployed on Google Cloud Run. Arin built this to demonstrate from-scratch Python and AI engineering skills — not inside a no-code platform, but in real, production-deployed code.