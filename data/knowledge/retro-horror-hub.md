# Retro Horror Hub

## What It Is
Retro Horror Hub is a platform Arin designed and built for 80s–90s horror fans and collectors, combining film data, media content, and collectible listings into a single structured experience. He approached it as a product — defining the user need, shaping the information architecture, and focusing on real collector value rather than treating it as a pure technical experiment.

## Technical Highlights
The platform integrates multiple external APIs including TMDB, YouTube, eBay, Discogs, Spotify, and Reddit. The core challenge was not simple aggregation but designing a curated data layer that transforms fragmented API responses into meaningful, usable information for collectors. Each movie page becomes a unified hub where metadata, trailers, vinyl pressings, and collectible items are connected in a consistent and purpose-driven way. The stack is Next.js, Supabase, Prisma, and Netlify, with NextAuth/Google OAuth for community contributions.

## eBay Curation Algorithm
A custom eBay curation algorithm scores and filters listings by collectible value, handles sequel detection, franchise grouping, and bonus passes for high-value items. This was built from scratch rather than using an off-the-shelf solution, ensuring only meaningful and high-quality listings surface to users — which matters both for user experience and for the eBay affiliate revenue model.

## Monetisation and Product Thinking
The platform is monetised through the eBay Partner Network affiliate program, turning discovery into a functional revenue model. This added real-world constraints around usability, relevance, and click-through value. Arin also worked on SEO, social media content creation for the collector audience, and outreach to horror blogs and collector communities to grow the site organically.