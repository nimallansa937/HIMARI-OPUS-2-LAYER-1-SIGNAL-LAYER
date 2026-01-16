# **Layer 1 Lightweight Web Crawler: Implementation Guide**

## **The Problem We're Solving**

Engine 4 (External Idea Harvester) searches for strategy templates—"RSI below 30, buy the dip" type rules. But strategy generation also benefits from *mechanism understanding*: how do funding rates behave during liquidation cascades? What's the typical spread widening during high volatility? Which technical patterns have decayed in recent years due to crowding?

This knowledge doesn't tell Layer 1 *what to trade*. It tells Layer 1 *what features matter under what conditions*—informing how generated strategies weight and combine the 60-dimensional feature vector.

The lightweight web crawler fills this gap by periodically harvesting market microstructure research, exchange documentation, and quantitative finance discussions. Think of it as building a knowledge base that makes the generation engines smarter, not just giving them more strategy templates to copy.

---

## **What to Search (And What to Ignore)**

**Target Content (High Value for Generation)**

| Category | Examples | Why It Matters |
| ----- | ----- | ----- |
| Market microstructure papers | Funding rate dynamics, liquidation cascade mechanics, order book resilience | Informs feature weighting in strategies |
| Exchange documentation | Fee tiers, rate limits, liquidation engine specs, new products | Execution environment awareness |
| Quantitative blog posts | Research from Paradigm, Delphi, Messari on on-chain dynamics | Identifies which signals have predictive power |
| Academic preprints | arXiv q-fin, SSRN working papers on crypto market structure | Novel mechanisms not yet crowded |
| Historical postmortems | Analysis of past crashes, depegs, liquidation events | Stress scenario understanding |

**Ignore (Already Handled Elsewhere)**

| Category | Why Ignore | Where It's Handled |
| ----- | ----- | ----- |
| Breaking news | Too slow for Layer 1, needs NLP interpretation | Layer 0 → sentiment features |
| Price predictions | Noise, no mechanism understanding | Ignored entirely |
| Social media sentiment | Real-time signal, not research | Layer 0 → Feature 51 |
| Trading signals/alerts | We generate our own | Layer 1 generation engines |
| Promotional content | No research value | Filtered out |

---

## **Architecture Overview**

┌─────────────────────────────────────────────────────────────────────────┐  
│                    LIGHTWEIGHT WEB CRAWLER                              │  
├─────────────────────────────────────────────────────────────────────────┤  
│                                                                         │  
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │  
│  │   Source     │    │   Content    │    │   Knowledge  │              │  
│  │   Crawler    │───▶│   Processor  │───▶│   Extractor  │              │  
│  └──────────────┘    └──────────────┘    └──────────────┘              │  
│         │                   │                   │                       │  
│         ▼                   ▼                   ▼                       │  
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │  
│  │ Rate Limiter │    │  Relevance   │    │   Feature    │              │  
│  │ & Scheduler  │    │   Scorer     │    │   Mapping    │              │  
│  └──────────────┘    └──────────────┘    └──────────────┘              │  
│                                                 │                       │  
│                                                 ▼                       │  
│                                    ┌────────────────────┐              │  
│                                    │  Knowledge Store   │              │  
│                                    │  (Neo4j / SQLite)  │              │  
│                                    └────────────────────┘              │  
│                                                 │                       │  
│                                                 ▼                       │  
│                                    ┌────────────────────┐              │  
│                                    │  Generation Engine │              │  
│                                    │     Conditioning   │              │  
│                                    └────────────────────┘              │  
│                                                                         │  
└─────────────────────────────────────────────────────────────────────────┘

---

## **Implementation**

### **Core Data Structures**

from dataclasses import dataclass, field  
from typing import List, Optional, Dict  
from enum import Enum  
from datetime import datetime  
import hashlib

class ContentType(Enum):  
    """Categories of crawled content."""  
    MICROSTRUCTURE \= "microstructure"      \# Market mechanism research  
    EXCHANGE\_DOCS \= "exchange\_docs"         \# Exchange technical documentation  
    QUANTITATIVE\_RESEARCH \= "quant\_research" \# Analytical blog posts, reports  
    ACADEMIC\_PAPER \= "academic\_paper"       \# arXiv, SSRN preprints  
    POSTMORTEM \= "postmortem"               \# Event analysis, crash forensics

class FeatureRelevance(Enum):  
    """Which feature categories this content informs."""  
    PRICE\_DYNAMICS \= "price"           \# Features 0-14  
    VOLUME\_PATTERNS \= "volume"         \# Features 15-24  
    TECHNICAL\_BEHAVIOR \= "technical"   \# Features 25-34  
    ORDER\_FLOW \= "order\_flow"          \# Features 35-44  
    FUNDING\_CARRY \= "funding"          \# Features 45-49  
    SENTIMENT\_CROSS \= "sentiment"      \# Features 50-54  
    REGIME \= "regime"                  \# Features 55-59

@dataclass  
class CrawledDocument:  
    """A single piece of crawled content."""  
    url: str  
    title: str  
    content: str  
    source: str  
    content\_type: ContentType  
    crawled\_at: datetime  
    published\_at: Optional\[datetime\] \= None  
      
    \# Computed fields  
    content\_hash: str \= field(default="")  
    relevance\_score: float \= 0.0  
    feature\_relevance: List\[FeatureRelevance\] \= field(default\_factory=list)  
      
    def \_\_post\_init\_\_(self):  
        self.content\_hash \= hashlib.sha256(self.content.encode()).hexdigest()\[:16\]

@dataclass  
class ExtractedKnowledge:  
    """Structured knowledge extracted from a document."""  
    document\_hash: str  
    knowledge\_type: str  \# "mechanism", "parameter", "decay", "regime\_behavior"  
      
    \# The actual insight  
    summary: str  
      
    \# Which features this affects  
    relevant\_features: List\[int\]  \# Indices into 60-dim vector  
      
    \# Quantitative details if available  
    parameters: Dict\[str, float\] \= field(default\_factory=dict)  
      
    \# Confidence and recency  
    confidence: float \= 0.5  
    valid\_until: Optional\[datetime\] \= None  \# Some knowledge decays  
      
    \# Source attribution  
    source\_url: str \= ""  
    extraction\_date: datetime \= field(default\_factory=datetime.now)

### **Source Configuration**

@dataclass  
class SourceConfig:  
    """Configuration for a single crawl source."""  
    name: str  
    base\_url: str  
    content\_type: ContentType  
      
    \# Crawl behavior  
    crawl\_frequency\_hours: int \= 24  
    max\_pages\_per\_crawl: int \= 20  
    rate\_limit\_seconds: float \= 2.0  
      
    \# Content selection  
    url\_patterns: List\[str\] \= field(default\_factory=list)  \# Regex patterns to include  
    exclude\_patterns: List\[str\] \= field(default\_factory=list)  \# Patterns to skip  
      
    \# Extraction hints  
    content\_selector: str \= "article"  \# CSS selector for main content  
    date\_selector: Optional\[str\] \= None  
    

\# Pre-configured sources for crypto market microstructure  
DEFAULT\_SOURCES \= \[  
    \# Academic preprints  
    SourceConfig(  
        name="arxiv\_qfin",  
        base\_url="https://arxiv.org/list/q-fin/recent",  
        content\_type=ContentType.ACADEMIC\_PAPER,  
        crawl\_frequency\_hours=24,  
        max\_pages\_per\_crawl=30,  
        url\_patterns=\[r"/abs/\\d+\\.\\d+"\],  
        content\_selector=".abstract",  
    ),  
    SourceConfig(  
        name="ssrn\_crypto",  
        base\_url="https://papers.ssrn.com/sol3/JELJOUR\_Results.cfm?form\_name=journalBrowse\&journal\_id=3312569",  
        content\_type=ContentType.ACADEMIC\_PAPER,  
        crawl\_frequency\_hours=48,  
        max\_pages\_per\_crawl=20,  
        content\_selector=".abstract-text",  
    ),  
      
    \# Quantitative research blogs  
    SourceConfig(  
        name="paradigm\_research",  
        base\_url="https://www.paradigm.xyz/writing",  
        content\_type=ContentType.QUANTITATIVE\_RESEARCH,  
        crawl\_frequency\_hours=24,  
        max\_pages\_per\_crawl=10,  
        content\_selector="article",  
    ),  
    SourceConfig(  
        name="delphi\_research",  
        base\_url="https://members.delphidigital.io/reports",  
        content\_type=ContentType.QUANTITATIVE\_RESEARCH,  
        crawl\_frequency\_hours=24,  
        max\_pages\_per\_crawl=15,  
        content\_selector=".report-content",  
    ),  
      
    \# Exchange documentation  
    SourceConfig(  
        name="binance\_docs",  
        base\_url="https://www.binance.com/en/support/announcement",  
        content\_type=ContentType.EXCHANGE\_DOCS,  
        crawl\_frequency\_hours=12,  \# More frequent—fee changes matter  
        max\_pages\_per\_crawl=20,  
        url\_patterns=\[r"futures", r"margin", r"fee", r"liquidation"\],  
        content\_selector=".article-body",  
    ),  
    SourceConfig(  
        name="bybit\_announcements",  
        base\_url="https://announcements.bybit.com/en-US/",  
        content\_type=ContentType.EXCHANGE\_DOCS,  
        crawl\_frequency\_hours=12,  
        max\_pages\_per\_crawl=20,  
        url\_patterns=\[r"derivatives", r"perpetual", r"funding"\],  
        content\_selector=".article-content",  
    ),  
      
    \# Market microstructure focused  
    SourceConfig(  
        name="kaiko\_research",  
        base\_url="https://www.kaiko.com/research",  
        content\_type=ContentType.MICROSTRUCTURE,  
        crawl\_frequency\_hours=24,  
        max\_pages\_per\_crawl=10,  
        content\_selector=".research-content",  
    ),  
\]

### **The Crawler Engine**

import asyncio  
import aiohttp  
from bs4 import BeautifulSoup  
from urllib.parse import urljoin, urlparse  
import re  
from datetime import datetime, timedelta  
import logging

logger \= logging.getLogger(\_\_name\_\_)

class LightweightWebCrawler:  
    """  
    Lightweight web crawler for market microstructure research.  
      
    Design principles:  
    \- Batch operation, not real-time (runs every 12-24 hours)  
    \- Respectful rate limiting (2+ seconds between requests)  
    \- Content deduplication via hashing  
    \- Focus on mechanism understanding, not news  
      
    Budget: \~$5-10/month (minimal compute, no API costs)  
    """  
      
    def \_\_init\_\_(  
        self,  
        sources: List\[SourceConfig\],  
        storage\_path: str \= "./crawler\_data",  
        user\_agent: str \= "HIMARI-Research-Bot/1.0 (academic research)"  
    ):  
        self.sources \= {s.name: s for s in sources}  
        self.storage\_path \= storage\_path  
        self.user\_agent \= user\_agent  
        self.seen\_hashes: set \= set()  
        self.last\_crawl: Dict\[str, datetime\] \= {}  
          
        \# Load previously seen content hashes  
        self.\_load\_seen\_hashes()  
      
    async def crawl\_all\_sources(self) \-\> List\[CrawledDocument\]:  
        """  
        Crawl all configured sources that are due for refresh.  
          
        Returns list of new documents (not previously seen).  
        """  
        all\_documents \= \[\]  
          
        for name, source in self.sources.items():  
            if self.\_should\_crawl(name, source):  
                logger.info(f"Crawling source: {name}")  
                try:  
                    docs \= await self.\_crawl\_source(source)  
                    new\_docs \= \[d for d in docs if d.content\_hash not in self.seen\_hashes\]  
                      
                    for doc in new\_docs:  
                        self.seen\_hashes.add(doc.content\_hash)  
                      
                    all\_documents.extend(new\_docs)  
                    self.last\_crawl\[name\] \= datetime.now()  
                      
                    logger.info(f"  Found {len(new\_docs)} new documents from {name}")  
                except Exception as e:  
                    logger.error(f"  Failed to crawl {name}: {e}")  
          
        self.\_save\_seen\_hashes()  
        return all\_documents  
      
    def \_should\_crawl(self, name: str, source: SourceConfig) \-\> bool:  
        """Check if source is due for crawling."""  
        if name not in self.last\_crawl:  
            return True  
          
        elapsed \= datetime.now() \- self.last\_crawl\[name\]  
        return elapsed \> timedelta(hours=source.crawl\_frequency\_hours)  
      
    async def \_crawl\_source(self, source: SourceConfig) \-\> List\[CrawledDocument\]:  
        """Crawl a single source with rate limiting."""  
        documents \= \[\]  
          
        async with aiohttp.ClientSession() as session:  
            \# Get index page  
            index\_urls \= await self.\_get\_index\_urls(session, source)  
              
            for url in index\_urls\[:source.max\_pages\_per\_crawl\]:  
                \# Rate limiting  
                await asyncio.sleep(source.rate\_limit\_seconds)  
                  
                try:  
                    doc \= await self.\_fetch\_and\_parse(session, url, source)  
                    if doc and len(doc.content) \> 200:  \# Skip tiny pages  
                        documents.append(doc)  
                except Exception as e:  
                    logger.warning(f"Failed to fetch {url}: {e}")  
          
        return documents  
      
    async def \_get\_index\_urls(  
        self,   
        session: aiohttp.ClientSession,   
        source: SourceConfig  
    ) \-\> List\[str\]:  
        """Extract article URLs from index page."""  
        try:  
            async with session.get(  
                source.base\_url,  
                headers={"User-Agent": self.user\_agent}  
            ) as response:  
                html \= await response.text()  
        except Exception as e:  
            logger.error(f"Failed to fetch index {source.base\_url}: {e}")  
            return \[\]  
          
        soup \= BeautifulSoup(html, 'html.parser')  
        urls \= \[\]  
          
        for link in soup.find\_all('a', href=True):  
            href \= link\['href'\]  
            full\_url \= urljoin(source.base\_url, href)  
              
            \# Check URL patterns  
            if source.url\_patterns:  
                if any(re.search(p, full\_url) for p in source.url\_patterns):  
                    if not any(re.search(p, full\_url) for p in source.exclude\_patterns):  
                        urls.append(full\_url)  
            else:  
                \# No patterns specified—take all links from same domain  
                if urlparse(full\_url).netloc \== urlparse(source.base\_url).netloc:  
                    urls.append(full\_url)  
          
        return list(set(urls))  \# Deduplicate  
      
    async def \_fetch\_and\_parse(  
        self,  
        session: aiohttp.ClientSession,  
        url: str,  
        source: SourceConfig  
    ) \-\> Optional\[CrawledDocument\]:  
        """Fetch and parse a single page."""  
        try:  
            async with session.get(  
                url,  
                headers={"User-Agent": self.user\_agent},  
                timeout=aiohttp.ClientTimeout(total=30)  
            ) as response:  
                if response.status \!= 200:  
                    return None  
                html \= await response.text()  
        except Exception:  
            return None  
          
        soup \= BeautifulSoup(html, 'html.parser')  
          
        \# Extract title  
        title\_tag \= soup.find('title')  
        title \= title\_tag.text.strip() if title\_tag else url  
          
        \# Extract main content  
        content\_elem \= soup.select\_one(source.content\_selector)  
        if not content\_elem:  
            content\_elem \= soup.find('body')  
          
        content \= content\_elem.get\_text(separator=' ', strip=True) if content\_elem else ""  
          
        \# Extract date if selector provided  
        published\_at \= None  
        if source.date\_selector:  
            date\_elem \= soup.select\_one(source.date\_selector)  
            if date\_elem:  
                published\_at \= self.\_parse\_date(date\_elem.text)  
          
        return CrawledDocument(  
            url=url,  
            title=title,  
            content=content,  
            source=source.name,  
            content\_type=source.content\_type,  
            crawled\_at=datetime.now(),  
            published\_at=published\_at  
        )  
      
    def \_parse\_date(self, date\_str: str) \-\> Optional\[datetime\]:  
        """Attempt to parse various date formats."""  
        formats \= \[  
            "%Y-%m-%d", "%B %d, %Y", "%d %B %Y",  
            "%Y/%m/%d", "%m/%d/%Y"  
        \]  
        for fmt in formats:  
            try:  
                return datetime.strptime(date\_str.strip(), fmt)  
            except ValueError:  
                continue  
        return None  
      
    def \_load\_seen\_hashes(self):  
        """Load previously crawled content hashes."""  
        import os  
        hash\_file \= os.path.join(self.storage\_path, "seen\_hashes.txt")  
        if os.path.exists(hash\_file):  
            with open(hash\_file, 'r') as f:  
                self.seen\_hashes \= set(line.strip() for line in f)  
      
    def \_save\_seen\_hashes(self):  
        """Persist seen hashes to disk."""  
        import os  
        os.makedirs(self.storage\_path, exist\_ok=True)  
        hash\_file \= os.path.join(self.storage\_path, "seen\_hashes.txt")  
        with open(hash\_file, 'w') as f:  
            for h in self.seen\_hashes:  
                f.write(h \+ '\\n')

### **Relevance Scoring**

Not all crawled content is equally valuable. The relevance scorer filters and ranks documents before knowledge extraction:

class RelevanceScorer:  
    """  
    Score crawled documents for relevance to strategy generation.  
      
    High relevance: Quantitative mechanisms, specific parameters,   
                    feature behavior analysis, regime dynamics  
      
    Low relevance: News summaries, price predictions, promotional content  
    """  
      
    \# Keywords indicating high-value mechanism content  
    MECHANISM\_KEYWORDS \= {  
        \# Market microstructure  
        "funding rate", "liquidation", "order book", "bid-ask spread",  
        "market maker", "slippage", "execution", "latency",  
          
        \# Quantitative signals  
        "sharpe ratio", "drawdown", "volatility", "correlation",  
        "mean reversion", "momentum", "carry trade", "basis",  
          
        \# Regime dynamics  
        "regime change", "volatility regime", "trend", "range-bound",  
        "bull market", "bear market", "crash", "recovery",  
          
        \# Feature-specific  
        "RSI", "MACD", "bollinger", "order flow", "CVD", "OBV",  
        "open interest", "funding", "perpetual", "futures basis"  
    }  
      
    \# Keywords indicating low-value content  
    NOISE\_KEYWORDS \= {  
        "price prediction", "will reach", "target price", "buy now",  
        "moon", "dump", "pump", "sponsored", "affiliate", "discount code",  
        "not financial advice", "DYOR"  
    }  
      
    \# Feature category keyword mapping  
    FEATURE\_KEYWORDS \= {  
        FeatureRelevance.PRICE\_DYNAMICS: \[  
            "price action", "support", "resistance", "trend", "breakout"  
        \],  
        FeatureRelevance.VOLUME\_PATTERNS: \[  
            "volume", "OBV", "CVD", "accumulation", "distribution"  
        \],  
        FeatureRelevance.TECHNICAL\_BEHAVIOR: \[  
            "RSI", "MACD", "stochastic", "ADX", "indicator"  
        \],  
        FeatureRelevance.ORDER\_FLOW: \[  
            "order book", "bid-ask", "spread", "depth", "liquidity", "microprice"  
        \],  
        FeatureRelevance.FUNDING\_CARRY: \[  
            "funding rate", "perpetual", "basis", "contango", "backwardation",  
            "open interest", "long-short ratio"  
        \],  
        FeatureRelevance.SENTIMENT\_CROSS: \[  
            "sentiment", "fear", "greed", "social", "dominance", "correlation"  
        \],  
        FeatureRelevance.REGIME: \[  
            "regime", "volatility state", "market phase", "cycle", "transition"  
        \]  
    }  
      
    def score(self, document: CrawledDocument) \-\> CrawledDocument:  
        """Score document relevance and identify feature relevance."""  
        text\_lower \= (document.title \+ " " \+ document.content).lower()  
          
        \# Count mechanism keywords  
        mechanism\_hits \= sum(  
            1 for kw in self.MECHANISM\_KEYWORDS   
            if kw in text\_lower  
        )  
          
        \# Count noise keywords  
        noise\_hits \= sum(  
            1 for kw in self.NOISE\_KEYWORDS   
            if kw in text\_lower  
        )  
          
        \# Base score: mechanism density minus noise penalty  
        word\_count \= len(text\_lower.split())  
        mechanism\_density \= mechanism\_hits / max(word\_count / 100, 1\)  
        noise\_penalty \= noise\_hits \* 0.2  
          
        \# Content type bonus  
        type\_bonus \= {  
            ContentType.ACADEMIC\_PAPER: 0.3,  
            ContentType.MICROSTRUCTURE: 0.25,  
            ContentType.QUANTITATIVE\_RESEARCH: 0.2,  
            ContentType.EXCHANGE\_DOCS: 0.15,  
            ContentType.POSTMORTEM: 0.2  
        }.get(document.content\_type, 0\)  
          
        \# Recency bonus (newer \= better, but not critical)  
        recency\_bonus \= 0  
        if document.published\_at:  
            days\_old \= (datetime.now() \- document.published\_at).days  
            if days\_old \< 30:  
                recency\_bonus \= 0.1  
            elif days\_old \< 90:  
                recency\_bonus \= 0.05  
          
        \# Final score (0 to 1\)  
        raw\_score \= mechanism\_density \+ type\_bonus \+ recency\_bonus \- noise\_penalty  
        document.relevance\_score \= max(0, min(1, raw\_score))  
          
        \# Identify feature relevance  
        document.feature\_relevance \= self.\_identify\_features(text\_lower)  
          
        return document  
      
    def \_identify\_features(self, text: str) \-\> List\[FeatureRelevance\]:  
        """Identify which feature categories this document informs."""  
        relevant \= \[\]  
          
        for category, keywords in self.FEATURE\_KEYWORDS.items():  
            if any(kw in text for kw in keywords):  
                relevant.append(category)  
          
        return relevant  
      
    def filter\_relevant(  
        self,   
        documents: List\[CrawledDocument\],   
        min\_score: float \= 0.3  
    ) \-\> List\[CrawledDocument\]:  
        """Filter and sort by relevance."""  
        scored \= \[self.score(doc) for doc in documents\]  
        relevant \= \[doc for doc in scored if doc.relevance\_score \>= min\_score\]  
        return sorted(relevant, key=lambda d: d.relevance\_score, reverse=True)

### **Knowledge Extraction (LLM-Assisted)**

The knowledge extractor uses an LLM to convert unstructured text into structured insights that can condition strategy generation:

class KnowledgeExtractor:  
    """  
    Extract structured knowledge from crawled documents using LLM.  
      
    Extracts:  
    \- Mechanism descriptions (how things work)  
    \- Quantitative parameters (specific numbers)  
    \- Feature behavior patterns (what predicts what)  
    \- Decay observations (what used to work but doesn't)  
    """  
      
    EXTRACTION\_PROMPT \= """You are a quantitative researcher extracting actionable insights from market research.

Document Title: {title}  
Content Type: {content\_type}  
Content:  
{content}

Extract structured knowledge relevant to algorithmic trading strategy generation. Focus on:

1\. MECHANISMS: How do market dynamics work? (e.g., "funding rates mean-revert over 4-8 hours")  
2\. PARAMETERS: Specific quantitative values mentioned (e.g., "liquidation cascades typically last 15-45 minutes")  
3\. FEATURE BEHAVIORS: What predicts what? (e.g., "order book imbalance \>0.3 precedes price moves by 2-5 seconds")  
4\. DECAY OBSERVATIONS: What used to work but doesn't? (e.g., "simple RSI oversold signals have decayed since 2021")

For each insight, identify which features from this list are relevant:  
\- Price features (0-14): close, SMA, EMA, Bollinger bands, ATR  
\- Volume features (15-24): volume, OBV, CVD, buy ratio  
\- Technical indicators (25-34): RSI, MACD, stochastic, ADX  
\- Order flow (35-44): spread, order book imbalance, depth, microprice  
\- Funding/carry (45-49): funding rate, open interest, long-short ratio  
\- Sentiment (50-54): fear/greed, social sentiment, BTC dominance  
\- Regime indicators (55-59): regime label, volatility regime, trend strength

Return JSON array of insights:  
\[  
  {{  
    "type": "mechanism|parameter|feature\_behavior|decay",  
    "summary": "Clear one-sentence description",  
    "relevant\_features": \[list of feature indices\],  
    "parameters": {{"key": value}} if quantitative values mentioned,  
    "confidence": 0.0-1.0 based on source quality and specificity,  
    "valid\_months": estimated months this insight remains valid (null if evergreen)  
  }}  
\]

Return ONLY the JSON array, no other text."""

    def \_\_init\_\_(self, api\_client, model: str \= "claude-sonnet-4-20250514"):  
        self.api\_client \= api\_client  
        self.model \= model  
      
    async def extract(self, document: CrawledDocument) \-\> List\[ExtractedKnowledge\]:  
        """Extract structured knowledge from a document."""  
        \# Truncate content if too long  
        content \= document.content\[:8000\] if len(document.content) \> 8000 else document.content  
          
        prompt \= self.EXTRACTION\_PROMPT.format(  
            title=document.title,  
            content\_type=document.content\_type.value,  
            content=content  
        )  
          
        try:  
            response \= await self.api\_client.messages.create(  
                model=self.model,  
                max\_tokens=2000,  
                messages=\[{"role": "user", "content": prompt}\]  
            )  
              
            insights \= self.\_parse\_response(response.content\[0\].text)  
              
            return \[  
                ExtractedKnowledge(  
                    document\_hash=document.content\_hash,  
                    knowledge\_type=ins.get("type", "unknown"),  
                    summary=ins.get("summary", ""),  
                    relevant\_features=ins.get("relevant\_features", \[\]),  
                    parameters=ins.get("parameters", {}),  
                    confidence=ins.get("confidence", 0.5),  
                    valid\_until=self.\_compute\_expiry(ins.get("valid\_months")),  
                    source\_url=document.url  
                )  
                for ins in insights  
                if ins.get("summary")  
            \]  
        except Exception as e:  
            logger.error(f"Knowledge extraction failed: {e}")  
            return \[\]  
      
    def \_parse\_response(self, text: str) \-\> List\[dict\]:  
        """Parse JSON response from LLM."""  
        import json  
        try:  
            \# Find JSON array in response  
            start \= text.find('\[')  
            end \= text.rfind('\]') \+ 1  
            if start \>= 0 and end \> start:  
                return json.loads(text\[start:end\])  
        except json.JSONDecodeError:  
            pass  
        return \[\]  
      
    def \_compute\_expiry(self, valid\_months: Optional\[int\]) \-\> Optional\[datetime\]:  
        """Compute expiry date from validity period."""  
        if valid\_months is None:  
            return None  
        return datetime.now() \+ timedelta(days=valid\_months \* 30\)

### **Knowledge Storage and Querying**

import sqlite3  
from typing import List, Optional  
import json

class KnowledgeStore:  
    """  
    Persistent storage for extracted knowledge.  
      
    Uses SQLite for simplicity—can migrate to Neo4j for  
    graph queries if knowledge base grows large.  
    """  
      
    def \_\_init\_\_(self, db\_path: str \= "./crawler\_data/knowledge.db"):  
        self.db\_path \= db\_path  
        self.\_init\_db()  
      
    def \_init\_db(self):  
        """Initialize database schema."""  
        conn \= sqlite3.connect(self.db\_path)  
        conn.execute("""  
            CREATE TABLE IF NOT EXISTS knowledge (  
                id INTEGER PRIMARY KEY AUTOINCREMENT,  
                document\_hash TEXT,  
                knowledge\_type TEXT,  
                summary TEXT,  
                relevant\_features TEXT,  \-- JSON array  
                parameters TEXT,          \-- JSON object  
                confidence REAL,  
                valid\_until TEXT,  
                source\_url TEXT,  
                extraction\_date TEXT,  
                created\_at TEXT DEFAULT CURRENT\_TIMESTAMP  
            )  
        """)  
        conn.execute("""  
            CREATE INDEX IF NOT EXISTS idx\_knowledge\_type   
            ON knowledge(knowledge\_type)  
        """)  
        conn.execute("""  
            CREATE INDEX IF NOT EXISTS idx\_features   
            ON knowledge(relevant\_features)  
        """)  
        conn.commit()  
        conn.close()  
      
    def store(self, knowledge: ExtractedKnowledge):  
        """Store a single knowledge entry."""  
        conn \= sqlite3.connect(self.db\_path)  
        conn.execute("""  
            INSERT INTO knowledge   
            (document\_hash, knowledge\_type, summary, relevant\_features,   
             parameters, confidence, valid\_until, source\_url, extraction\_date)  
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)  
        """, (  
            knowledge.document\_hash,  
            knowledge.knowledge\_type,  
            knowledge.summary,  
            json.dumps(knowledge.relevant\_features),  
            json.dumps(knowledge.parameters),  
            knowledge.confidence,  
            knowledge.valid\_until.isoformat() if knowledge.valid\_until else None,  
            knowledge.source\_url,  
            knowledge.extraction\_date.isoformat()  
        ))  
        conn.commit()  
        conn.close()  
      
    def query\_by\_features(  
        self,   
        feature\_indices: List\[int\],  
        min\_confidence: float \= 0.5,  
        include\_expired: bool \= False  
    ) \-\> List\[ExtractedKnowledge\]:  
        """Query knowledge relevant to specific features."""  
        conn \= sqlite3.connect(self.db\_path)  
          
        \# SQLite doesn't have native JSON array search, so we filter in Python  
        cursor \= conn.execute("""  
            SELECT \* FROM knowledge   
            WHERE confidence \>= ?  
            ORDER BY confidence DESC, extraction\_date DESC  
        """, (min\_confidence,))  
          
        results \= \[\]  
        now \= datetime.now()  
          
        for row in cursor.fetchall():  
            features \= json.loads(row\[4\])  \# relevant\_features column  
              
            \# Check feature overlap  
            if any(f in features for f in feature\_indices):  
                \# Check expiry  
                valid\_until \= row\[7\]  
                if valid\_until and not include\_expired:  
                    if datetime.fromisoformat(valid\_until) \< now:  
                        continue  
                  
                results.append(ExtractedKnowledge(  
                    document\_hash=row\[1\],  
                    knowledge\_type=row\[2\],  
                    summary=row\[3\],  
                    relevant\_features=features,  
                    parameters=json.loads(row\[5\]),  
                    confidence=row\[6\],  
                    valid\_until=datetime.fromisoformat(valid\_until) if valid\_until else None,  
                    source\_url=row\[8\],  
                    extraction\_date=datetime.fromisoformat(row\[9\])  
                ))  
          
        conn.close()  
        return results  
      
    def query\_by\_type(  
        self,   
        knowledge\_type: str,  
        limit: int \= 20  
    ) \-\> List\[ExtractedKnowledge\]:  
        """Query knowledge by type (mechanism, parameter, decay, etc.)."""  
        conn \= sqlite3.connect(self.db\_path)  
        cursor \= conn.execute("""  
            SELECT \* FROM knowledge   
            WHERE knowledge\_type \= ?  
            ORDER BY confidence DESC, extraction\_date DESC  
            LIMIT ?  
        """, (knowledge\_type, limit))  
          
        results \= \[\]  
        for row in cursor.fetchall():  
            results.append(ExtractedKnowledge(  
                document\_hash=row\[1\],  
                knowledge\_type=row\[2\],  
                summary=row\[3\],  
                relevant\_features=json.loads(row\[4\]),  
                parameters=json.loads(row\[5\]),  
                confidence=row\[6\],  
                valid\_until=datetime.fromisoformat(row\[7\]) if row\[7\] else None,  
                source\_url=row\[8\],  
                extraction\_date=datetime.fromisoformat(row\[9\])  
            ))  
          
        conn.close()  
        return results  
      
    def get\_decay\_warnings(self) \-\> List\[ExtractedKnowledge\]:  
        """Get all decay observations (signals that no longer work)."""  
        return self.query\_by\_type("decay", limit=50)

### **Integration with Generation Engines**

The crawler's output conditions how generation engines operate:

class CrawlerConditioner:  
    """  
    Apply crawled knowledge to condition strategy generation.  
      
    Three conditioning modes:  
    1\. Feature weighting: Upweight features with recent positive research  
    2\. Decay filtering: Penalize strategies using decayed signals  
    3\. Parameter seeding: Use extracted parameters as generation hints  
    """  
      
    def \_\_init\_\_(self, knowledge\_store: KnowledgeStore):  
        self.store \= knowledge\_store  
        self.\_feature\_weights \= np.ones(60)  
        self.\_decayed\_features \= set()  
        self.\_last\_update \= None  
      
    def update\_conditioning(self):  
        """Refresh conditioning from knowledge store."""  
        \# 1\. Update feature weights based on recent mechanism research  
        for i in range(60):  
            knowledge \= self.store.query\_by\_features(\[i\], min\_confidence=0.6)  
            if knowledge:  
                \# More high-confidence research \= higher weight  
                weight \= 1.0 \+ 0.1 \* len(knowledge)  
                self.\_feature\_weights\[i\] \= min(2.0, weight)  
          
        \# 2\. Identify decayed features  
        decay\_warnings \= self.store.get\_decay\_warnings()  
        self.\_decayed\_features \= set()  
        for warning in decay\_warnings:  
            if warning.confidence \> 0.7:  
                self.\_decayed\_features.update(warning.relevant\_features)  
          
        self.\_last\_update \= datetime.now()  
      
    def get\_feature\_weights(self) \-\> np.ndarray:  
        """Get current feature weights for strategy generation."""  
        return self.\_feature\_weights.copy()  
      
    def is\_feature\_decayed(self, feature\_index: int) \-\> bool:  
        """Check if a feature's predictive power has decayed."""  
        return feature\_index in self.\_decayed\_features  
      
    def get\_parameter\_hints(self, strategy\_type: str) \-\> Dict\[str, float\]:  
        """Get parameter hints for specific strategy types."""  
        hints \= {}  
          
        \# Query parameter-type knowledge  
        params \= self.store.query\_by\_type("parameter", limit=30)  
          
        for p in params:  
            if strategy\_type.lower() in p.summary.lower():  
                hints.update(p.parameters)  
          
        return hints  
      
    def filter\_decayed\_strategies(  
        self,   
        strategies: List\[StrategyGenome\]  
    ) \-\> List\[StrategyGenome\]:  
        """Filter out strategies relying heavily on decayed features."""  
        filtered \= \[\]  
          
        for strategy in strategies:  
            \# Check which features the strategy uses  
            used\_features \= self.\_extract\_used\_features(strategy)  
            decayed\_count \= sum(1 for f in used\_features if f in self.\_decayed\_features)  
              
            \# Reject if \>50% of features are decayed  
            if decayed\_count / max(len(used\_features), 1\) \<= 0.5:  
                filtered.append(strategy)  
          
        return filtered  
      
    def \_extract\_used\_features(self, strategy: StrategyGenome) \-\> List\[int\]:  
        """Extract which feature indices a strategy references."""  
        \# Would parse strategy tree to find feature references  
        \# Placeholder implementation  
        return list(range(10))  \# Simplified

### **Orchestration: Putting It Together**

class WebCrawlerOrchestrator:  
    """  
    Main orchestrator for the lightweight web crawler.  
      
    Runs as a background process, typically once every 12-24 hours.  
    """  
      
    def \_\_init\_\_(  
        self,  
        api\_client,  \# For LLM extraction  
        storage\_path: str \= "./crawler\_data",  
        sources: List\[SourceConfig\] \= None  
    ):  
        self.crawler \= LightweightWebCrawler(  
            sources=sources or DEFAULT\_SOURCES,  
            storage\_path=storage\_path  
        )  
        self.scorer \= RelevanceScorer()  
        self.extractor \= KnowledgeExtractor(api\_client)  
        self.store \= KnowledgeStore(db\_path=f"{storage\_path}/knowledge.db")  
        self.conditioner \= CrawlerConditioner(self.store)  
      
    async def run\_cycle(self) \-\> Dict:  
        """  
        Run one complete crawl-extract-store cycle.  
          
        Returns summary statistics.  
        """  
        stats \= {  
            "documents\_crawled": 0,  
            "documents\_relevant": 0,  
            "knowledge\_extracted": 0,  
            "sources\_crawled": \[\]  
        }  
          
        \# Step 1: Crawl all due sources  
        logger.info("Starting crawl cycle...")  
        documents \= await self.crawler.crawl\_all\_sources()  
        stats\["documents\_crawled"\] \= len(documents)  
          
        \# Step 2: Score and filter for relevance  
        relevant\_docs \= self.scorer.filter\_relevant(documents, min\_score=0.3)  
        stats\["documents\_relevant"\] \= len(relevant\_docs)  
        logger.info(f"Found {len(relevant\_docs)} relevant documents")  
          
        \# Step 3: Extract knowledge from relevant documents  
        for doc in relevant\_docs:  
            knowledge\_items \= await self.extractor.extract(doc)  
              
            for item in knowledge\_items:  
                self.store.store(item)  
                stats\["knowledge\_extracted"\] \+= 1  
              
            \# Rate limit LLM calls  
            await asyncio.sleep(1)  
          
        \# Step 4: Update generation conditioning  
        self.conditioner.update\_conditioning()  
          
        logger.info(f"Cycle complete. Extracted {stats\['knowledge\_extracted'\]} knowledge items")  
        return stats  
      
    def get\_conditioner(self) \-\> CrawlerConditioner:  
        """Get the conditioner for integration with generation engines."""  
        return self.conditioner

\# Usage in Layer 1 Explorer  
async def main():  
    import anthropic  
      
    client \= anthropic.AsyncAnthropic()  
      
    orchestrator \= WebCrawlerOrchestrator(  
        api\_client=client,  
        storage\_path="./crawler\_data"  
    )  
      
    \# Run one cycle  
    stats \= await orchestrator.run\_cycle()  
    print(f"Crawl stats: {stats}")  
      
    \# Get conditioner for generation  
    conditioner \= orchestrator.get\_conditioner()  
      
    \# Example: Get feature weights for evolutionary search  
    weights \= conditioner.get\_feature\_weights()  
    print(f"Feature weights: {weights}")  
      
    \# Example: Check for decayed signals  
    for i in range(60):  
        if conditioner.is\_feature\_decayed(i):  
            print(f"Warning: Feature {i} has decayed predictive power")

if \_\_name\_\_ \== "\_\_main\_\_":  
    asyncio.run(main())

---

## **Configuration**

\# config/web\_crawler.yaml

crawler:  
  storage\_path: "./crawler\_data"  
  user\_agent: "HIMARI-Research-Bot/1.0 (academic research)"  
    
  \# Global rate limiting  
  min\_delay\_between\_requests\_seconds: 2.0  
  max\_concurrent\_requests: 1  \# Be respectful  
    
  \# Scheduling  
  default\_crawl\_frequency\_hours: 24  
  exchange\_docs\_frequency\_hours: 12  \# More frequent for fee changes

scoring:  
  min\_relevance\_score: 0.3  
  mechanism\_keyword\_weight: 1.0  
  noise\_keyword\_penalty: 0.2  
  recency\_bonus\_days: 30

extraction:  
  llm\_model: "claude-sonnet-4-20250514"  
  max\_content\_length: 8000  
  max\_insights\_per\_document: 10  
  min\_confidence\_to\_store: 0.4

conditioning:  
  feature\_weight\_boost\_per\_paper: 0.1  
  max\_feature\_weight: 2.0  
  decay\_confidence\_threshold: 0.7  
  decay\_feature\_rejection\_threshold: 0.5  \# Reject if \>50% features decayed

\# Budget tracking  
budget:  
  max\_llm\_calls\_per\_day: 50  
  estimated\_cost\_per\_call\_usd: 0.01  
  max\_monthly\_cost\_usd: 15

---

## **Cost Analysis**

| Component | Cost Driver | Monthly Estimate |
| ----- | ----- | ----- |
| Crawling | Compute only (no APIs) | \~$0 |
| Storage | SQLite on disk | \~$0 |
| LLM Extraction | \~50 docs/day × $0.01 | \~$15 |
| **Total** |  | **\~$15/month** |

This fits well within the Layer 1 budget of $50–100/month, leaving room for the generation engines.

---

## **Integration Points**

The crawler integrates with the rest of Layer 1 at two points:

**1\. Generation Engine Conditioning**

\# In EngineOrchestrator  
conditioner \= crawler\_orchestrator.get\_conditioner()

\# Apply feature weights to evolutionary fitness  
def modified\_fitness(strategy, base\_fitness):  
    weights \= conditioner.get\_feature\_weights()  
    used\_features \= extract\_used\_features(strategy)  
    weight\_bonus \= np.mean(\[weights\[f\] for f in used\_features\])  
    return base\_fitness \* weight\_bonus

\# Filter decayed strategies before validation  
candidates \= conditioner.filter\_decayed\_strategies(candidates)

**2\. HIFA Validation Enhancement**

\# In HIFAPipeline, add decay check before full backtest  
def \_stage2\_5\_decay\_check(self, strategy: StrategyGenome) \-\> HIFAResult:  
    """Reject strategies relying on decayed signals."""  
    used \= self.\_extract\_used\_features(strategy)  
    decayed \= \[f for f in used if self.conditioner.is\_feature\_decayed(f)\]  
      
    if len(decayed) / len(used) \> 0.5:  
        return HIFAResult(  
            passed=False,  
            score=0,  
            metrics={"decayed\_features": decayed},  
            reason=f"Strategy relies on decayed features: {decayed}",  
            latency\_ms=1  
        )  
      
    return HIFAResult(passed=True, score=1, metrics={}, reason="", latency\_ms=1)

This gives Layer 1 a lightweight but valuable web research capability—informing generation without bloating the real-time decision path.

