"""
Knowledge Store

SQLite-based persistence for extracted trading knowledge.
Supports querying by features, category, and recency.
"""

import logging
import json
import sqlite3
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path

from .extractor import ExtractedKnowledge
from .scorer import ContentCategory

logger = logging.getLogger(__name__)


@dataclass
class StoredKnowledge:
    """Knowledge record from storage."""
    id: int
    source_url: str
    source_title: str
    category: str
    mechanism: str
    causal_hypothesis: str
    signals_json: str
    market_regime: str
    timeframe: str
    expected_sharpe: float
    expected_drawdown: float
    extraction_confidence: float
    features_used: List[str]
    created_at: datetime
    last_accessed: datetime
    access_count: int

    def to_extracted(self) -> ExtractedKnowledge:
        """Convert back to ExtractedKnowledge."""
        from .extractor import ExtractedSignal, SignalDirection

        signals_data = json.loads(self.signals_json)
        signals = [
            ExtractedSignal(
                name=s['name'],
                description=s.get('description', ''),
                direction=SignalDirection(s.get('direction', 'both')),
                features_used=s.get('features_used', []),
                threshold_type=s.get('threshold_type', 'above'),
                suggested_threshold=s.get('suggested_threshold')
            )
            for s in signals_data
        ]

        return ExtractedKnowledge(
            source_url=self.source_url,
            source_title=self.source_title,
            category=ContentCategory(self.category),
            mechanism=self.mechanism,
            causal_hypothesis=self.causal_hypothesis,
            signals=signals,
            entry_conditions=[],
            exit_conditions=[],
            risk_params=None,
            market_regime=self.market_regime,
            timeframe=self.timeframe,
            expected_performance={
                'sharpe': self.expected_sharpe,
                'max_drawdown': self.expected_drawdown
            },
            extraction_confidence=self.extraction_confidence,
            raw_text="",
            extracted_at=self.created_at
        )


class KnowledgeStore:
    """
    SQLite-based store for extracted trading knowledge.

    Features:
    - Persistent storage with schema migration
    - Query by features, category, regime
    - Decay-based filtering (prefer recent knowledge)
    - Deduplication by URL
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS knowledge (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_url TEXT UNIQUE,
        source_title TEXT,
        category TEXT,
        mechanism TEXT,
        causal_hypothesis TEXT,
        signals_json TEXT,
        market_regime TEXT,
        timeframe TEXT,
        expected_sharpe REAL,
        expected_drawdown REAL,
        extraction_confidence REAL,
        features_used TEXT,
        created_at TIMESTAMP,
        last_accessed TIMESTAMP,
        access_count INTEGER DEFAULT 0
    );

    CREATE INDEX IF NOT EXISTS idx_category ON knowledge(category);
    CREATE INDEX IF NOT EXISTS idx_regime ON knowledge(market_regime);
    CREATE INDEX IF NOT EXISTS idx_created ON knowledge(created_at);
    CREATE INDEX IF NOT EXISTS idx_confidence ON knowledge(extraction_confidence);
    """

    def __init__(
        self,
        db_path: str = "data/knowledge.db",
        auto_create: bool = True
    ):
        """
        Initialize knowledge store.

        Args:
            db_path: Path to SQLite database
            auto_create: Create database/tables if not exists
        """
        self.db_path = Path(db_path)

        if auto_create:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_schema()

    def _init_schema(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(self.SCHEMA)
            conn.commit()

    def store(self, knowledge: ExtractedKnowledge) -> int:
        """
        Store extracted knowledge.

        Args:
            knowledge: Knowledge to store

        Returns:
            ID of stored record
        """
        signals_json = json.dumps([
            {
                'name': s.name,
                'description': s.description,
                'direction': s.direction.value,
                'features_used': s.features_used,
                'threshold_type': s.threshold_type,
                'suggested_threshold': s.suggested_threshold
            }
            for s in knowledge.signals
        ])

        features_used = list(set(
            f for s in knowledge.signals for f in s.features_used
        ))

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO knowledge (
                        source_url, source_title, category, mechanism,
                        causal_hypothesis, signals_json, market_regime,
                        timeframe, expected_sharpe, expected_drawdown,
                        extraction_confidence, features_used,
                        created_at, last_accessed, access_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    knowledge.source_url,
                    knowledge.source_title,
                    knowledge.category.value,
                    knowledge.mechanism,
                    knowledge.causal_hypothesis,
                    signals_json,
                    knowledge.market_regime,
                    knowledge.timeframe,
                    knowledge.expected_performance.get('sharpe', 0),
                    knowledge.expected_performance.get('max_drawdown', 0.2),
                    knowledge.extraction_confidence,
                    json.dumps(features_used),
                    knowledge.extracted_at,
                    datetime.now(),
                    0
                ))
                conn.commit()
                return cursor.lastrowid

            except Exception as e:
                logger.error(f"Failed to store knowledge: {e}")
                return -1

    def store_batch(self, knowledge_list: List[ExtractedKnowledge]) -> int:
        """Store multiple knowledge items."""
        stored = 0
        for k in knowledge_list:
            if self.store(k) > 0:
                stored += 1
        return stored

    def query_by_features(
        self,
        features: List[str],
        min_confidence: float = 0.5,
        limit: int = 20
    ) -> List[StoredKnowledge]:
        """
        Query knowledge relevant to specific features.

        Args:
            features: List of feature names
            min_confidence: Minimum extraction confidence
            limit: Maximum results

        Returns:
            List of relevant knowledge
        """
        results = []

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Query all above confidence threshold
            cursor.execute("""
                SELECT * FROM knowledge
                WHERE extraction_confidence >= ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (min_confidence, limit * 3))

            rows = cursor.fetchall()

            for row in rows:
                features_used = json.loads(row[12])

                # Check feature overlap
                overlap = len(set(features) & set(features_used))
                if overlap > 0:
                    results.append(self._row_to_stored(row))

                    # Update access tracking
                    cursor.execute("""
                        UPDATE knowledge
                        SET last_accessed = ?, access_count = access_count + 1
                        WHERE id = ?
                    """, (datetime.now(), row[0]))

            conn.commit()

        return results[:limit]

    def query_by_category(
        self,
        category: ContentCategory,
        min_confidence: float = 0.5,
        limit: int = 20
    ) -> List[StoredKnowledge]:
        """Query knowledge by category."""
        results = []

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM knowledge
                WHERE category = ? AND extraction_confidence >= ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (category.value, min_confidence, limit))

            for row in cursor.fetchall():
                results.append(self._row_to_stored(row))

        return results

    def query_by_regime(
        self,
        regime: str,
        min_confidence: float = 0.5,
        limit: int = 20
    ) -> List[StoredKnowledge]:
        """Query knowledge by market regime."""
        results = []

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM knowledge
                WHERE (market_regime = ? OR market_regime = 'any')
                AND extraction_confidence >= ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (regime, min_confidence, limit))

            for row in cursor.fetchall():
                results.append(self._row_to_stored(row))

        return results

    def query_recent(
        self,
        days: int = 30,
        min_confidence: float = 0.5,
        limit: int = 50
    ) -> List[StoredKnowledge]:
        """Query recent knowledge."""
        cutoff = datetime.now() - timedelta(days=days)
        results = []

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM knowledge
                WHERE created_at >= ? AND extraction_confidence >= ?
                ORDER BY extraction_confidence DESC, created_at DESC
                LIMIT ?
            """, (cutoff, min_confidence, limit))

            for row in cursor.fetchall():
                results.append(self._row_to_stored(row))

        return results

    def get_all(
        self,
        min_confidence: float = 0.3,
        limit: int = 100
    ) -> List[StoredKnowledge]:
        """Get all stored knowledge."""
        results = []

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM knowledge
                WHERE extraction_confidence >= ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (min_confidence, limit))

            for row in cursor.fetchall():
                results.append(self._row_to_stored(row))

        return results

    def _row_to_stored(self, row: tuple) -> StoredKnowledge:
        """Convert database row to StoredKnowledge."""
        return StoredKnowledge(
            id=row[0],
            source_url=row[1],
            source_title=row[2],
            category=row[3],
            mechanism=row[4],
            causal_hypothesis=row[5],
            signals_json=row[6],
            market_regime=row[7],
            timeframe=row[8],
            expected_sharpe=row[9],
            expected_drawdown=row[10],
            extraction_confidence=row[11],
            features_used=json.loads(row[12]),
            created_at=datetime.fromisoformat(row[13]) if row[13] else datetime.now(),
            last_accessed=datetime.fromisoformat(row[14]) if row[14] else datetime.now(),
            access_count=row[15]
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get store statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Total count
            cursor.execute("SELECT COUNT(*) FROM knowledge")
            total = cursor.fetchone()[0]

            # By category
            cursor.execute("""
                SELECT category, COUNT(*)
                FROM knowledge
                GROUP BY category
            """)
            by_category = dict(cursor.fetchall())

            # By regime
            cursor.execute("""
                SELECT market_regime, COUNT(*)
                FROM knowledge
                GROUP BY market_regime
            """)
            by_regime = dict(cursor.fetchall())

            # Average confidence
            cursor.execute("""
                SELECT AVG(extraction_confidence)
                FROM knowledge
            """)
            avg_confidence = cursor.fetchone()[0] or 0

        return {
            'total_records': total,
            'by_category': by_category,
            'by_regime': by_regime,
            'average_confidence': avg_confidence
        }

    def cleanup_old(self, days: int = 180) -> int:
        """
        Remove knowledge older than specified days.

        Args:
            days: Age threshold in days

        Returns:
            Number of records removed
        """
        cutoff = datetime.now() - timedelta(days=days)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM knowledge
                WHERE created_at < ?
            """, (cutoff,))
            conn.commit()
            return cursor.rowcount

    def check_exists(self, url: str) -> bool:
        """Check if knowledge from URL already exists."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 1 FROM knowledge WHERE source_url = ?
            """, (url,))
            return cursor.fetchone() is not None
