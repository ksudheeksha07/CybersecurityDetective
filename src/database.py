import sqlite3
from pathlib import Path


DATABASE_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "cybersecurity.db"
)


def get_connection():
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row

    return connection


def initialize_database():
    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS investigations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            source_ip TEXT,
            destination_ip TEXT,
            source_port INTEGER,
            destination_port INTEGER,
            protocol TEXT,
            prediction TEXT,
            ml_probability REAL,
            anomaly_score REAL,
            risk_score REAL,
            threat_level TEXT,
            evidence TEXT
        )
    """)

    connection.commit()
    connection.close()


def save_investigation(
    timestamp,
    source_ip,
    destination_ip,
    source_port,
    destination_port,
    protocol,
    prediction,
    ml_probability,
    anomaly_score,
    risk_score,
    threat_level,
    evidence
):
    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO investigations (
            timestamp,
            source_ip,
            destination_ip,
            source_port,
            destination_port,
            protocol,
            prediction,
            ml_probability,
            anomaly_score,
            risk_score,
            threat_level,
            evidence
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        timestamp,
        source_ip,
        destination_ip,
        source_port,
        destination_port,
        protocol,
        prediction,
        ml_probability,
        anomaly_score,
        risk_score,
        threat_level,
        evidence
    ))

    connection.commit()

    investigation_id = cursor.lastrowid

    connection.close()

    return investigation_id


def get_recent_investigations(limit=20):
    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        SELECT *
        FROM investigations
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))

    results = cursor.fetchall()

    connection.close()

    return results


if __name__ == "__main__":
    initialize_database()

    print("Database initialized successfully.")
    print(f"Database path: {DATABASE_PATH}")