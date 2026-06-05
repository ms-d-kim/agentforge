"""Generate a tiny synthetic BIRD-like fixture for offline pipeline tests.

Creates ``data/fixture/dev.json`` + ``data/fixture/dev_databases/school/school.sqlite``
in exactly the layout the real loader (src.tasks) expects, so the whole pipeline
can be exercised with zero downloads and zero API calls. The five questions span
easy/medium/hard so the workflow arms can be made to differ in the dry run.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from src.config import FIXTURE_DIR

DB_ID = "school"

SCHOOLS = [
    (1, "Alpha High", "Alpha"),
    (2, "Beta High", "Beta"),
    (3, "Gamma High", "Alpha"),
]
# Alpha High (1) has the most students; some grade-12 students in each school.
STUDENTS = [
    (1, "Ann", 12, 1),
    (2, "Bob", 11, 1),
    (3, "Cara", 12, 1),
    (4, "Dan", 10, 1),
    (5, "Eve", 12, 2),
    (6, "Finn", 11, 2),
    (7, "Gail", 12, 3),
]

QUESTIONS = [
    {
        "question_id": 1,
        "db_id": DB_ID,
        "question": "How many students are enrolled in total?",
        "evidence": "",
        "SQL": "SELECT COUNT(*) FROM students",
        "difficulty": "simple",
    },
    {
        "question_id": 2,
        "db_id": DB_ID,
        "question": "What are the names of schools located in Alpha county?",
        "evidence": "Alpha county refers to county = 'Alpha'.",
        "SQL": "SELECT name FROM schools WHERE county = 'Alpha'",
        "difficulty": "simple",
    },
    {
        "question_id": 3,
        "db_id": DB_ID,
        "question": "What is the name of the school with the most students?",
        "evidence": "",
        "SQL": (
            "SELECT s.name FROM schools AS s "
            "JOIN students AS st ON st.school_id = s.id "
            "GROUP BY s.id ORDER BY COUNT(*) DESC LIMIT 1"
        ),
        "difficulty": "challenging",
    },
    {
        "question_id": 4,
        "db_id": DB_ID,
        "question": "List the names of students in grade 12.",
        "evidence": "",
        "SQL": "SELECT name FROM students WHERE grade = 12",
        "difficulty": "simple",
    },
    {
        "question_id": 5,
        "db_id": DB_ID,
        "question": "How many schools have at least one student in grade 12?",
        "evidence": "",
        "SQL": "SELECT COUNT(DISTINCT school_id) FROM students WHERE grade = 12",
        "difficulty": "moderate",
    },
]


def build_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    con = sqlite3.connect(str(db_path))
    con.executescript(
        """
        CREATE TABLE schools (
            id     INTEGER PRIMARY KEY,
            name   TEXT NOT NULL,
            county TEXT
        );
        CREATE TABLE students (
            id        INTEGER PRIMARY KEY,
            name      TEXT NOT NULL,
            grade     INTEGER,
            school_id INTEGER REFERENCES schools(id)
        );
        """
    )
    con.executemany("INSERT INTO schools VALUES (?, ?, ?)", SCHOOLS)
    con.executemany("INSERT INTO students VALUES (?, ?, ?, ?)", STUDENTS)
    con.commit()
    con.close()


def ensure_fixture() -> Path:
    """Build the fixture if missing; return the fixture root."""
    db_path = FIXTURE_DIR / "dev_databases" / DB_ID / f"{DB_ID}.sqlite"
    dev_json = FIXTURE_DIR / "dev.json"
    if not db_path.exists():
        build_db(db_path)
    if not dev_json.exists():
        dev_json.write_text(json.dumps(QUESTIONS, indent=2))
    return FIXTURE_DIR


def main() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    db_path = FIXTURE_DIR / "dev_databases" / DB_ID / f"{DB_ID}.sqlite"
    build_db(db_path)
    (FIXTURE_DIR / "dev.json").write_text(json.dumps(QUESTIONS, indent=2))
    print(f"fixture written:\n  {FIXTURE_DIR/'dev.json'}\n  {db_path}")


if __name__ == "__main__":
    main()
