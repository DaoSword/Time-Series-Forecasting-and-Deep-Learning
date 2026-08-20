import unittest

from scripts.update_paper_stats import (
    STATS_END,
    STATS_START,
    count_papers,
    update_statistics,
)


SAMPLE_README = f"""# Example

## Papers

{STATS_START}
stale
{STATS_END}

### 2024

- [Paper A](https://example.com/a)

  - [[Official Code](https://example.com/code-a)]

* Paper B

### 2023

- [Paper C](https://example.com/c)

## Blogs

- [Not a paper](https://example.com/blog)
"""


class PaperStatisticsTests(unittest.TestCase):
    def test_count_papers_by_year(self) -> None:
        self.assertEqual(count_papers(SAMPLE_README), {"2024": 2, "2023": 1})

    def test_update_statistics_is_idempotent(self) -> None:
        updated = update_statistics(SAMPLE_README)

        self.assertIn("**Paper count:** **3**", updated)
        self.assertIn("| 2024 | 2 |", updated)
        self.assertIn("| 2023 | 1 |", updated)
        self.assertEqual(update_statistics(updated), updated)

    def test_rejects_paper_entry_before_year_heading(self) -> None:
        malformed = SAMPLE_README.replace(
            f"{STATS_END}\n\n### 2024",
            f"{STATS_END}\n\n- Paper without a year\n\n### 2024",
        )

        with self.assertRaisesRegex(ValueError, "before the first year"):
            count_papers(malformed)

    def test_requires_statistics_markers(self) -> None:
        without_markers = SAMPLE_README.replace(STATS_START, "").replace(
            STATS_END, ""
        )

        with self.assertRaisesRegex(ValueError, "exactly one"):
            update_statistics(without_markers)


if __name__ == "__main__":
    unittest.main()
