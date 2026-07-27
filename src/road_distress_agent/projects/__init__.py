"""Inspection-project ledger: cross-session aggregation unit for the delivery layer.

A single conversation (``thread_id``) tracks one defect. An inspection project is
the higher-level container that aggregates many confirmed defects (one road
segment, many defects) so the delivery layer can produce segment-level outputs.
"""
