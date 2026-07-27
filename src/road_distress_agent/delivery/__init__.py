"""Project-level delivery layer: supervisor + report/cost/work-order specialists.

This subgraph runs on top of the (untouched) diagnosis graph. It aggregates the
confirmed defects of one inspection project and fans out to specialist agents
that produce real deliverables (docx report, xlsx cost sheet, work orders with
dry-run email/calendar actions), then an independent compliance critic gates the
package before it is handed back to the user.
"""
