from copy import deepcopy
from docx import Document
from docx.shared import Inches
from pathlib import Path

src = Path('AI_Video_Summarization_IEEE_Revised_v9.docx')
out = Path('AI_Video_Summarization_IEEE_Revised_v10.docx')
doc = Document(src)

by_text = {p.text.strip(): p for p in doc.paragraphs if p.text.strip()}
lead = by_text['Results include the 107-s software smoke test and the 909.8-s long-video experiments on the hardware in Section IV-A. Table V reports runtime and optimization measurements; Table VI reports the proposed selector and three same-source baselines. The selector comparison covers one source video and reports interval allocation only. It is not a corpus-wide benchmark and does not measure human-judged summary quality.']
lead.text = ('Results include the 107-s software smoke test, a 909.8-s Hindi long-video experiment, and a separate 610.52-s Hindi source run. '
             'Table V reports the measured optimization profile, Table VI reports fixed-budget selector outputs, and Table VII summarizes the three supplied job records. '
             'A same-source historical run comparison shows lower elapsed time in the newer run, but its saved configuration is incomplete and human-judged quality remains unmeasured.')

# Add a concise run-record comparison immediately before limitations.
reference = by_text['VI.  LIMITATIONS']
heading_template = by_text['E. Selector Baseline Outputs']
body_template = next(p for p in doc.paragraphs if p.text.startswith('All four outputs used the same 909.8-s'))
caption_template = by_text['TABLE VI. SAME-SOURCE SELECTOR BASELINE OUTPUTS']

heading = doc.add_paragraph(style=heading_template.style)
if heading_template._p.pPr is not None:
    heading._p.insert(0, deepcopy(heading_template._p.pPr))
r = heading.add_run('F. Additional Job-Record Profiles')
if heading_template.runs and heading_template.runs[0]._r.rPr is not None:
    r._r.insert(0, deepcopy(heading_template.runs[0]._r.rPr))

body = doc.add_paragraph(style=body_template.style)
if body_template._p.pPr is not None:
    body._p.insert(0, deepcopy(body_template._p.pPr))
body.add_run(
    'The supplied job records provide a same-source runtime comparison and a second source profile. '
    'Runs d7f91028 and b2221dc7 have the same source SHA-256, 909.8-s duration, and 25% summary budget. '
    'Elapsed time decreased from 1,127.53 s to 725.30 s (35.7%); however, the older run does not retain its transcription configuration, '
    'and the heuristic importance-weighted coverage changed from 1.000 to 0.972, with the newer report listing one uncovered topic. '
    'These records therefore show a runtime difference, not a controlled quality-preserving speedup. '
    'Run 9304a2d2 processed a separate 610.52-s Hindi source in 594.88 s and selected 152.06 s under its 25% target; its reported coverage of 1.000 is also an internal proxy, not human evaluation. '
    'The three job IDs and values are retained in Table VII for traceability.'
)

caption = doc.add_paragraph(style=caption_template.style)
if caption_template._p.pPr is not None:
    caption._p.insert(0, deepcopy(caption_template._p.pPr))
caption.add_run('TABLE VII. SUPPLIED JOB-RUN PROFILE')

table = doc.add_table(rows=4, cols=4)
table.style = doc.tables[5].style
table.autofit = False
widths = [Inches(0.82), Inches(1.05), Inches(1.12), Inches(1.18)]
rows = [
    ('Run', 'Source and budget', 'Runtime; selected summary', 'Heuristic coverage'),
    ('d7f91028', '909.80 s; 25%', '1,127.53 s; 227.12 s', '1.000; no reported gap'),
    ('b2221dc7', '909.80 s; 25%', '725.30 s; 227.34 s', '0.972; one reported gap'),
    ('9304a2d2', '610.52 s; 25%', '594.88 s; 152.06 s', '1.000; no reported gap'),
]
for ri, values in enumerate(rows):
    row = table.rows[ri]
    row._tr.get_or_add_trPr()
    for ci, value in enumerate(values):
        cell = row.cells[ci]
        cell.width = widths[ci]
        cell.text = value
        for p in cell.paragraphs:
            p.paragraph_format.space_after = 0
            for run in p.runs:
                run.font.size = __import__('docx').shared.Pt(7.2)
                if ri == 0:
                    run.bold = True

for element in (heading._p, body._p, caption._p, table._tbl):
    reference._p.addprevious(element)

# Make the interpretation explicit in limitations and conclusion.
lim = next(p for p in doc.paragraphs if p.text.startswith('• Runtime measurements come from a small number of videos'))
lim.add_run(' The supplied same-source pair suggests a 35.7% lower runtime in the newer run, but the older run lacks saved ASR settings and its heuristic coverage differs; this is not evidence that summary quality was preserved.')
conclusion = next(p for p in doc.paragraphs if p.text.startswith('This paper described a local timestamped extractive video-summarization pipeline'))
conclusion.add_run(' A second same-source job record was 35.7% slower than the newer run, while a separate 610.52-s source completed in 594.88 s. Because the historical configuration is incomplete and the internal coverage proxy differs, these observations do not demonstrate a quality-preserving speedup. Human reference intervals and reviewer ratings remain necessary.')

doc.save(out)
print(out.resolve())
