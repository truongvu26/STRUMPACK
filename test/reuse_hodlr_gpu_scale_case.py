#!/usr/bin/env python3
"""Reuse a completed scaling case only for an identical executable and command."""
import json
from pathlib import Path
import shlex
import shutil
import sys
source,destination=map(Path,sys.argv[1:3])
name=sys.argv[3]
expected=sys.argv[4:]
summary=json.loads((source/'scale_summary.json').read_text())
case=next((case for case in summary['cases'] if case['label']==name),None)
if not case or case['status']!='PASS': raise SystemExit('Source case has not passed numerical validation')
if (source/'executable.sha256').read_text().split()[0]!=(destination/'executable.sha256').read_text().split()[0]:
    raise SystemExit('Cannot reuse a case from a different executable')
command=shlex.split((source/(name+'.command')).read_text())
if expected[0] not in command or command[command.index(expected[0]):]!=expected:
    raise SystemExit('Cannot reuse a case with different solver arguments')
for pattern in [name+'.rank*.log',name+'.time',name+'.command',name+'.exit_code']:
    for path in source.glob(pattern): shutil.copy2(str(path),str(destination/path.name))
record_path=destination/'reused_cases.json'
record=json.loads(record_path.read_text()) if record_path.exists() else {}
record[name]=dict(origin_job=source.name,origin_artifacts=str(source),
                  source_commit=summary['source_commit'],reason='Resume after analysis-only failure; solver step completed with exit zero')
record_path.write_text(json.dumps(record,indent=2)+'\n')
print('Reused '+name+' from allocation '+source.name)
