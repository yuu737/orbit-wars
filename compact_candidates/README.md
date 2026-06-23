# Compact Candidate Layout

This directory keeps research candidates compact.

## Idea

During research, keep only one shared `orbit_lite/` and store each candidate as a named Python file:

- `sample48_4p_sample_opening_domain_factory.py`
- `sample48_4p_sample_opening_domain_factory.params.json`
- `sample50_4p_halite_pw_adapted.py`
- `sample50_4p_halite_pw_adapted.params.json`
- `orbit_lite/`

For Kaggle submission, build a normal folder/zip with:

- `main.py`
- `params.json`
- `oracle_rules.json` if present
- `orbit_lite/`

## Sync From Folder Candidate

```powershell
C:\tmp\ow\Scripts\python.exe tools\sync_compact_candidate.py sample50_4p_halite_pw_adapted
```

If `orbit_lite` changed and should become the shared compact copy:

```powershell
C:\tmp\ow\Scripts\python.exe tools\sync_compact_candidate.py sample50_4p_halite_pw_adapted --sync-orbit
```

## Build Submission Zip

```powershell
C:\tmp\ow\Scripts\python.exe tools\pack_compact_candidate.py sample50_4p_halite_pw_adapted
```

Output:

- `submission_builds/sample50_4p_halite_pw_adapted/`
- `submission_builds/sample50_4p_halite_pw_adapted.zip`

