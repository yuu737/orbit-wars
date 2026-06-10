$ErrorActionPreference = "Stop"

Write-Host "Running vs V8.2..."
C:\tmp\ow\Scripts\python.exe evaluate.py --players 2 --agent bots\my_v8_4_hybrid_pytorch.py --opponent bots\my_v8_2_projection_split.py --games 10 --both-seats --workers 8 --seed-start 71000000 > eval_v8_4_vs_v8_2.txt

Write-Host "Running vs Hairate..."
C:\tmp\ow\Scripts\python.exe evaluate.py --players 2 --agent bots\my_v8_4_hybrid_pytorch.py --opponent bots\hairate.py --games 10 --both-seats --workers 8 --seed-start 71000000 > eval_v8_4_vs_hairate.txt

Write-Host "Running vs Hairate2..."
C:\tmp\ow\Scripts\python.exe evaluate.py --players 2 --agent bots\my_v8_4_hybrid_pytorch.py --opponent bots\hairate2.py --games 10 --both-seats --workers 8 --seed-start 71000000 > eval_v8_4_vs_hairate2.txt

Write-Host "All evaluations finished."
