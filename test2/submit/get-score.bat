@echo off
if "%1"=="" (set GATE_RESULTS=airvision_submission.json) else (set GATE_RESULTS=%1)
echo mAP score for %GATE_RESULTS%
python ../scorer/score_detections.py -g training_GT_labels_v2.json -p %GATE_RESULTS%
