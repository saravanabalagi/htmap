#!/bin/sh
for i in "2014-12-09-13-21-02" "2014-12-10-18-10-50" "2015-05-19-14-06-38"; do
  python process_htmap_loops_inliers.py --results_dir ../output/$i/results
  python process_htmap_loops_inliers.py --results_dir ../output/$i/results_backup
done
