#!/bin/bash
# re-render the portrait frames 686-1089 in fresh Blender processes (a long session went black after ~690 frames)
cd "$(dirname "$0")/.."
for a in 686 786 886 986; do
  python3 kettle/render_kettle.py kettle/frames_p_in kettle/frames_p_png 432 768 12 $a $((a+100)) >> kettle/render_p2.log 2>&1
done
cd kettle && python3 make_portrait_movie.py ../media/kettle_portrait.mp4 frames_p_in frames_p_png run1 ../boiler/interface/frames20p > portrait_movie.log 2>&1
echo finished >> portrait_movie.log
