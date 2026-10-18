#!/usr/bin/env bash
# grad_eps sweep + Fisher contribution diagnostics (docs/eps_contrib_report.md).
#
#   bash sim/ablation/run_eps_contrib.sh [OUT_DIR] [NUM_STEPS] [REPS]
#
# Arm = full_no_stop from run_nbv_ablation.sh (tangent + feasibility + mask + rate,
# stop off so every run uses the same step budget); only --grad_eps varies.
# Every run also writes contrib_diag.jsonl (read-only scoring every 30 steps).
# Jobs are ordered rep-major so a partial batch is still balanced across eps.
# Serial on purpose: the ground scene peaks ~14 GB per process on a 24 GB card.
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH="$PWD"
PY="${PY:-$HOME/anaconda3/envs/InfoFlow/bin/python}"   # needs modified_diff_gaussian_rasterization
OUT="${1:-sim/sim_outputs/eps_contrib}"
STEPS="${2:-300}"
REPS="${3:-3}"
EPS_LIST=(${EPS_LIST:-0.01 0.05 0.1 0.2})
mkdir -p "$OUT"

COMMON=(--pcd_path sim/sim_outputs/phase2/instance_gs.ply --point_scale 1.0 --recolor texture
        --config sim/ablation/sim_ablation_config.yaml --headless --cartesian
        --num_steps "$STEPS" --dt 0.1 --fisher_step_scale 0.03 --spherical_speed_min 0.0
        --linear_vel_max 0.05
        --fisher_num_samples 16 --fisher_num_dense_points 64
        --quality_every 10 --quality_views 16 --log_min_level INFO --no-log_file
        --ground --ground_z -0.076 --ground_size 2.0
        --contrib_diag_every 30 --contrib_diag_dirs 24)
ARM=(--nbv_set gradient_space=tangent
     --nbv_set feasibility.enabled=true
     --nbv_set fisher_mask.enabled=true --nbv_set fisher_mask.roi_bounds=object
     --nbv_set fisher_mask.roi_padding_m=0.02 --nbv_set fisher_mask.support_plane_z=object_min_z
     --nbv_set velocity_mapping.mode=normalized --nbv_set velocity_mapping.v_nominal_mps=0.05)

STARTS=("s20:0.0:0.35" "s45:2.1:0.785" "s70:4.2:1.22")

for rep in $(seq 1 "$REPS"); do
  for start in "${STARTS[@]}"; do
    IFS=: read -r sname th ph <<< "$start"
    for eps in "${EPS_LIST[@]}"; do
      dir="$OUT/eps${eps}/${sname}/r${rep}"
      if [ -f "$dir/summary.json" ]; then echo "skip $dir"; continue; fi
      mkdir -p "$dir"
      t0=$(date +%s)
      "$PY" sim/sim_fisher_closed_loop.py "${COMMON[@]}" "${ARM[@]}" \
          --grad_eps "$eps" --save_dir "$dir" --init_theta "$th" --init_phi "$ph" \
          > "$dir/stdout.log" 2>&1
      echo "done exit=$? $(( $(date +%s) - t0 ))s $dir"
    done
  done
done
echo EPS_CONTRIB_DONE
