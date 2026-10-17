#!/usr/bin/env bash
# nbv_policy ablation: every arm differs from the full method by exactly one switch.
#
#   bash sim/ablation/run_nbv_ablation.sh [OUT_DIR] [NUM_STEPS] [JOBS]
#
# Scene: the phase2 object on a ground plane (the table analogue -- ground
# gaussians are what pull a Fisher-greedy camera toward the hemisphere rim).
# --point_scale 1.0 is required: instance_gs.ply is already in metres.
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH="$PWD"
PY="${PY:-$HOME/anaconda3/envs/InfoFlow/bin/python}"   # needs modified_diff_gaussian_rasterization
OUT="${1:-sim/sim_outputs/nbv_ablation}"
STEPS="${2:-300}"
JOBS="${3:-2}"
mkdir -p "$OUT"

COMMON=(--pcd_path sim/sim_outputs/phase2/instance_gs.ply --point_scale 1.0
        --config config/sim_rtabmap_config.yaml --headless --cartesian
        --num_steps "$STEPS" --dt 0.1 --fisher_step_scale 0.03 --spherical_speed_min 0.0
        --linear_vel_max 0.05
        --fisher_num_samples 16 --fisher_num_dense_points 64
        --quality_every 10 --quality_views 16 --log_min_level INFO --no-log_file)
GROUND=(--ground --ground_z -0.076 --ground_size 2.0)

# Switch sets.
TANGENT=(--nbv_set gradient_space=tangent)
ANGLE=(--nbv_set gradient_space=angle)
FEAS=(--nbv_set feasibility.enabled=true)
MASK=(--nbv_set fisher_mask.enabled=true --nbv_set fisher_mask.roi_bounds=object
      --nbv_set fisher_mask.roi_padding_m=0.02 --nbv_set fisher_mask.support_plane_z=object_min_z)
RATE=(--nbv_set velocity_mapping.mode=normalized --nbv_set velocity_mapping.v_nominal_mps=0.05)
STOP=(--nbv_set stop.enabled=true)

declare -A ARMS
ARMS[legacy]="${ANGLE[*]}"
ARMS[tangent_only]="${TANGENT[*]}"
ARMS[full]="${TANGENT[*]} ${FEAS[*]} ${MASK[*]} ${RATE[*]} ${STOP[*]}"
ARMS[full_no_tangent]="${ANGLE[*]} ${FEAS[*]} ${MASK[*]} ${RATE[*]} ${STOP[*]}"
ARMS[full_no_feas]="${TANGENT[*]} ${MASK[*]} ${RATE[*]} ${STOP[*]}"
ARMS[full_no_mask]="${TANGENT[*]} ${FEAS[*]} ${RATE[*]} ${STOP[*]}"
ARMS[full_no_rate]="${TANGENT[*]} ${FEAS[*]} ${MASK[*]} ${STOP[*]}"
ARMS[full_no_stop]="${TANGENT[*]} ${FEAS[*]} ${MASK[*]} ${RATE[*]}"

# Start poses (theta, phi): low, mid, high elevation.
STARTS=("s20:0.0:0.35" "s45:2.1:0.785" "s70:4.2:1.22")

jobs_file="$OUT/jobs.txt"; : > "$jobs_file"
for start in "${STARTS[@]}"; do
  IFS=: read -r sname th ph <<< "$start"
  for arm in legacy tangent_only full full_no_tangent full_no_feas full_no_mask full_no_rate full_no_stop; do
    echo "ground ${arm} ${sname} ${th} ${ph} ${ARMS[$arm]}" >> "$jobs_file"
  done
  # Control without a ground plane: original vs full.
  for arm in legacy full; do
    echo "noground ${arm} ${sname} ${th} ${ph} ${ARMS[$arm]}" >> "$jobs_file"
  done
done

run_one() {
  local scene=$1 arm=$2 sname=$3 th=$4 ph=$5; shift 5
  local dir="$OUT/${scene}/${arm}/${sname}"
  if [ -f "$dir/summary.json" ]; then echo "skip $dir"; return; fi
  mkdir -p "$dir"
  local extra=()
  [ "$scene" = ground ] && extra=("${GROUND[@]}")
  # shellcheck disable=SC2068
  "$PY" sim/sim_fisher_closed_loop.py "${COMMON[@]}" "${extra[@]}" \
      --save_dir "$dir" --init_theta "$th" --init_phi "$ph" $@ > "$dir/stdout.log" 2>&1
  echo "done exit=$? $dir"
}
export -f run_one
export OUT PY
export COMMON_STR="${COMMON[*]}" GROUND_STR="${GROUND[*]}"

# xargs cannot see bash arrays; re-expand from strings inside the worker.
worker() {
  read -r -a COMMON <<< "$COMMON_STR"; read -r -a GROUND <<< "$GROUND_STR"
  run_one "$@"
}
export -f worker
xargs -P "$JOBS" -L 1 bash -c 'worker "$@"' _ < "$jobs_file"
echo ABLATION_DONE
