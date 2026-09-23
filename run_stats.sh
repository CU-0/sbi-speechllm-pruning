#!/usr/bin/env bash
#
# run_stats.sh — build the CSV + PNG for every task in one or more result folders.
# csv files are required for plotting.
#
#   ./run_stats.sh results/qwen2audio_shortGPT
#   ./run_stats.sh results/qwen2audio_shortGPT results/voxtral_shortGPT
#   ./run_stats.sh --all                      # every folder under $RESULTS_ROOT
#   ./run_stats.sh --all -n                   # print the commands, run nothing
#
# Tasks are discovered from performance_results_<task>.jsonl, so a new task
# needs no change here.  Run from the repo root — `python -m evaluate.stats`
# has to be importable.
#
# Options:
#   -t, --tasks a,b,c   only these tasks (default: whatever is in the folder)
#   -m, --mode M        plot mode passed to performance_plot (default: pruning)
#   -f, --force         rebuild even if the CSV/PNG is newer than its inputs
#   -n, --dry-run       print commands only
#   -h, --help          this text

set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results}"
PYTHON="${PYTHON:-python}"
MODE=pruning
FORCE=0
DRY=0
declare -a WANT_TASKS=()
declare -a FOLDERS=()

die() { printf 'run_stats: %s\n' "$*" >&2; exit 2; }
usage() { sed -n '2,/^$/s/^# \{0,1\}//p' "$0"; exit 0; }

while (($#)); do
  case "$1" in
    -t|--tasks)      [[ ${2:-} ]] || die "--tasks needs a value"
                     IFS=', ' read -r -a WANT_TASKS <<<"$2"; shift 2 ;;
    -m|--mode)       [[ ${2:-} ]] || die "--mode needs a value"
                     MODE="$2"; shift 2 ;;
    -f|--force)      FORCE=1; shift ;;
    -n|--dry-run)    DRY=1; shift ;;
    -h|--help)       usage ;;
    --all)           [[ -d $RESULTS_ROOT ]] || die "RESULTS_ROOT is not a directory: $RESULTS_ROOT"
                     while IFS= read -r d; do FOLDERS+=("$d"); done \
                       < <(find "$RESULTS_ROOT" -mindepth 1 -maxdepth 1 -type d | sort)
                     shift ;;
    --)              shift; while (($#)); do FOLDERS+=("$1"); shift; done ;;
    -*)              die "unknown option: $1" ;;
    *)               FOLDERS+=("$1"); shift ;;
  esac
done

((${#FOLDERS[@]})) || die "no result folder given (try --all, or -h for help)"

ok=0; skipped=0; failed=0
declare -a FAILURES=()

run() {                       # run <description> <cmd...>
  local what="$1"; shift
  if ((DRY)); then
    printf '  would run:'; printf ' %q' "$@"; printf '\n'
    return 0
  fi
  if ! "$@"; then
    printf '  FAILED: %s\n' "$what" >&2
    FAILURES+=("$what")
    return 1
  fi
  return 0
}

# newer_than FILE REF -> true when FILE exists and is not older than REF
newer_than() { [[ -f $1 && -f $2 && ! $1 -ot $2 ]]; }

for folder in "${FOLDERS[@]}"; do
  folder="${folder%/}"
  if [[ ! -d $folder ]]; then
    printf '%s\n  not a directory, skipping\n' "$folder" >&2
    ((skipped++)); continue
  fi

  # discover tasks
  declare -a tasks=()
  if ((${#WANT_TASKS[@]})); then
    tasks=("${WANT_TASKS[@]}")
  else
    for p in "$folder"/performance_results_*.jsonl; do
      [[ -e $p ]] || continue
      p="${p##*/performance_results_}"
      tasks+=("${p%.jsonl}")
    done
  fi

  printf '\n%s\n' "$folder"
  if ((${#tasks[@]} == 0)); then
    printf '  no performance_results_*.jsonl found, skipping\n' >&2
    ((skipped++)); continue
  fi
  for task in "${tasks[@]}"; do
    perf="$folder/performance_results_${task}.jsonl"
    csv="$folder/${task}.csv"
    png="$folder/${task}.png"

    if [[ ! -f $perf ]]; then
      printf '  %-6s missing %s, skipping\n' "$task" "${perf##*/}" >&2
      ((skipped++)); continue
    fi

    printf '  %-6s -> %s, %s\n' "$task" "${csv##*/}" "${png##*/}"

    # ---- table
    if ((FORCE)) || ! newer_than "$csv" "$perf"; then
      run "$folder:$task table" \
        "$PYTHON" -m evaluate.stats performance_table "$perf" -o "$csv" \
        || { ((failed++)); continue; }
    else
      printf '    csv up to date\n'
    fi

    # ---- plot
    if ((DRY)) || [[ -f $csv ]]; then
      if ((FORCE)) || ! newer_than "$png" "$csv"; then
        run "$folder:$task plot" \
          "$PYTHON" -m evaluate.stats performance_plot "$csv" -m "$MODE" -o "$png" \
          || { ((failed++)); continue; }
      else
        printf '    png up to date\n'
      fi
    else
      printf '    no csv produced, skipping plot\n' >&2
      ((failed++)); continue
    fi

    ((ok++))
  done
done

printf '\n%d done, %d skipped, %d failed\n' "$ok" "$skipped" "$failed"
if ((failed)); then
  printf 'failures:\n'; printf '  %s\n' "${FAILURES[@]}"
  exit 1
fi