#!/usr/bin/env bash
# Le os hiperparametros de <abordagem>/run_g5k_hydra.sh sem executar nada.
#
# Existe para que o job do SDumont use exatamente a mesma receita da bateria do
# Grid'5000/GPPD -- se os flags fossem recopiados aqui, bastaria um ajuste num
# lado para os numeros dos dois clusters deixarem de ser comparaveis, e o desvio
# so apareceria depois de gastar as GPU-horas.
#
# Dois detalhes fazem funcionar:
#   - g5k_run_common.sh vira no-op: ele espera o state.env do OAR, que nao
#     existe aqui. Ja o common_recipe.sh precisa carregar de verdade (define os
#     HCPA_* que pytorch_opt e tensorflow_opt consomem).
#   - argv[0] tem de ser ./run_g5k_hydra.sh, por isso o `bash -c ... "$0"`: o
#     launcher deriva TOOLS_DIR de "$0" e, com outro nome, monta um caminho
#     inexistente e aborta antes de definir qualquer coisa.
#
# Uso: bash _recipe.sh <project_dir> <abordagem>   -> atribuicoes bash no stdout
set -uo pipefail

PROJECT_DIR="${1:?project_dir}"
APPROACH="${2:?abordagem}"

cd "${PROJECT_DIR}/${APPROACH}" || exit 90

# O eco da receita vai para stderr, onde vira registro no log do job; o stdout
# carrega apenas as atribuicoes que o chamador vai avaliar.
exec bash -c '
  source() { case "$1" in */g5k_run_common.sh) return 0 ;; *) builtin source "$@" ;; esac; }
  g5k_run_all() { :; }
  builtin source "$0" 1>&2 || true
  set +u
  [ "${#TRAIN_STATIC_ARGS[@]}" -gt 0 ] || exit 91
  printf "ENTRY=%q\n"                 "${ENTRY}"
  printf "RESULTS_FLAG=%q\n"          "${RESULTS_FLAG}"
  printf "EXEC_FLAG=%q\n"             "${EXEC_FLAG}"
  printf "SEED_FLAG=%q\n"             "${SEED_FLAG:-}"
  printf "TRAIN_ACCEPTS_DATASET=%q\n" "${TRAIN_ACCEPTS_DATASET:-1}"
  printf "TRAIN_STATIC_ARGS=("; printf "%q " "${TRAIN_STATIC_ARGS[@]}"; printf ")\n"
  for v in KERAS_BACKEND TF_GPU_ALLOCATOR TF_FORCE_GPU_ALLOW_GROWTH TF_ENABLE_GPU_GC \
           TF_NUM_INTEROP_THREADS TF_CPP_MIN_LOG_LEVEL DALI_LOG; do
    [ -n "${!v:-}" ] && printf "export %s=%q\n" "$v" "${!v}"
  done
  exit 0
' ./run_g5k_hydra.sh
