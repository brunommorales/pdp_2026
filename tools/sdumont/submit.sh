#!/usr/bin/env bash
# Gera e submete o plano experimental do HCPA no SDumont, por fases.
#
# Por padrao NAO submete nada: imprime a matriz, o walltime estimado de cada job
# e o total em GPU-hora/UA. Confira, depois rode de novo com SUBMIT=1.
#
#   ./tools/sdumont/submit.sh P0            # smoke (2 epocas, 1 run)  -- FACA ISTO PRIMEIRO
#   ./tools/sdumont/submit.sh P1            # 1 GPU, HCPA, 10 runs
#   ./tools/sdumont/submit.sh P2            # 1 GPU, misto, 10 runs
#   ./tools/sdumont/submit.sh P3            # escalabilidade 2 e 4 GPU, 3 runs
#   ./tools/sdumont/submit.sh P4            # escalabilidade 6 e 8 GPU, misto, 3 runs
#   ./tools/sdumont/submit.sh all           # matriz completa (so relatorio)
#   SUBMIT=1 ARCH=gh200 ./tools/sdumont/submit.sh P1
#
# As filas de H100 e GH200 sao DESCOBERTAS SOZINHAS pelo sinfo (nomes nao estao
# no wiki do LNCC). So exporte SD_PART_H100/SD_PART_GH200 se quiser forcar outra.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SUBMIT="${SUBMIT:-0}"
ARCH_FILTER="${ARCH:-both}"          # gh200 | h100 | both
APPROACH_FILTER="${APPROACHES:-}"    # lista separada por espaco; vazio = todas
SAFETY="${SAFETY:-1.4}"              # margem sobre o tempo medido no GH200/Lyon
GPUS_PER_NODE_MAX="${GPUS_PER_NODE_MAX:-4}"   # os nos H100 e GH200 tem 4 GPUs
MAX_JOBS="${MAX_JOBS:-100}"          # teto de jobs por PROJETO no SDumont
# Runs nos pontos multi-GPU (P3/P4). 10 = mesma amostragem dos pontos de 1 GPU.
# Baixar para 3 corta ~70% do custo dessas fases e ainda da intervalo apertado
# para desempenho e energia, mas nao para a variancia de AUC.
SCALE_RUNS="${SCALE_RUNS:-10}"

# Filas. Vazias = descobertas sozinhas pelo sinfo (nomes das filas de H100 e
# GH200 nao estao no wiki do LNCC, so na maquina). Exporte para forcar.
SD_PART_H100="${SD_PART_H100:-}"
SD_PART_GH200="${SD_PART_GH200:-}"
SD_PART_DEV_H100="${SD_PART_DEV_H100:-}"    # filas *_dev: prioridade alta p/ o smoke
SD_PART_DEV_GH200="${SD_PART_DEV_GH200:-}"

# shellcheck source=_partitions.sh
source "${PROJECT_DIR}/tools/sdumont/_partitions.sh"
if [[ -z "${SD_PART_H100}" || -z "${SD_PART_GH200}" ]]; then
  sd_detect_partitions || true
  echo "[filas] GH200 -> ${SD_PART_GH200:-<nao encontrada>} (${SD_IDLE_GH200:-0} nos livres)" >&2
  echo "[filas] H100  -> ${SD_PART_H100:-<nao encontrada>} (${SD_IDLE_H100:-0} nos livres)" >&2
  echo >&2
fi

APPROACHES_ALL=(pytorch_base pytorch_opt tensorflow_base tensorflow_opt
                hybrid_simple hybrid_token_reduction hybrid_token_reduction_opt
                retfound_green)

# Horas por run, 200 epocas, 1 GPU, dataset HCPA. Medido: 10 runs por abordagem
# no GH200 (new_results/nvidia-gh200-480gb_g5k_hydra/gh200_all_runs.csv,
# coluna total_time_s). NAO sao chutes.
h_hcpa() { case "$1" in
  pytorch_opt)                 echo 0.20 ;;
  tensorflow_opt)              echo 0.27 ;;
  pytorch_base)                echo 0.41 ;;
  hybrid_token_reduction_opt)  echo 0.41 ;;
  tensorflow_base)             echo 0.54 ;;
  retfound_green)              echo 0.54 ;;
  hybrid_simple)               echo 0.57 ;;
  hybrid_token_reduction)      echo 0.57 ;;
  *) echo 0.60 ;; esac; }

# ddr_eyepacs tem 59.956 imagens de treino contra 9.350 do HCPA = 6,41x.
# val e test escalam no mesmo fator, entao o custo por run escala junto.
DATA_FACTOR_all_tfrec_v2=1.00
DATA_FACTOR_ddr_eyepacs=6.41

# Eficiencia de escalabilidade assumida (wall = h1 / (N * eff)). Conservador:
# 1 e 2 GPU sao intra-no com NVLink; 6 e 8 atravessam a rede entre nos.
eff() { case "$1" in 1) echo 1.00 ;; 2) echo 0.90 ;; 4) echo 0.80 ;; 6) echo 0.70 ;; 8) echo 0.65 ;; *) echo 0.60 ;; esac; }

part_for() { case "$1" in gh200) echo "${SD_PART_GH200}" ;; h100) echo "${SD_PART_H100}" ;; esac; }
dev_for()  { case "$1" in gh200) echo "${SD_PART_DEV_GH200:-}" ;; h100) echo "${SD_PART_DEV_H100:-}" ;; esac; }
gpn_for()  {
  local v
  case "$1" in gh200) v="${SD_GPN_GH200:-}" ;; h100) v="${SD_GPN_H100:-}" ;; esac
  [[ "${v}" =~ ^[1-9][0-9]*$ ]] && echo "${v}" || echo "${GPUS_PER_NODE_MAX}"
}

# ------------------------------------------------------------------ plano ----
# fase|arch|abordagem|dataset|ngpus|run_start|run_end|epocas
emit_plan() {
  local phase="$1"
  local archs=() ; case "${ARCH_FILTER}" in both) archs=(gh200 h100) ;; *) archs=("${ARCH_FILTER}") ;; esac
  local aps=() ; if [[ -n "${APPROACH_FILTER}" ]]; then read -ra aps <<<"${APPROACH_FILTER}"; else aps=("${APPROACHES_ALL[@]}"); fi

  for arch in "${archs[@]}"; do for ap in "${aps[@]}"; do
    case "${phase}" in
      P0)  # smoke: prova que container, DDP, dataset e telemetria funcionam
        for ds in all-tfrec-v2 ddr_eyepacs; do echo "P0|${arch}|${ap}|${ds}|1|0|0|2"; done
        echo "P0|${arch}|${ap}|all-tfrec-v2|2|0|0|2"   # valida o caminho DDP
        ;;
      P1)  echo "P1|${arch}|${ap}|all-tfrec-v2|1|0|9|200" ;;
      P2)  # misto em 1 GPU e o bloco mais caro; parte em 2 jobs de 5 runs
           # porque o backfill do SLURM favorece walltime menor
        echo "P2|${arch}|${ap}|ddr_eyepacs|1|0|4|200"
        echo "P2|${arch}|${ap}|ddr_eyepacs|1|5|9|200" ;;
      P3)  for n in 2 4; do for ds in all-tfrec-v2 ddr_eyepacs; do
             echo "P3|${arch}|${ap}|${ds}|${n}|0|$((SCALE_RUNS-1))|200"; done; done ;;
      P4)  # 6 e 8 GPU so no misto: com 9.350 imagens e batch 96/GPU, 8 GPUs dao
           # 12 passos por epoca -- mediria o launcher, nao o modelo
        for n in 6 8; do echo "P4|${arch}|${ap}|ddr_eyepacs|${n}|0|$((SCALE_RUNS-1))|200"; done ;;
    esac
  done; done
}

data_factor() { local v; eval "v=\${DATA_FACTOR_${1//-/_}:-1.00}"; echo "${v}"; }

# PACK muda o custo por run: N runs passam a rodar de uma vez, ao preco de ~15%
# de contencao. Entra na conta de quantos runs cabem no walltime da fila.
pack_factor() {
  local ph="$1" n="$2"
  if [[ -n "${PACK:-}" ]] && (( n == 1 )) && [[ "${ph}" == P1 || "${ph}" == P2 ]]; then
    awk -v k="${PACK}" 'BEGIN{printf "%.4f", 1.15/k}'
  else
    echo 1
  fi
}

# Horas de walltime que UM run desta celula consome.
hours_per_run() {
  local ph="$1" ap="$2" ds="$3" n="$4" ep="$5"
  awk -v h="$(h_hcpa "${ap}")" -v f="$(data_factor "${ds}")" -v n="${n}" \
      -v e="$(eff "${n}")" -v ep="${ep}" -v s="${SAFETY}" -v k="$(pack_factor "${ph}" "${n}")" \
      'BEGIN{printf "%.5f", h*f*(ep/200.0)/(n*e)*s*k}'
}

TOTAL_GPUH=0
NJOBS=0

# ------------------------------------------------------------- relatorio -----
emit_row() {
  local ph="$1" arch="$2" ap="$3" ds="$4" n="$5" rs="$6" re="$7" ep="$8" maxwall="$9"
  local nruns=$(( re - rs + 1 ))
  local hrun; hrun="$(hours_per_run "${ph}" "${ap}" "${ds}" "${n}" "${ep}")"
  local wall; wall=$(awk -v hr="${hrun}" -v r="${nruns}" 'BEGIN{printf "%.2f", hr*r}')

  local maxgpn; maxgpn="$(gpn_for "${arch}")"
  local nnodes=$(( (n + maxgpn - 1) / maxgpn ))
  local gpn=$(( n / nnodes ))
  # divisao desigual (ex.: 6 GPUs em nos de 4) vira reparticao simetrica: 3+3.
  # O SLURM aceitaria 4+2, mas isso desbalanceia a medida de potencia por no.
  (( gpn * nnodes == n )) || { nnodes=2; gpn=$(( n / 2 )); }

  local pack_env=""
  if [[ -n "${PACK:-}" ]] && (( n == 1 )) && [[ "${ph}" == P1 || "${ph}" == P2 ]]; then
    gpn="${PACK}"; nnodes=1; pack_env=",PACK=${PACK}"
  fi
  local gpuh; gpuh=$(awk -v w="${wall}" -v g="$(( gpn * nnodes ))" 'BEGIN{printf "%.1f", w*g}')

  local partition; partition="$(part_for "${arch}")"
  # O smoke ganha com a prioridade da fila *_dev, mas ela tem walltime de 20 min.
  # Vale para as celulas do HCPA (2 epocas em 9.350 imagens); as do misto sao
  # 6,4x maiores e vao para a fila normal, para nao morrerem por TIME LIMIT no
  # meio do staging de 5,1 GB.
  # SD_USE_DEV=0 desliga esse desvio. As *_dev tem prioridade alta, mas se a
  # associacao do usuario nao permitir nos nelas (REASON AssocMaxNodePerJobLimit)
  # os 32 smokes do HCPA ficariam pendentes para sempre, o que e pior do que
  # esperar na fila normal.
  if [[ "${SD_USE_DEV:-1}" == "1" && "${ph}" == "P0" && "${ds}" == "all-tfrec-v2" ]]; then
    local devp; devp="$(dev_for "${arch}")"
    [[ -n "${devp}" ]] && partition="${devp}"
  fi

  TOTAL_GPUH=$(awk -v a="${TOTAL_GPUH}" -v b="${gpuh}" 'BEGIN{printf "%.1f", a+b}')
  NJOBS=$(( NJOBS + 1 ))
  printf '%-4s %-6s %-27s %-14s %4d %7s %6s %7sh %8s  %s\n' \
         "${ph}" "${arch}" "${ap}" "${ds}" "${n}" "${nnodes}x${gpn}" \
         "${rs}-${re}" "${wall}" "${gpuh}" "${partition:-<SD_PART_?>}"

  [[ "${SUBMIT}" == "1" ]] || return 0
  [[ -n "${partition}" ]] || { echo "  !! sem fila para ${arch}; exporte SD_PART_${arch^^}" >&2; exit 3; }

  local tlimit; tlimit=$(awk -v w="${wall}" -v m="${maxwall}" \
    'BEGIN{t=int(w+1); if(t>m)t=m; if(t<1)t=1; printf "%02d:00:00", t}')
  # -A obrigatorio: sem ele o SLURM usa a associacao default `none`, que tem
  # MaxSubmit=0, e recusa a submissao com AssocMaxSubmitJobLimit.
  local acct; acct="$(sd_account_for "${partition}")"
  sbatch --parsable \
    -p "${partition}" ${acct:+-A "${acct}"} \
    -J "hcpa_${ph}_${ap}_${ds}_${n}g_r${rs}" \
    --nodes="${nnodes}" --ntasks-per-node=1 \
    --gpus-per-node="${gpn}" --cpus-per-gpu="${CPUS_PER_GPU:-12}" \
    --time="${tlimit}" \
    --chdir="${PROJECT_DIR}" \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",APPROACH="${ap}",DATASET="${ds}",NGPUS="${n}",RUN_START="${rs}",RUN_END="${re}",EPOCHS="${ep}",BATCH_SIZE="${BATCH_SIZE:-}${pack_env}" \
    "${PROJECT_DIR}/tools/sdumont/run.srm"
}

run() {
  local phases=("$@")
  printf '%-4s %-6s %-27s %-14s %4s %7s %6s %8s %8s  %s\n' \
         FASE ARCH ABORDAGEM DATASET GPUS NOSxGPU RUNS WALL GPU-H FILA
  printf '%s\n' "-------------------------------------------------------------------------------------------------------"
  local phase ph arch ap ds n rs re ep
  for phase in "${phases[@]}"; do
    while IFS='|' read -r ph arch ap ds n rs re ep; do
      [[ -n "${ph:-}" ]] || continue

      # Teto de walltime da fila. As ict-* do SDumont sao de 24h, nao das 96h do
      # wiki: um job estimado acima disso e recusado com PartitionTimeLimit e
      # fica PENDING para sempre. A faixa de runs e picotada para caber.
      local part0; part0="$(part_for "${arch}")"
      local maxwall; maxwall="$(sd_maxwall_h "${part0:-x}")"
      local hrun; hrun="$(hours_per_run "${ph}" "${ap}" "${ds}" "${n}" "${ep}")"
      local maxruns; maxruns=$(awk -v m="${maxwall}" -v hr="${hrun}" \
                       'BEGIN{r=int(m*0.85/hr); print (r<1?1:r)}')
      local total=$(( re - rs + 1 ))

      if (( maxruns < total )); then
        local a=${rs} b
        while (( a <= re )); do
          b=$(( a + maxruns - 1 )); (( b > re )) && b=${re}
          emit_row "${ph}" "${arch}" "${ap}" "${ds}" "${n}" "${a}" "${b}" "${ep}" "${maxwall}"
          a=$(( b + 1 ))
        done
      else
        emit_row "${ph}" "${arch}" "${ap}" "${ds}" "${n}" "${rs}" "${re}" "${ep}" "${maxwall}"
      fi
    done < <(emit_plan "${phase}")
  done

  printf '%s\n' "-------------------------------------------------------------------------------------------------------"
  awk -v j="${NJOBS}" -v g="${TOTAL_GPUH}" \
      'BEGIN{printf "jobs=%d   total=%.0f GPU-h   ~%.0f UA (100 UA por GPU-hora)\n", j, g, g*100}'
  (( NJOBS <= MAX_JOBS )) || echo "AVISO: ${NJOBS} jobs excede o teto de ${MAX_JOBS} por projeto. Submeta em fases."
  [[ "${SUBMIT}" == "1" ]] || echo "(relatorio apenas -- rode com SUBMIT=1 para submeter)"
}

case "${1:-all}" in
  all) run P0 P1 P2 P3 P4 ;;
  P0|P1|P2|P3|P4) run "$1" ;;
  *) echo "uso: $0 {P0|P1|P2|P3|P4|all}" >&2; exit 2 ;;
esac
