#!/usr/bin/env bash
# PUXA o projeto HCPA e os datasets do GPPD para o SDumont.
#
# ATENCAO ao sentido: este script roda NO SDUMONT (nó de login), nao no GPPD.
# A conexao so existe SDumont -> GPPD, entao quem inicia o rsync e o SDumont.
# Os nós de login do SDumont tem internet liberada; os de execucao nao.
#
# Uso, no nó de login do SDumont:
#   ./tools/sdumont/fetch.sh              # codigo + datasets
#   ONLY=code ./tools/sdumont/fetch.sh    # so codigo (rapido, para iterar)
#   ONLY=data ./tools/sdumont/fetch.sh    # so datasets
#   DRY=1 ./tools/sdumont/fetch.sh        # simula, nao transfere
#
# Volume: ~360 MB de codigo e pesos, 1,6 GB do HCPA, 5,1 GB do misto = 7,1 GB.
set -euo pipefail

# Alias do ~/.ssh/config do SDumont, que ja carrega usuario, host e chave.
# Nao use IP nem FQDN aqui: o resolvedor do SDumont nao resolve *.inf.ufrgs.br e
# a rota direta por IP publico da timeout na porta 22. O que funciona e o alias.
GPPD="${GPPD:-gppd}"
# Caminho RELATIVO ao home remoto (o alias ja entra no home certo).
GPPD_SRC="${GPPD_SRC:-projects/hcpa}"
REMOTE="${GPPD}"

# Area de dados do SDumont. NAO use o $HOME: a cota e pequena e os nós de
# execucao leem do /scratch. Confira o seu com `echo $SCRATCH`.
DEST="${DEST:-${SCRATCH:-/scratch/${USER}}/hcpa}"

ONLY="${ONLY:-all}"
DRY="${DRY:-0}"

OPTS=(-a --info=progress2 --partial --human-readable)
[[ "${DRY}" == "1" ]] && OPTS+=(--dry-run --stats)

# Fora da transferencia: resultados antigos, venvs, caches de build, .git, os
# .sif (sao construidos aqui, por arquitetura) e o vit_pure, que nao entra nesta
# bateria. O dataset vai no bloco separado, sem -z.
CODE_EXCLUDES=(
  --exclude='data/'          --exclude='new_results/'
  --exclude='results/'       --exclude='results_*/'
  --exclude='_backup_*/'     --exclude='env_arm/'
  --exclude='env_x86/'       --exclude='.venv/'
  --exclude='.git/'          --exclude='__pycache__/'
  --exclude='*.pyc'          --exclude='logs/'
  --exclude='.cache/'        --exclude='*.sif'
  --exclude='*.tar.gz'       --exclude='*.zip'
  --exclude='vit_pure/'      --exclude='.g5k_run_scripts/'
  --exclude='*.ipynb'
)

mkdir -p "${DEST}/data"

fetch_code() {
  echo "==> codigo: ${REMOTE}:${GPPD_SRC} -> ${DEST}"
  rsync "${OPTS[@]}" -z "${CODE_EXCLUDES[@]}" "${REMOTE}:${GPPD_SRC}/" "${DEST}/"
}

# Sem -z: TFRecord de fundus ja e JPEG comprimido, comprimir so gasta CPU.
# --size-only: os shards sao imutaveis, evita rechecar 2.351 arquivos por mtime.
fetch_data() {
  echo "==> datasets: ${REMOTE}:${GPPD_SRC}/data -> ${DEST}/data"
  rsync "${OPTS[@]}" --size-only \
    "${REMOTE}:${GPPD_SRC}/data/all-tfrec-v2" \
    "${REMOTE}:${GPPD_SRC}/data/ddr_eyepacs" \
    "${DEST}/data/"
}

case "${ONLY}" in
  code) fetch_code ;;
  data) fetch_data ;;
  all)  fetch_code; fetch_data ;;
  *)    echo "ONLY invalido: ${ONLY} (use code|data|all)" >&2; exit 2 ;;
esac

if [[ "${DRY}" == "1" ]]; then
  echo; echo "==> DRY=1: nada foi transferido, conferencia pulada."
  exit 0
fi

echo
echo "==> conferencia"
for d in pytorch_base pytorch_opt tensorflow_base tensorflow_opt \
         hybrid_simple hybrid_token_reduction hybrid_token_reduction_opt retfound_green; do
  [[ -f "${DEST}/${d}/run_g5k_hydra.sh" ]] && echo "  [ok] ${d}" || echo "  [FALTA] ${d}"
done
[[ -f "${DEST}/retfound_green/weights/retfoundgreen_statedict.pth" ]] \
  && echo "  [ok] pesos do RETFound" || echo "  [FALTA] pesos do RETFound"
[[ -d "${DEST}/weights_cache/hf" ]] \
  && echo "  [ok] weights_cache (InceptionV3 pre-treinado)" \
  || echo "  [FALTA] weights_cache: deveria vir no rsync do codigo; refaca com ONLY=code"
for ds in all-tfrec-v2 ddr_eyepacs; do
  n=$(ls "${DEST}/data/${ds}" 2>/dev/null | grep -c '\.tfrec$' || true)
  echo "  [${ds}] ${n} shards"     # esperado: 160 e 1015
done

echo
echo "==> proximo passo: cd ${DEST} && ./tools/sdumont/setup.sh probe"
