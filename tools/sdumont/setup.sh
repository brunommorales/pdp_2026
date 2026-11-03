#!/usr/bin/env bash
# Prepara o SDumont: sonda as filas/GPUs e constroi os containers Singularity.
#
# O SDumont NAO tem Docker nos nos de execucao (nenhum cluster HPC compartilhado
# tem: exige daemon root). O manual oficial usa Singularity/Apptainer, com
# `singularity pull docker://...` a partir de um NO DE LOGIN e `singularity run
# --nv` dentro do job. Este script segue exatamente isso.
#
# ARQUITETURAS. Sao duas, e um .sif nao serve para as duas:
#   H100  (XH3145-H)  -> x86_64  = mesma arquitetura dos nos de login
#   GH200 (XH3515-H)  -> aarch64 = Grace ARM, so compila NO PROPRIO NO
# Por isso os .sif vao para containers/<arch>/ e o job escolhe pelo `uname -m`.
#
# Uso:
#   ./tools/sdumont/setup.sh probe                 # so mostra filas/GPUs/limites
#   ./tools/sdumont/setup.sh build-x86             # aloca um no H100 e builda la
#   ./tools/sdumont/setup.sh build-arm             # aloca um no GH200 e builda la
#   ./tools/sdumont/setup.sh build                 # builda para a arch ATUAL (usado
#                                                  # pelos dois acima, dentro do job)
#   ./tools/sdumont/setup.sh prefetch              # so no caminho offline
#   ./tools/sdumont/setup.sh build-offline         # so no caminho offline
#   ./tools/sdumont/setup.sh status                # quais dos 4 .sif ja existem
#   ./tools/sdumont/setup.sh space                 # cota e espaco livre
#   ./tools/sdumont/setup.sh clean                 # apaga restos de pull interrompido
#
# ONDE CADA COISA RODA:
#   probe, status, space, clean, queue -> no de login (so leem/limpam)
#   build-x86 / build-arm              -> alocam um no e fazem TUDO la dentro
#
# Os nos de execucao do SDumont TEM internet (confirmado com `nettest`), entao o
# build inteiro roda neles: pull + pip. Isso evita o teto de 30 min do no de
# login e o cgroup de memoria que matava o mksquashfs com SIGKILL.
# O par prefetch/build-offline continua existindo para o caso de a rede dos nos
# ser cortada; ative com PREFETCH_FIRST=1.
#   build-arm               -> no login, mas ele proprio aloca um no GH200, o
#                              MESMO tipo de no onde os treinos vao rodar
#   build-offline           -> dentro de um no GH200 (aarch64)
#
# ATENCAO ao limite de 30 min do no de login: o manual do LNCC mata processos
# continuos ali. As imagens NGC tem ~8-10 GB, entao `build` e `prefetch` podem
# encostar nesse teto. Os dois sao reaproveitaveis: o que ja baixou fica no
# cache e a repeticao continua de onde parou. Se for morto, rode de novo.
#
# Sao QUATRO .sif no total, um por (arquitetura x framework):
#   containers/x86_64/hcpa-pt.sif    containers/x86_64/hcpa-tf.sif    -> H100
#   containers/aarch64/hcpa-pt.sif   containers/aarch64/hcpa-tf.sif   -> GH200
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ARCH="$(uname -m)"
CONT_DIR="${PROJECT_DIR}/containers/${ARCH}"
CACHE_DIR="${PROJECT_DIR}/containers/_cache"
PT_BASE="${PT_BASE:-nvcr.io/nvidia/pytorch:24.11-py3}"
TF_BASE="${TF_BASE:-nvcr.io/nvidia/tensorflow:25.02-tf2-py3}"

SING="$(command -v singularity || command -v apptainer || true)"
[[ -n "${SING}" ]] || { echo "singularity/apptainer nao encontrado no PATH" >&2; exit 1; }

# O desempacotamento do .sif e o que estoura a cota: o rootfs de uma imagem NGC
# passa de 20 GB, e ele nasce no TMPDIR. Deixar o TMPDIR dentro do projeto joga
# esse pico inteiro contra a cota do usuario. Preferir o disco local do no, que
# nao conta na cota, resolve o "disk quota exceeded" no meio do unpack.
# Para forcar: export SINGULARITY_TMPDIR=<caminho>
# Ordem de preferencia FIXA, nao por `df`: o gargalo na area compartilhada e a
# COTA do usuario, e `df` so ve o filesystem (que tem dezenas de TB livres e
# mente sobre o problema). O disco local do no nao tem cota, entao vem primeiro.
pick_tmpdir() {
  local d
  for d in "${SINGULARITY_TMPDIR:-}" "/tmp/${USER}" "/dev/shm/${USER}" "${CACHE_DIR}/tmp"; do
    [[ -n "${d}" ]] || continue
    mkdir -p "${d}" 2>/dev/null || continue
    # precisa de ~25 GB para desempacotar uma imagem NGC
    local free; free="$(df -Pm "${d}" 2>/dev/null | awk 'NR==2{print $4}')"
    [[ "${free}" =~ ^[0-9]+$ ]] && (( free > 25000 )) && { echo "${d}"; return; }
  done
  echo "${CACHE_DIR}/tmp"
}

export SINGULARITY_TMPDIR="$(pick_tmpdir)"
export APPTAINER_TMPDIR="${SINGULARITY_TMPDIR}"
# Onde ficam os blobs baixados. DENTRO de um job (SLURM_JOB_ID definido) vao para
# o disco local do no: sao ~9 GB por imagem e, na area compartilhada, competem
# com a cota -- foi o que estourou com "disk quota exceeded" no pull do TF, logo
# depois do PT ter ocupado o seu espaco.
# FORA do job (caminho antigo, no login) ficam no compartilhado, porque ali o
# valor deles e sobreviver a uma execucao morta pelo teto de 30 min.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  export SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-${SINGULARITY_TMPDIR}/sing-cache}"
else
  export SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-${CACHE_DIR}/sing}"
fi
export APPTAINER_CACHEDIR="${SINGULARITY_CACHEDIR}"
mkdir -p "${SINGULARITY_CACHEDIR}" "${SINGULARITY_TMPDIR}" "${CONT_DIR}"
echo "[tmp] unpack em ${SINGULARITY_TMPDIR} ($(df -Ph "${SINGULARITY_TMPDIR}" 2>/dev/null | awk 'NR==2{print $4}') livres)" >&2

# ---------------------------------------------------------------- probe ------
probe() {
  echo "### no atual: $(hostname)   arch=${ARCH}"
  echo
  # shellcheck source=_partitions.sh
  source "${PROJECT_DIR}/tools/sdumont/_partitions.sh"
  sd_print_partitions || true
  echo
  echo "### limites por fila"
  sacctmgr list user "${USER}" -s format=partition%25,qos%20,maxjobs,maxsubmit,maxnodes,maxwall -P 2>/dev/null \
    || echo "(sacctmgr indisponivel)"
  echo
  echo "### projeto e programa de alocacao"
  sacctmgr list account "$(id -gn)" format=account,descr -P 2>/dev/null || true
  # O LNCC distribui software por `module`. Se o sitio ja mantiver um
  # singularity/apptainer mais novo, ou ate um container de PyTorch/TF pronto,
  # da para economizar o build inteiro. Vale olhar antes de baixar 10 GB.
  echo "### modulos relacionados (module avail)"
  if command -v module >/dev/null 2>&1 || [[ -n "${MODULESHOME:-}" ]]; then
    module avail 2>&1 | grep -iE "singularity|apptainer|pytorch|tensorflow|python|cuda|nccl" | head -20 \
      || echo "  (nenhum modulo obviamente relacionado)"
  else
    echo "  (comando 'module' indisponivel neste no)"
  fi
  echo
  echo "### containers ja prontos no sitio"
  local d found=0
  for d in /scratch/app /petrobr/app_sequana /petrobr/parceirosbr/app; do
    [[ -d "${d}" ]] || continue
    local hits; hits="$(find "${d}" -maxdepth 3 -name '*.sif' 2>/dev/null | head -5)"
    if [[ -n "${hits}" ]]; then echo "${hits}" | sed 's/^/  /'; found=1; fi
  done
  (( found )) || echo "  (nenhum .sif encontrado em /scratch/app nem /petrobr/app_sequana)"
  echo
  echo ">>> o submit.sh detecta as filas sozinho. So exporte SD_PART_* se"
  echo ">>> quiser forcar outra."
}

# ------------------------------------------------------- requirements uniao --
# As 8 abordagens usam apenas 2 imagens base (7 PyTorch + 2 TensorFlow) e os
# requirements-ngc.txt delas so diferem em torchvision/timm. Consolidar em UM
# .sif por framework alem de economizar ~30 min de build por arch, deixa as 7
# abordagens PyTorch com bibliotecas identicas -- que e exatamente a premissa do
# "pipeline nivelado" (comparar abordagem, nao versao de biblioteca).
write_reqs() {
  cat "${PROJECT_DIR}"/{pytorch_base,pytorch_opt,hybrid_simple,hybrid_token_reduction,hybrid_token_reduction_opt,retfound_green}/requirements-ngc.txt 2>/dev/null \
    | sed 's/[[:space:]]*$//' | grep -vE '^\s*(#|$)' | sort -u > "${CONT_DIR}/reqs-pt.txt"
  grep -qx 'torchvision' "${CONT_DIR}/reqs-pt.txt" || echo torchvision >> "${CONT_DIR}/reqs-pt.txt"
  cat "${PROJECT_DIR}/tensorflow_opt/requirements" \
    | sed 's/[[:space:]]*$//' | grep -vE '^\s*(#|$)' | sort -u > "${CONT_DIR}/reqs-tf.txt"
  echo "reqs PT: $(wc -l < "${CONT_DIR}/reqs-pt.txt") pacotes / reqs TF: $(wc -l < "${CONT_DIR}/reqs-tf.txt")"
}

# ----------------------------------------------------------------- espaco ----
# O pull do NGC precisa de ~3x o tamanho da imagem em PICO: os blobs no cache, o
# rootfs desempacotado no tmp e o .sif final. A imagem PyTorch passa de 20 GB
# desempacotada, entao um `disk quota exceeded` no meio do unpack e o sintoma
# normal de area apertada -- e nao de bug no pull.
space() {
  echo "### cota"
  quota -s 2>/dev/null || true
  command -v lfs >/dev/null 2>&1 && lfs quota -h -u "${USER}" "${PROJECT_DIR}" 2>/dev/null || true
  echo
  # Só `df` nas áreas: `du` no diretório do projeto varreria os ~7 GB de dataset
  # (e no GPPD, 74 GB), o que leva minutos e não responde a pergunta.
  printf '%-44s %10s\n' "AREA" "LIVRE"
  local d
  for d in "${PROJECT_DIR}" "${SINGULARITY_TMPDIR}" "${SCRATCH:-}" "${HOME}" /tmp /scratch/parceirosbr; do
    [[ -n "${d}" && -d "${d}" ]] || continue
    printf '%-44s %10s\n' "${d}" "$(df -Ph "${d}" 2>/dev/null | awk 'NR==2{print $4}')"
  done
  echo
  echo "### o que os containers ja ocupam"
  if [[ -d "${PROJECT_DIR}/containers" ]]; then
    du -sh "${PROJECT_DIR}/containers"/* 2>/dev/null | sort -h | tail -8
  else
    echo "  (containers/ ainda nao existe)"
  fi
  echo
  echo "### estimativa do que falta baixar"
  echo "  4 imagens .sif finais           ~36 GB  (9 GB cada)"
  echo "  + site-*/ com as dependencias   ~4 GB"
  echo "  pico transitorio por imagem     ~20 GB  (fica no TMPDIR, fora da cota"
  echo "                                           se o TMPDIR for disco local)"
}

# Limpa o que sobra de pull interrompido. O Ctrl+C deixa build-temp-*/rootfs
# meio desempacotado, e o proprio singularity nao consegue remover ("directory
# not empty"), entao isso fica comendo cota calado ate alguem apagar na mao.
clean() {
  echo "### o que existe hoje em containers/_cache"
  [[ -d "${CACHE_DIR}" ]] && du -sh "${CACHE_DIR}"/* 2>/dev/null | sort -h || echo "  (vazio)"
  echo

  # Duas gerações de lixo moram aqui. O cache de blobs e os build-temp ficavam
  # DENTRO do projeto quando o TMPDIR apontava para cá, e continuam ocupando cota
  # depois que o TMPDIR passou para o disco local. Já os diretórios por
  # arquitetura são o prefetch (imagens base + wheels), que custaram download e
  # não devem ir junto.
  echo "### removendo cache de blobs e restos de pull interrompido"
  local d n=0
  for d in "${CACHE_DIR}/sing" "${CACHE_DIR}/tmp" "${SINGULARITY_CACHEDIR}" \
           "${CACHE_DIR}"/build-temp-* "${SINGULARITY_TMPDIR}"/build-temp-* \
           "${CACHE_DIR}"/bundle-temp-* "${SINGULARITY_TMPDIR}"/bundle-temp-*; do
    [[ -e "${d}" ]] || continue
    echo "  rm -rf ${d} ($(du -sh "${d}" 2>/dev/null | cut -f1))"
    chmod -R u+w "${d}" 2>/dev/null || true
    rm -rf "${d}" && n=$(( n + 1 ))
  done
  "${SING}" cache clean --force >/dev/null 2>&1 || true
  (( n )) || echo "  (nada a remover)"

  echo
  echo "### preservado (prefetch: imagens base e wheels)"
  local kept=0
  for d in "${CACHE_DIR}"/x86_64 "${CACHE_DIR}"/aarch64 "${CACHE_DIR}"/arm64; do
    [[ -d "${d}" ]] || continue
    echo "  ${d} ($(du -sh "${d}" 2>/dev/null | cut -f1))"; kept=1
  done
  (( kept )) || echo "  (nenhum prefetch ainda)"
  [[ "${CLEAN_PREFETCH:-0}" == "1" ]] && {
    echo "  CLEAN_PREFETCH=1: removendo tambem"
    rm -rf "${CACHE_DIR}"/x86_64 "${CACHE_DIR}"/aarch64 "${CACHE_DIR}"/arm64
  }

  mkdir -p "${SINGULARITY_CACHEDIR}" "${SINGULARITY_TMPDIR}"
  echo
  space
}

# ---------------------------------------------------- imagem base + site -----
# O SDumont NAO tem subuid/subgid mapeados para o usuario, entao `singularity
# build --fakeroot` falha com "no valid mapping entry found" e o build sem
# fakeroot tambem: o %post de um .def simplesmente nao roda aqui.
#
# O que FUNCIONA sem privilegio nenhum e `singularity pull`, que so baixa e
# converte a imagem. Entao a receita e:
#   1) pull da imagem base do NGC, intacta          -> hcpa-<t>.sif
#   2) pip install --target num diretorio do HOST   -> site-<t>/
#   3) o job monta site-<t>/ e poe no PYTHONPATH
# Mesmo efeito do %post, zero privilegio. O pip roda DENTRO do container de
# proposito: garante a versao de Python e a ABI certas para as wheels.

# Symlink que o %post fazia: o NGC entrega so libcuda.so.1, e sem libcuda.so o
# TorchInductor cai em eager silenciosamente e o torch.compile do pytorch_opt
# vira placebo. Como nao da para escrever na imagem, o symlink mora no host e
# entra via LD_LIBRARY_PATH. O alvo e um caminho de DENTRO do container.
make_ldextra() {
  local d="${CONT_DIR}/ldextra"
  mkdir -p "${d}"
  [[ -e "${d}/libcuda.so" ]] || ln -s /usr/local/cuda/compat/lib/libcuda.so.1 "${d}/libcuda.so"
  echo "${d}"
}

# O Singularity 4.x tem um bug na barra de progresso que estoura ao finalizar o
# pull ("panic: index out of range" em progress_roundtripper.go). O download
# funciona: quebra o display. `--quiet` nao cria a barra e evita o panic.
# uso: sing_pull <destino.sif> <uri> [flags extras do pull]
# O empacotamento em squashfs e o passo mais pesado do pull, e a config do sitio
# tem `mksquashfs procs = 0` (usa TODOS os cores). Num no de login com dezenas
# de cores isso vira dezenas de threads de compressao e um pico de memoria que
# o cgroup do usuario mata com SIGKILL ("create command failed: signal: killed").
# Limitar a afinidade limita o numero de threads, e com isso a memoria.
sing_pull() {
  local dest="$1" uri="$2"; shift 2
  local pre=()
  # Sem valor padrao: limitar cores so ajuda se a causa for MEMORIA. Se a causa
  # for o teto de 30 min, menos threads deixa mais lento e piora.
  local ncpu="${PULL_CPUS:-}"
  if [[ -n "${ncpu}" ]] && command -v taskset >/dev/null 2>&1 && [[ "${ncpu}" =~ ^[0-9]+$ ]] && (( ncpu > 0 )); then
    pre=(taskset -c "0-$(( ncpu - 1 ))")
    echo "  (mksquashfs limitado a ${ncpu} cores; PULL_CPUS ajusta)" >&2
  fi
  "${pre[@]}" "${SING}" --quiet pull --force "$@" "${dest}" "${uri}" && return 0
  echo "  pull com --quiet falhou; tentando --silent" >&2
  "${pre[@]}" "${SING}" --silent pull --force "$@" "${dest}" "${uri}"
}

pull_base() {
  local tag="$1" ref="$2" sif="${CONT_DIR}/hcpa-$1.sif"
  if [[ -f "${sif}" && "${FORCE:-0}" != "1" ]]; then
    echo "[${ARCH}/${tag}] imagem ja existe (FORCE=1 para refazer)"; return 0
  fi
  echo "[${ARCH}/${tag}] pull ${ref}"
  local t0=${SECONDS}
  sing_pull "${sif}" "docker://${ref}" || {
    echo "  [${tag}] morreu depois de $(( (SECONDS-t0)/60 ))min$(( (SECONDS-t0)%60 ))s" >&2
    echo "  Perto de 30 min = limite do no de login (nao adianta reduzir CPUs;" >&2
    echo "  repita, que o cache de blobs faz a 2a tentativa so refazer o squashfs)." >&2
    echo "  Bem antes disso = memoria; tente PULL_CPUS=2." >&2
    return 1
  }
  echo "  [${tag}] ok em $(( (SECONDS-t0)/60 ))min$(( (SECONDS-t0)%60 ))s"
  # Os blobs desta imagem ja viraram .sif. Dentro de um job nao ha o que retomar,
  # entao guardar so aumenta o pico do proximo pull. KEEP_CACHE=1 preserva.
  if [[ -n "${SLURM_JOB_ID:-}" && "${KEEP_CACHE:-0}" != "1" ]]; then
    "${SING}" cache clean --force >/dev/null 2>&1 || true
  fi
}

# Instala os requirements num diretorio do host, com o python do container.
# Flags extras ("--no-index --find-links DIR") servem ao caminho offline.
install_site() {
  local tag="$1"; shift
  local sif="${CONT_DIR}/hcpa-${tag}.sif"
  local site="${CONT_DIR}/site-${tag}"
  local reqs="${CONT_DIR}/reqs-${tag}.txt"
  if [[ -d "${site}" && "${FORCE:-0}" != "1" ]]; then
    echo "[${ARCH}/${tag}] site ja existe (FORCE=1 para refazer)"; return 0
  fi
  mkdir -p "${site}"
  echo "[${ARCH}/${tag}] pip install --target ${site}"
  "${SING}" exec --bind "${CONT_DIR}:${CONT_DIR}" "${sif}" \
    python -m pip install --no-cache-dir --target "${site}" -r "${reqs}" "$@" \
    || { echo "[${ARCH}/${tag}] pip falhou" >&2; return 1; }
  "${SING}" exec --bind "${CONT_DIR}:${CONT_DIR}" "${sif}" python -c "import pynvml" 2>/dev/null \
    || "${SING}" exec --bind "${CONT_DIR}:${CONT_DIR}" "${sif}" \
         python -m pip install --no-cache-dir --target "${site}" nvidia-ml-py "$@" || true
}

verify() {
  local tag="$1" imp="$2"
  local sif="${CONT_DIR}/hcpa-${tag}.sif" site="${CONT_DIR}/site-${tag}"
  "${SING}" exec --bind "${site}:/opt/site" --env PYTHONPATH=/opt/site \
    --bind "$(make_ldextra):/opt/ldextra" --env LD_LIBRARY_PATH=/opt/ldextra \
    "${sif}" python -c "${imp}" || echo "[${ARCH}/${tag}] VERIFICACAO FALHOU"
}

# ---------------------------------------------------------------- status -----
# Sao QUATRO containers, nao dois: GH200 e aarch64 e H100 e x86_64, e cada
# arquitetura precisa do seu PyTorch e do seu TensorFlow.
status() {
  echo
  echo "### containers (precisa dos 4 antes de submeter as duas arquiteturas)"
  local a t p sitedir n=0
  for a in x86_64 aarch64; do
    for t in pt tf; do
      p="${PROJECT_DIR}/containers/${a}/hcpa-${t}.sif"
      sitedir="${PROJECT_DIR}/containers/${a}/site-${t}"
      if [[ -f "${p}" && -d "${sitedir}" ]]; then
        printf '  [ok]      %-8s %-3s  sif %-6s site %s\n' "${a}" "${t}" \
               "$(du -h "${p}" | cut -f1)" "$(du -sh "${sitedir}" 2>/dev/null | cut -f1)"
        n=$(( n + 1 ))
      elif [[ -f "${p}" ]]; then
        printf '  [PARCIAL] %-8s %-3s  sif ok, FALTA site-%s/\n' "${a}" "${t}" "${t}"
      else
        printf '  [FALTA]   %-8s %-3s  %s\n' "${a}" "${t}" "${p}"
      fi
    done
  done
  echo "  ${n}/4 prontos"
  (( n == 4 )) || cat <<'MSG'

  x86_64  -> ./tools/sdumont/setup.sh build-x86   (aloca 1 no H100 e faz tudo la)
  aarch64 -> ./tools/sdumont/setup.sh build-arm   (aloca 1 no GH200 e faz tudo la)

  Os dois fazem pull + pip dentro do job: os nos tem internet. [PARCIAL] quer
  dizer que a imagem existe mas faltam as dependencias, e o comando acima
  completa sem rebaixar o .sif que ja esta pronto.
MSG
}

# ---------------------------------------------------------------- build ------
build() {
  write_reqs
  pull_base pt "${PT_BASE}"
  pull_base tf "${TF_BASE}"
  install_site pt
  install_site tf
  make_ldextra >/dev/null
  echo
  verify pt "import torch,timm;print('pt ok: torch',torch.__version__,'timm',timm.__version__)"
  verify tf "import tensorflow as tf;print('tf ok:',tf.__version__)"
  status
}

# -------------------------------------------------------------- prefetch -----
# Baixa, NO NO DE LOGIN (unico com internet), a imagem base e as wheels de uma
# arquitetura. Depois o build-* instala num no daquela arquitetura, offline.
#   PREFETCH_ARCHS="aarch64"   so o ARM (default: as duas)
prefetch() {
  local archs="${PREFETCH_ARCHS:-x86_64 aarch64}" a
  for a in ${archs}; do
    local sarch pf t
    case "${a}" in aarch64) sarch=arm64 ;; x86_64) sarch=amd64 ;; *) echo "arch ${a}?" >&2; continue ;; esac
    pf="${CACHE_DIR}/${a}"; mkdir -p "${pf}/wheels-pt" "${pf}/wheels-tf"
    echo "==> [${a}] imagens base (${sarch})"
    sing_pull "${pf}/base-pt.sif" "docker://${PT_BASE}" --arch "${sarch}"
    sing_pull "${pf}/base-tf.sif" "docker://${TF_BASE}" --arch "${sarch}"
    ARCH="${a}" CONT_DIR="${pf}" write_reqs
    echo "==> [${a}] wheels"
    for t in pt tf; do
      python3 -m pip download -r "${pf}/reqs-${t}.txt" -d "${pf}/wheels-${t}" \
        --platform "manylinux2014_${a}" --platform any --only-binary=:all: \
        --python-version "${PREFETCH_PY:-312}" 2>&1 | tail -3 \
        || echo "  aviso: alguma wheel nao existe para ${a}; ajuste PREFETCH_PY"
    done
    echo "   [${a}] pronto: $(du -sh "${pf}" | cut -f1)"
  done
}

# O prefetch antigo gravava em _cache/arm64; o atual usa _cache/aarch64 (o nome
# que `uname -m` devolve). Aceita os dois, para nao descartar um download de 8 GB.
prefetch_dir() {
  local a="$1" d
  for d in "${CACHE_DIR}/${a}" \
           $( [[ "${a}" == aarch64 ]] && echo "${CACHE_DIR}/arm64" ) \
           $( [[ "${a}" == x86_64 ]] && echo "${CACHE_DIR}/amd64" ); do
    [[ -f "${d}/base-pt.sif" ]] && { echo "${d}"; return 0; }
  done
  echo "${CACHE_DIR}/${a}"
}

# ------------------------------------------------------- build via SLURM -----
# Aloca um no da arquitetura pedida e roda o build-offline la, para o container
# nascer no MESMO tipo de no onde os treinos vao rodar. Sem salloc interativo.
build_on() {
  local kind="$1"
  # shellcheck source=_partitions.sh
  source "${PROJECT_DIR}/tools/sdumont/_partitions.sh"
  local part="${SD_PART_BUILD:-}"
  if [[ -z "${part}" ]]; then
    sd_detect_partitions >/dev/null 2>&1 || true
    case "${kind}" in gh200) part="${SD_PART_GH200:-}" ;; h100) part="${SD_PART_H100:-}" ;; esac
  fi
  [[ -n "${part}" ]] || { echo "[FAIL] nao achei a fila de ${kind}; exporte SD_PART_BUILD" >&2; exit 1; }

  local a; case "${kind}" in gh200) a=aarch64 ;; h100) a=x86_64 ;; esac

  # Os nos de execucao TEM internet (confirmado por `nettest`: nvcr.io, pypi.org
  # e files.pythonhosted.org respondem). Entao o build roda inteiro no no: pull
  # + pip, sem depender do prefetch. Isso resolve os dois problemas do login, que
  # sao o teto de 30 min e o cgroup de memoria que matava o mksquashfs.
  # Se um dia a rede for cortada, PREFETCH_FIRST=1 volta ao caminho offline.
  local target=build
  if [[ "${PREFETCH_FIRST:-0}" == "1" ]]; then
    target=build-offline
    [[ -f "$(prefetch_dir "${a}")/base-pt.sif" ]] || {
      echo "[FAIL] prefetch de ${a} ausente." >&2
      echo "       rode no login: PREFETCH_ARCHS=${a} $0 prefetch" >&2
      exit 1
    }
  fi

  echo "==> alocando 1 no em ${part} (${kind}/${a}) para o ${target}"
  local acct; acct="$(sd_account_for "${part}")"
  [[ -n "${acct}" ]] && echo "==> conta: ${acct}"
  srun -p "${part}" ${acct:+-A "${acct}"} \
       --nodes=1 --gpus-per-node=1 --cpus-per-gpu="${BUILD_CPUS:-12}" \
       --time="${BUILD_TIME:-02:00:00}" \
       bash -c "cd '${PROJECT_DIR}' && ./tools/sdumont/setup.sh ${target}"
}

build_arm() { build_on gh200; }
build_x86() { build_on h100; }

# Instala as dependencias NO NO em que este script esta rodando, sem rede,
# a partir do que o prefetch baixou para esta arquitetura.
build_offline() {
  local pf; pf="$(prefetch_dir "${ARCH}")"
  [[ -f "${pf}/base-pt.sif" && -f "${pf}/base-tf.sif" ]] || {
    echo "[FAIL] prefetch de ${ARCH} ausente em ${pf}" >&2
    echo "       rode no no de login: PREFETCH_ARCHS=${ARCH} $0 prefetch" >&2
    exit 1
  }
  write_reqs
  local t
  for t in pt tf; do
    if [[ -f "${CONT_DIR}/hcpa-${t}.sif" && "${FORCE:-0}" != "1" ]]; then
      echo "[${ARCH}/${t}] imagem ja existe (FORCE=1 para refazer)"
    else
      echo "[${ARCH}/${t}] instalando base pre-baixada de ${pf}"
      cp -f "${pf}/base-${t}.sif" "${CONT_DIR}/hcpa-${t}.sif"
    fi
  done
  for t in pt tf; do
    install_site "${t}" --no-index --find-links "${pf}/wheels-${t}"
  done
  make_ldextra >/dev/null
  echo
  verify pt "import torch,timm;print('pt ok: torch',torch.__version__,'timm',timm.__version__)"
  verify tf "import tensorflow as tf;print('tf ok:',tf.__version__)"
}

# Descobre se os nos de EXECUCAO alcancam o registro. O manual do LNCC diz que
# nao, mas se alcancarem o pull pode ir para dentro de um job: sem o teto de 30
# min do login e com a memoria do no inteiro, que e onde o squashfs esta morrendo.
nettest() {
  # shellcheck source=_partitions.sh
  source "${PROJECT_DIR}/tools/sdumont/_partitions.sh"
  sd_detect_partitions >/dev/null 2>&1 || true

  # SD_USE_DEV=0 pula direto para as filas de producao.
  # Cadeia de fallback. Para cada arquitetura, tenta a fila *_dev (prioridade
  # alta) e depois a de producao. Pula quem nao tem no ocioso, para nao ficar
  # pendurado na fila, e tambem pula quem REJEITA a submissao -- a gh200_dev
  # devolve AssocMaxNodePerJobLimit mesmo para 1 no, e insistir nela nao leva a
  # lugar nenhum. Se nenhuma fila de nenhuma arquitetura servir, encerra.
  local kind part idle tested=0
  for kind in gh200 h100; do
    local cands=()
    case "${kind}" in
      gh200) cands=($( [[ "${SD_USE_DEV:-1}" == "1" ]] && echo "${SD_PART_DEV_GH200:-}" ) "${SD_PART_GH200:-}") ;;
      h100)  cands=($( [[ "${SD_USE_DEV:-1}" == "1" ]] && echo "${SD_PART_DEV_H100:-}"  ) "${SD_PART_H100:-}")  ;;
    esac

    local ok_arch=0
    for part in "${cands[@]}"; do
      [[ -n "${part}" ]] || continue
      # O teste usa 1 no, sempre: a politica de saida e da PARTICAO, nao do no,
      # e o SLURM escolhe qualquer um dela. A contagem de ociosos nao muda o que
      # e testado, so evita esperar 90s por um job que ja se sabe que ficaria
      # pendente.
      idle="$(sd_idle_in "${part}")"
      if (( idle < 1 )); then
        echo "[${kind}] ${part}: sem no ocioso agora, pulando"
        continue
      fi
      echo "==> [${kind}] testando 1 no de ${part} (${idle} livres no momento)"
      if nettest_one "${part}"; then ok_arch=1; tested=1; break; fi
      echo "    ${part} nao aceitou o job; tentando a proxima fila"
    done
    (( ok_arch )) || echo "[${kind}] nenhuma fila utilizavel agora"
    echo
  done

  if (( tested == 0 )); then
    cat >&2 <<'MSG'
Nenhuma fila de nenhuma arquitetura aceitou o teste agora.
Pode ser ocupacao passageira (tente mais tarde) ou limite de associacao. Para ver:
  sacctmgr show assoc user=$USER format=account,partition,qos,maxnodes,maxjobs,maxsubmit -P
  sacctmgr show qos format=name,maxnodesperjob,maxwall -P | grep -i ict
MSG
    return 1
  fi

  cat <<'MSG'
Leitura do resultado:
  nvcr.io respondendo 401 conta como [ok]: e o registro pedindo token, ou seja,
  a rede chegou la.
  Se der [ok], o pull passa a rodar dentro de um job e o teto de 30 min do no de
  login deixa de importar.
MSG
}

# Um teste numa fila. Retorna 0 se o job rodou, 1 se nao entrou ou foi rejeitado.
nettest_one() {
  local part="$1"
  ( sleep "${NETTEST_WAIT:-90}"
    j="$(squeue --me -h -o '%i %T %j' 2>/dev/null | awk '$2=="PENDING" && $3=="hcpa_net"{print $1; exit}')"
    if [[ -n "${j}" ]]; then
      echo "    nao entrou em ${NETTEST_WAIT:-90}s. motivo:"
      squeue --me -h -o "      %.12i %.12P %.9T %.26r" 2>/dev/null
      scancel "${j}" 2>/dev/null
    fi ) &
  local watchdog=$!

  local rc=0
  local acct; acct="$(sd_account_for "${part}")"
  srun -p "${part}" ${acct:+-A "${acct}"} -J hcpa_net \
       --nodes=1 --gpus-per-node=1 --time=00:05:00 bash -c '
    echo "    no: $(hostname)  arch: $(uname -m)"
    for u in https://nvcr.io/v2/ https://pypi.org/simple/ https://files.pythonhosted.org/; do
      code=$(curl -s -o /dev/null -m 15 -w "%{http_code}" "$u" 2>/dev/null || echo 000)
      case "$code" in
        2*|3*|401) echo "    [ok]   $u  (HTTP $code)" ;;
        *)         echo "    [BLOQ] $u  (HTTP $code)" ;;
      esac
    done' || rc=1

  kill "${watchdog}" 2>/dev/null; wait "${watchdog}" 2>/dev/null || true
  return "${rc}"
}

# Mostra a fila e identifica jobs que NUNCA vao rodar. Isso importa porque o
# projeto tem teto de 100 jobs (RUNNING + PENDING): jobs mortos ocupam vaga e
# seguram as submissoes novas, sem dar nenhum sinal obvio.
#   QOSMinGRES              -> pediu fila de GPU sem pedir GPU; pendente eterno
#   DependencyNeverSatisfied-> depende de algo que ja falhou
#   Dependency              -> depende de um dos acima, entao morre junto
queue() {
  echo "### seus jobs"
  squeue --me -o "%.12i %.12P %.14j %.9T %.26r" 2>/dev/null || { echo "(squeue indisponivel)"; return 1; }
  echo
  local dead; dead="$(squeue --me -h -o '%i %T %r' 2>/dev/null \
    | awk '$2=="PENDING" && ($3=="QOSMinGRES" || $3 ~ /^Dependency/) {print $1}')"
  local n; n="$(printf '%s\n' "${dead}" | grep -c . || true)"
  local total; total="$(squeue --me -h 2>/dev/null | grep -c . || true)"
  echo "total: ${total} job(s); ${n} sem chance de rodar"
  if (( n > 0 )); then
    if [[ "${PURGE:-0}" == "1" ]]; then
      echo "cancelando ${n}..."
      printf '%s\n' "${dead}" | xargs -r scancel
      echo "ok"
    else
      echo "para cancelar so esses:  PURGE=1 $0 queue"
    fi
  fi
}

case "${1:-probe}" in
  probe)         probe ;;
  build)         build ;;
  prefetch)      prefetch ;;
  build-offline) build_offline; status ;;
  build-arm)     build_arm; status ;;
  build-x86)     build_x86; status ;;
  status)        status ;;
  space)         space ;;
  clean)         clean ;;
  nettest)       nettest ;;
  queue)         queue ;;
  *) echo "uso: $0 {probe|build|prefetch|build-arm|build-x86|build-offline|status|space|clean|nettest|queue}" >&2; exit 2 ;;
esac
