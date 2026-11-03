#!/usr/bin/env bash
# Monta weights_cache/ com TODOS os pesos pre-treinados que o treino baixaria da
# internet em tempo de execucao.
#
# Por que isso existe: os nos de execucao do SDumont nao alcancam a internet sem
# liberacao do helpdesk. 7 das 8 abordagens usam InceptionV3 pre-treinado: 5 via
# timm com pretrained=True (pytorch_base, pytorch_opt e os 3 hibridos) e 2 via
# Keras (tensorflow_base e tensorflow_opt). Sem este cache viajando junto no
# rsync, todo job morre na primeira epoca por falha de rede, depois de ja ter
# esperado na fila. O retfound_green nao entra aqui porque usa pretrained=False
# e carrega de retfound_green/weights/retfoundgreen_statedict.pth.
#
# Rode no GPPD (que tem internet) ANTES de puxar do SDumont com fetch.sh.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
DEST="${PROJECT_DIR}/weights_cache"
mkdir -p "${DEST}/hf" "${DEST}/keras/models" "${DEST}/torch"

copy_in() {
  local src="$1" dst="$2" label="$3"
  [[ -e "${src}" ]] || { echo "  [--] ${label}: ausente em ${src}"; return 1; }
  rsync -a --exclude='*.lock' --exclude='xet/logs/' "${src}/" "${dst}/"
  echo "  [ok] ${label}: $(du -sh "${dst}" | cut -f1)"
}

echo "==> cache HuggingFace/timm (InceptionV3 para PyTorch e hibridos)"
copy_in "${PROJECT_DIR}/.cache/hf"        "${DEST}/hf"    "hf (raiz do projeto)" \
  || copy_in "${PROJECT_DIR}/pytorch_opt/.cache/hf" "${DEST}/hf" "hf (pytorch_opt)" \
  || copy_in "${HOME}/.cache/huggingface" "${DEST}/hf"    "hf (home)" \
  || echo "  !! nenhum cache HF encontrado"

echo "==> cache Keras (InceptionV3 ImageNet para TensorFlow)"
copy_in "${PROJECT_DIR}/tensorflow_opt/.keras" "${DEST}/keras" "keras (tensorflow_opt)" || true
for f in "${HOME}"/.keras/models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5; do
  [[ -f "${f}" ]] && cp -n "${f}" "${DEST}/keras/models/" && echo "  [ok] keras (home): $(basename "${f}")"
done

echo "==> cache torch.hub"
[[ -d "${HOME}/.cache/torch/hub" ]] && copy_in "${HOME}/.cache/torch/hub" "${DEST}/torch/hub" "torch.hub" || true

echo
echo "weights_cache pronto: $(du -sh "${DEST}" | cut -f1)"
find "${DEST}" -name '*.safetensors' -o -name '*.h5' -o -name '*.pth' -o -name '*.bin' | sed 's/^/  /'
echo
echo "Confira que o InceptionV3 aparece nas DUAS listas acima (timm e keras)."
echo "Se faltar algum, rode o treino local por 1 epoca para popular o cache e repita."
