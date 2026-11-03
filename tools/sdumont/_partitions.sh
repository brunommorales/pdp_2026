#!/usr/bin/env bash
# Descobre sozinho quais filas do SDumont tem H100 e quais tem GH200, e quantos
# nos de cada uma estao livres agora.
#
# Existe porque as filas da expansao (H100 XH3145-H e GH200 XH3515-H) nao estao
# no wiki publico do LNCC, entao os nomes so podem ser lidos da propria maquina.
# Em vez de exigir que voce descubra e exporte SD_PART_H100/SD_PART_GH200 na mao,
# isto le o `sinfo` e resolve.
#
# A classificacao tenta tres fontes, em ordem de confiabilidade:
#   1) o GRES da fila (ex.: gpu:h100:4, gpu:gh200:4)
#   2) as features dos nos (%f), onde alguns sitios registram o modelo
#   3) o nome da propria fila
# Se as tres falharem, a fila fica como desconhecida e voce escolhe na mao.
#
# Uso:
#   source _partitions.sh && sd_detect_partitions      # popula as variaveis
#   ./_partitions.sh                                   # so imprime a tabela
#
# Exporta: SD_PART_H100, SD_PART_GH200 (se ainda nao estiverem definidas)
#          SD_IDLE_H100, SD_IDLE_GH200 (nos ociosos na fila escolhida)
#          SD_GPN_H100,  SD_GPN_GH200  (GPUs por no, para montar a geometria)
#          SD_PART_DEV_H100, SD_PART_DEV_GH200 (filas *_dev, para o smoke)

# Filas as quais o usuario tem acesso. Sem isso, daria para escolher uma fila
# que existe mas que rejeita a submissao com AssociationJobLimit.
sd_allowed_partitions() {
  sacctmgr -nP list user "${USER}" -s format=partition 2>/dev/null \
    | tr ',' '\n' | sed 's/^ *//; s/ *$//' | grep -v '^$' | sort -u
}

# Uma linha por fila: fila|tipo_gpu|gpus_por_no|nos_idle|nos_total
sd_scan_partitions() {
  local allowed; allowed="$(sd_allowed_partitions)"
  sinfo -h -o "%P|%G|%D|%T|%f" 2>/dev/null | awk -F'|' -v allowed="${allowed}" '
    BEGIN {
      n = split(allowed, a, "\n")
      for (i = 1; i <= n; i++) if (a[i] != "") ok[a[i]] = 1
      have_acl = (n > 0 && allowed != "")
    }
    {
      part = $1; sub(/\*$/, "", part)      # tira o * da fila default
      gres = tolower($2); nodes = $3; state = tolower($4); feat = tolower($5)

      if (have_acl && !(part in ok)) next  # fila sem acesso nao serve
      if (gres == "" || gres ~ /^\(null\)$/) next   # fila sem GPU nao interessa

      # O GRES manda. So se ele nao nomear o modelo (ex.: "gpu:4") e que valem
      # as features e, por ultimo, o nome da fila. Sem essa ordem, uma fila
      # chamada "grace" com L40S seria classificada como GH200: "grace" e o nome
      # da CPU Arm, nao da GPU, e as duas coisas aparecem juntas so no GH200.
      model = ""
      if (match(gres, /gpu:[a-z][a-z0-9_-]*:[0-9]+/)) {
        split(substr(gres, RSTART, RLENGTH), g, ":")
        model = g[2]
      }
      hay = (model != "" ? model : feat " " tolower(part))
      type = "?"
      if      (hay ~ /gh200|gracehopper|hopper.?super/) type = "gh200"
      else if (hay ~ /h100/)  type = "h100"
      else if (hay ~ /a100/)  type = "a100"
      else if (hay ~ /l40/)   type = "l40s"
      else if (hay ~ /v100/)  type = "v100"
      else if (model != "")   type = model

      # gpus por no, do gres: gpu:h100:4 ou gpu:4
      gpn = 0
      if (match(gres, /gpu:[a-z0-9_-]*:?[0-9]+/)) {
        tok = substr(gres, RSTART, RLENGTH)
        m = split(tok, p, ":")
        gpn = p[m] + 0
      }

      key = part
      kind[key] = (type != "?" ? type : kind[key])
      per[key]  = (gpn > 0 ? gpn : per[key])
      tot[key] += nodes
      if (state ~ /^idle/) idle[key] += nodes
      # Filas *_dev tem walltime de ~20 min: servem para o smoke, nunca para os
      # 200 epocas. Marcadas para nao serem escolhidas so por estarem mais vazias.
      isdev[key] = (part ~ /(^|[_-])dev([_-]|$)/) ? 1 : 0
    }
    END {
      for (k in tot)
        printf "%s|%s|%d|%d|%d|%d\n", k, (kind[k] ? kind[k] : "?"), per[k], idle[k], tot[k], isdev[k]
    }' | sort -t'|' -k2,2 -k6,6n -k4,4nr
}

# Escolhe, para cada arquitetura, a fila com mais nos ociosos.
sd_detect_partitions() {
  local scan; scan="$(sd_scan_partitions)"
  if [[ -z "${scan}" ]]; then
    echo "[part] sinfo nao retornou nada utilizavel. Defina SD_PART_H100 e SD_PART_GH200 na mao." >&2
    return 1
  fi

  # A ordenacao ja poe as filas de producao antes das *_dev e, dentro de cada
  # grupo, as mais vazias primeiro. Entao a primeira de cada tipo e a escolha.
  local part type per idle tot dev
  while IFS='|' read -r part type per idle tot dev; do
    if [[ "${dev}" == "1" ]]; then
      # Fila de teste guardada POR ARQUITETURA: mandar um job GH200 para a fila
      # dev dos H100 rodaria no .sif da arquitetura errada.
      case "${type}" in
        gh200) [[ -z "${SD_PART_DEV_GH200:-}" ]] && SD_PART_DEV_GH200="${part}" ;;
        h100)  [[ -z "${SD_PART_DEV_H100:-}"  ]] && SD_PART_DEV_H100="${part}"  ;;
      esac
      continue
    fi
    case "${type}" in
      gh200) [[ -z "${SD_PART_GH200:-}" ]] && { SD_PART_GH200="${part}"; SD_IDLE_GH200="${idle}"; SD_GPN_GH200="${per}"; } ;;
      h100)  [[ -z "${SD_PART_H100:-}"  ]] && { SD_PART_H100="${part}";  SD_IDLE_H100="${idle}";  SD_GPN_H100="${per}"; } ;;
    esac
  done <<<"${scan}"

  export SD_PART_GH200 SD_PART_H100 SD_PART_DEV_GH200 SD_PART_DEV_H100 \
         SD_IDLE_GH200 SD_IDLE_H100 SD_GPN_GH200 SD_GPN_H100
  return 0
}

# Walltime maximo da fila, em horas. As filas ict-* do SDumont sao 1-00:00:00
# (24h), NAO as 96h das filas antigas documentadas no wiki. Um job estimado
# acima disso e recusado com PartitionTimeLimit e fica PENDING para sempre.
# Formatos do SLURM: "1-00:00:00", "05:00:00", "00:20:00", "UNLIMITED".
sd_maxwall_h() {
  local part="$1" raw
  raw="$(sacctmgr -nP list user "${USER}" -s format=partition,maxwall 2>/dev/null \
         | awk -F'|' -v p="${part}" '$1==p && $2!="" {print $2; exit}')"
  [[ -n "${raw}" ]] || { echo "${SD_MAXWALL_DEFAULT_H:-24}"; return; }
  [[ "${raw}" == "UNLIMITED" ]] && { echo 96; return; }
  awk -v s="${raw}" 'BEGIN{
    d=0; t=s
    if (index(s,"-")) { split(s,a,"-"); d=a[1]+0; t=a[2] }
    n=split(t,b,":")
    h=(n>=1?b[1]:0)+0; m=(n>=2?b[2]:0)+0
    printf "%d", d*24 + h + (m>=30?1:0)
  }'
}

# Conta (account) do SLURM a usar numa fila. Isto NAO e detalhe: o usuario tem
# uma associacao default `none` com MaxNodes=0/MaxJobs=0/MaxSubmit=0, e sem
# `-A <conta>` o SLURM usa ela e recusa tudo com AssocMaxNodePerJobLimit ou
# AssocMaxSubmitJobLimit -- mesmo com dezenas de nos ociosos. A conta do projeto
# (ex.: avalgpu) e a que tem a associacao com a fila.
sd_account_for() {
  sacctmgr -nP show assoc user="${USER}" format=account,partition 2>/dev/null \
    | awk -F'|' -v p="$1" '$2==p && $1!="none" {print $1; exit}'
}

# Nos ociosos numa fila, agora. Serve para decidir se vale submeter ou se o job
# so ficaria pendente.
sd_idle_in() {
  sinfo -h -p "$1" -o "%D %t" 2>/dev/null \
    | awk '$2 ~ /^idle/ {n += $1} END {print n+0}'
}

# Fila de CPU ARM (sem GPU), se existir. Serve para o build aarch64, que e so
# `pip install` e nao toca GPU nenhuma: usar um no GH200 para isso queimaria
# 100 UA por hora sem necessidade. No SDumont e a lncc-grace, identificada pela
# QOS ict_cpu-arm; o nome tambem entra como pista secundaria.
sd_arm_cpu_partition() {
  sacctmgr -nP list user "${USER}" -s format=partition,qos 2>/dev/null \
    | awk -F'|' '
        $1=="" { next }
        { p=tolower($1); q=tolower($2) }
        q ~ /arm/ || p ~ /(^|[_-])(grace|arm)([_-]|$)/ { print $1; exit }'
}

sd_print_partitions() {
  echo "### filas com GPU as quais ${USER} tem acesso"
  printf '%-24s %-8s %6s %8s %8s\n' FILA GPU GPU/NO LIVRES TOTAL
  printf '%s\n' "------------------------------------------------------------"
  local part type per idle tot dev
  while IFS='|' read -r part type per idle tot dev; do
    printf '%-24s %-8s %6s %8s %8s  %s\n' "${part}" "${type}" "${per:-?}" "${idle}" "${tot}" \
           "$( [[ "${dev}" == "1" ]] && echo "(dev, walltime curto)" )"
  done < <(sd_scan_partitions)
  echo
  sd_detect_partitions || return 1
  echo "escolhido para GH200: ${SD_PART_GH200:-<nenhuma>}  (${SD_IDLE_GH200:-0} nos livres, ${SD_GPN_GH200:-?} gpu/no)"
  echo "escolhido para H100 : ${SD_PART_H100:-<nenhuma>}  (${SD_IDLE_H100:-0} nos livres, ${SD_GPN_H100:-?} gpu/no)"
  [[ -n "${SD_PART_DEV_GH200:-}" ]] && echo "fila de teste GH200 : ${SD_PART_DEV_GH200}"
  [[ -n "${SD_PART_DEV_H100:-}"  ]] && echo "fila de teste H100  : ${SD_PART_DEV_H100}"
  echo
  # O .sif fica no $SCRATCH (compartilhado) e o job escolhe o par pelo `uname -m`
  # do no em que cair. Isso so e seguro se a fila tiver um tipo de GPU so. Se
  # aparecer mais de um, o container certo pode nao existir para parte dos nos.
  echo "### homogeneidade das filas escolhidas"
  local p
  for p in "${SD_PART_GH200:-}" "${SD_PART_H100:-}"; do
    [[ -n "${p}" ]] || continue
    local kinds; kinds="$(sinfo -h -p "${p}" -o "%G" 2>/dev/null | sort -u | tr '\n' ' ')"
    local n; n="$(sinfo -h -p "${p}" -o "%G" 2>/dev/null | sort -u | grep -c .)"
    if (( n <= 1 )); then
      echo "  [ok]   ${p}: um tipo de GPU (${kinds%% })"
    else
      echo "  [ATENCAO] ${p}: ${n} tipos de GPU (${kinds}). Um .sif so pode nao"
      echo "            servir para todos os nos; considere fixar com --constraint."
    fi
  done
  echo
  echo "Para forcar outra fila, exporte antes de submeter:"
  echo "  export SD_PART_GH200=<fila>  SD_PART_H100=<fila>"
}

# Executado direto (nao sourced): so imprime.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  set -uo pipefail
  sd_print_partitions
fi
