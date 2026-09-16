#!/usr/bin/env sh
# Сборка и запуск сервиса контроля качества денситометрии (ТЗ 7.2).
# Проверено на Linux и macOS; нужен только Docker.
#
#   ./run.sh build                      собрать образ
#   ./run.sh predict <папка|архив.zip> [<выход>]   обработать исследования (по умолчанию в ./outputs)
#   ./run.sh serve [<порт>]             веб-интерфейс и HTTP API (по умолчанию 8000)
#   ./run.sh test                       прогнать на тестовых файлах организаторов
set -eu

IMAGE="${DXAQC_IMAGE:-dxa-qc}"
PLATFORM="${DXAQC_PLATFORM:-linux/amd64}"
ROOT="$(cd "$(dirname "$0")" && pwd)"

die() { echo "Ошибка: $*" >&2; exit 1; }
have_image() { docker image inspect "$IMAGE" >/dev/null 2>&1; }
ensure_image() { have_image || { echo "Образ $IMAGE не найден, собираю…"; build; }; }

build() {
    command -v docker >/dev/null 2>&1 || die "Docker не установлен"
    docker build --platform "$PLATFORM" -t "$IMAGE" "$ROOT"
    echo "Готово: образ $IMAGE"
}

predict() {
    [ $# -ge 1 ] || die "укажите папку или архив: ./run.sh predict <папка|архив.zip> [<папка вывода>]"
    if [ -d "$1" ]; then
        IN="$(cd "$1" && pwd)"; TARGET=/data
    elif [ -f "$1" ]; then
        IN="$(cd "$(dirname "$1")" && pwd)"; TARGET="/data/$(basename "$1")"
    else
        die "не найдено: $1"
    fi
    OUT="${2:-$ROOT/outputs}"
    mkdir -p "$OUT"
    OUT="$(cd "$OUT" && pwd)"
    ensure_image
    docker run --rm --platform "$PLATFORM" \
        -v "$IN":/data:ro -v "$OUT":/out \
        "$IMAGE" "$TARGET" --out /out/results.csv --vis /out/vis
    echo "Результаты: $OUT/results.csv (и .xlsx), визуализация: $OUT/results_vis.zip"
}

serve() {
    PORT="${1:-8000}"
    ensure_image
    echo "Веб-интерфейс: http://localhost:$PORT"
    docker run --rm --platform "$PLATFORM" -p "$PORT":8000 \
        --entrypoint uvicorn "$IMAGE" dxaqc.api:app --host 0.0.0.0 --port 8000
}

selftest() {
    [ -d "$ROOT/data/test_sample" ] || die "нет папки data/test_sample с тестовыми файлами"
    predict "$ROOT/data/test_sample" "$ROOT/outputs/selftest"
}

CMD="${1:-}"
[ $# -gt 0 ] && shift || true
case "$CMD" in
    build)   build ;;
    predict) predict "$@" ;;
    serve)   serve "$@" ;;
    test)    selftest ;;
    *)       sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//' ;;
esac
